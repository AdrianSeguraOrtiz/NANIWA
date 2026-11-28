"""Delegate the selected external inference tools to ANDREA."""
import json
import math
import os
import shutil
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path

from .inputs import network_rows
from .manifest import inference_manifest
from .outputs import output_directory


@dataclass(frozen=True)
class InferenceResult:
    network: Path
    dataset: Path  # ANDREA's frozen manifest for the measurements used in inference.

    def __str__(self):
        return f"Raw network: {self.network}\nDataset manifest: {self.dataset}"


def catalog():
    resource = files("naniwa").joinpath("catalog/catalog.json")
    return json.loads(resource.read_text(encoding="utf-8"))


def _andrea_cli(executable):
    if executable == "andrea" and shutil.which(executable) is None:
        sibling = Path(sys.executable).with_name("andrea")
        if sibling.is_file():
            executable = str(sibling)
    try:
        # Rich abbreviates options in narrow terminals and inserts ANSI codes in color output.
        probe = subprocess.run(
            [str(executable), "infer-network", "execute", "--help"],
            text=True, capture_output=True, env={**os.environ, "COLUMNS": "160", "TERM": "dumb"},
        )
    except FileNotFoundError as exc:
        raise ValueError("Install NANIWA's infer extra or specify the ANDREA executable with --andrea") from exc
    if probe.returncode or "--custom-tools" not in probe.stdout:
        raise ValueError(f"{executable} does not expose ANDREA infer-network execute --custom-tools")
    # ANDREA 0.1.0 predates output profiles and the expanded custom-tool schema.
    return str(executable), "--output-profile" in probe.stdout


def _save(path, payload):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
        stream.write("\n")


def _completed_run(out, selected):
    reports = list(out.glob("*/run_report.json"))
    if len(reports) != 1:
        raise ValueError(f"Expected one ANDREA run report under {out}, found {len(reports)}")
    report_path = reports[0]
    report = json.loads(report_path.read_text(encoding="utf-8"))
    results = report.get("tools", {})
    expected = set(selected)
    if report.get("status") != "executed" or results.get("failed") or results.get("skipped"):
        raise ValueError(f"Incomplete ANDREA inference: {report_path}")
    if set(results.get("selected", [])) != expected or set(results.get("completed", [])) != expected:
        raise ValueError(f"ANDREA did not complete the requested tool selection: {report_path}")
    origins = results.get("tool_origins", {})
    if any(origins.get(tool) != "custom" for tool in selected):
        raise ValueError(f"ANDREA did not run every selected tool as an external image: {report_path}")
    outputs = report.get("outputs", {})
    path = outputs.get("merged_network_raw")
    if not isinstance(path, str) or not path.strip():
        raise ValueError(f"ANDREA omitted the raw merged network from {report_path}")
    source = report_path.parent / path
    observed = Counter(tool for tool, _, _, _ in network_rows(source))
    if set(observed) != expected or dict(observed) != outputs.get("rows_per_tool"):
        raise ValueError(f"Raw CSV coverage differs from ANDREA's per-tool row counts or selection: {report_path}")
    snapshot = report.get("inputs", {}).get("dataset_manifest_path")
    if not isinstance(snapshot, str) or not snapshot.strip() or not (report_path.parent / snapshot).is_file():
        raise ValueError(f"Missing frozen dataset manifest referenced by {report_path}")
    return InferenceResult(source, report_path.parent / snapshot)


def _run_andrea(dataset, out, tools, andrea, max_cores, max_ram_gb):
    panel = catalog()
    selected = list(panel if tools is None else tools)
    if not selected or len(selected) != len(set(selected)) or set(selected).difference(panel):
        raise ValueError(f"Choose distinct tools from: {', '.join(panel)}")
    executable, current = _andrea_cli(andrea)
    descriptors, runs = [], []
    for tool in selected:
        spec = panel[tool]
        external = {"run_id": tool, "execution_mode": "global",
                    "name": spec["name"], "docker_image": spec["docker_image"], "extra_inputs": spec["extra_inputs"]}
        if current:
            external.update(outputs=spec["outputs"], runtime_resources=spec["runtime_resources"])
        descriptors.append(external)
        runs.append({"run_id": tool, "tool_id": f"custom_{tool}",
                     "execution": {"mode": "global"}, "params": spec["params"]})
    out.mkdir(parents=True, exist_ok=True)
    custom_path, params_path = out / "custom_tools.json", out / "tools_params.json"
    _save(custom_path, {"tools": descriptors})
    _save(params_path, {"runs": runs})
    command = [executable, "infer-network", "execute", "--dataset-manifest", str(dataset.resolve()),
               "--custom-tools", str(custom_path), "--tools-params", str(params_path), "--output-dir", str(out)]
    if current:
        command += ["--output-profile", "canonical"]
    if max_cores is not None:
        command += ["--max-cores", str(max_cores)]
    if max_ram_gb is not None:
        command += ["--max-ram-gb", str(max_ram_gb)]
    result = subprocess.run(command)
    if result.returncode:
        raise ValueError(f"ANDREA exited with code {result.returncode}. Run files: {out}")
    return _completed_run(out, selected)


def run_inference(dataset, out, tools=None, *, andrea="andrea", max_cores=None, max_ram_gb=None,
                  expression=None, samples=None, regulators=None):
    """Run the selected panel and identify its raw network and frozen input."""
    destination = output_directory(Path(out).resolve())
    if max_cores is not None and (not isinstance(max_cores, int) or max_cores < 1):
        raise ValueError("--max-cores must be a positive integer")
    if max_ram_gb is not None and (not math.isfinite(max_ram_gb) or max_ram_gb <= 0):
        raise ValueError("--max-ram-gb must be finite and positive")
    with inference_manifest(dataset, expression, samples, regulators) as manifest:
        return _run_andrea(manifest, destination, tools, andrea, max_cores, max_ram_gb)


def infer(dataset, out, tools=None, *, andrea="andrea", max_cores=None, max_ram_gb=None):
    """Run inference and return ANDREA's original merged CSV."""
    return run_inference(dataset, out, tools, andrea=andrea, max_cores=max_cores, max_ram_gb=max_ram_gb).network
