"""Pinned native NIMCE inference on independent observed bulk time courses."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import subprocess
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

UPSTREAM_COMMIT = "095fdb20aaa4748f31c2d728bec5920f395156be"
UPSTREAM_ARCHIVE_SHA256 = (
    "d352b634915662e7e197eda9609db1435cffa36c98292a2da77496877c844f83"
)
UPSTREAM_SOURCES = {
    "NIMCE.m": "dbc0ffe45f6f9d4a6ba1418a0acbf8b61b43e7abeaf823c209bd595bd1a5f64f",
    "createTime_lag.m": "19cfa38ea6632f01227a3e4fb6efe210be6362a805747b79d2c7a1daf94df507",
    "creatG.m": "c0fb75c7628f361c33526459d7108d5e96891e8a8b57da2eb9e68d635e2cb0c9",
    "edgereduce.m": "d9e6d0621698f6fb4f6902022efd33f8ba755e0cbfd6f1fa5782fd468eaeda9f",
    "creatGcmi.m": "08c2fd2028d156f7b26b46bd54df20220f1367e8af7bcd3cc1160ee76580dfa5",
    "horderkernelcmi.m": "85930b86fb1808f2cb2fca8eaf5b6181f76000988635f582a8dd1a6334af29aa",
    "coupla_transform.m": "bddfe17e348fb62e4b35a1599fbd3030b6c626d537ae39898f07f15bb08d25aa",
    "ssqd.m": "45566dfbf0881e88be93d72862090a60b4045af807675d7d4ce7256dd3b7382c",
}
PROGRESS_PREFIX = "ANDREA_NIMCE_PROGRESS\t"


def patched_source(name: str, raw: bytes) -> bytes:
    """Observe completed native loops without changing their numerical work."""
    if (
        name not in UPSTREAM_SOURCES
        or hashlib.sha256(raw).hexdigest() != UPSTREAM_SOURCES[name]
    ):
        raise ValueError(f"Unexpected NIMCE source: pinned SHA-256 required for {name}")
    patches = {
        "creatG.m": (
            b"    end\r\nend\r\n\r\n",
            b"    end\r\n    fprintf(1, 'ANDREA_NIMCE_PROGRESS\\tte\\t%d\\t%d\\n', i, targetnum); fflush(1);\r\nend\r\n\r\n",
        ),
        "creatGcmi.m": (
            b"      end\r\nend   \r\n",
            b"      end\r\n      fprintf(1, 'ANDREA_NIMCE_PROGRESS\\tce\\t%d\\t%d\\n', i, col); fflush(1);\r\nend   \r\n",
        ),
        "edgereduce.m": (
            b"[~,Ctrs] = kmeans(G_1d,2,'Start','uniform','Replicates',3);",
            b"fprintf(1, 'ANDREA_NIMCE_PROGRESS\\tclustering\\n'); fflush(1);\r\n[~,Ctrs] = kmeans(G_1d,2,'Start','uniform','Replicates',3);",
        ),
    }
    if name in patches:
        old, new = patches[name]
        if raw.count(old) != 1:
            raise ValueError(f"NIMCE observation patch context changed: {name}")
        return raw.replace(old, new)
    return raw


def prepare_upstream(*, output_dir: Path, archive_path: Path | None = None) -> dict:
    """Fetch a pinned public archive and install only eight native functions."""
    raw = (
        Path(archive_path).read_bytes()
        if archive_path is not None
        else urllib.request.urlopen(
            f"https://codeload.github.com/CSUBioGroup/NIMCE/tar.gz/{UPSTREAM_COMMIT}",
            timeout=60,
        ).read()
    )
    if hashlib.sha256(raw).hexdigest() != UPSTREAM_ARCHIVE_SHA256:
        raise ValueError("Unexpected NIMCE archive: pinned SHA-256 is required")
    sources = {}
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as archive:
        for member in archive.getmembers():
            parts = Path(member.name).parts
            if len(parts) != 2 or parts[1] not in UPSTREAM_SOURCES:
                continue
            if not member.isfile() or parts[1] in sources:
                raise ValueError("Expected one regular file per pinned native source")
            sources[parts[1]] = archive.extractfile(member).read()
    if set(sources) != set(UPSTREAM_SOURCES):
        raise ValueError("The pinned NIMCE archive lacks required native sources")
    patched = {name: patched_source(name, source) for name, source in sources.items()}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    originals = output_dir / "original"
    originals.mkdir(exist_ok=True)
    receipt = {
        "upstream_commit": UPSTREAM_COMMIT,
        "archive_sha256": UPSTREAM_ARCHIVE_SHA256,
        "patches": "Completed target/source loop observations and clustering phase only; numerical source unchanged.",
        "sources": {},
    }
    for name in sorted(sources):
        (originals / name).write_bytes(sources[name])
        (output_dir / name).write_bytes(patched[name])
        receipt["sources"][name] = {
            "original_sha256": UPSTREAM_SOURCES[name],
            "installed_sha256": hashlib.sha256(patched[name]).hexdigest(),
        }
    (output_dir / "nimce-source-receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )
    return receipt


if __name__ == "__main__" and sys.argv[1:2] == ["--prepare-upstream"]:
    parser = argparse.ArgumentParser(
        description="Prepare verified native NIMCE source; no inference"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--archive", dest="archive_path", type=Path)
    arguments = parser.parse_args(sys.argv[2:])
    print(json.dumps(prepare_upstream(**vars(arguments)), sort_keys=True))
    raise SystemExit(0)


import numpy as np

from _column_metadata import load_column_descriptors, summarize_temporal_metadata
from _run_tool_common import (
    load_execution_mode,
    load_params,
    validate_runtime_inputs,
    write_progress,
)

DEFAULTS = {"window_num": 3, "time_lag": 2, "h": 0.45, "lamda2": 0.04, "seed": 0}
MAX_THREADS = 128


def parse_params(raw: dict) -> dict:
    if not isinstance(raw, dict):
        raise ValueError("NIMCE parameters must be a JSON object")
    unknown = set(raw) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown NIMCE parameters: {sorted(unknown)}")
    params = {**DEFAULTS, **raw}
    for name, minimum, maximum in (
        ("window_num", 3, None),
        ("time_lag", 2, None),
        ("seed", 0, 2**32 - 1),
    ):
        value = params[name]
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < minimum
            or maximum is not None
            and value > maximum
        ):
            raise ValueError(
                f"{name} must be an integer from {minimum}"
                + (f" to {maximum}" if maximum is not None else "")
            )
    if params["time_lag"] >= params["window_num"]:
        raise ValueError("time_lag must be smaller than window_num")
    for name, strictly_positive in (("h", True), ("lamda2", False)):
        value = params[name]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
            or strictly_positive
            and value == 0
        ):
            raise ValueError(
                f"{name} must be a finite {'positive' if strictly_positive else 'nonnegative'} number"
            )
    return params


def parameter_warnings(params: dict) -> list[str]:
    warnings = []
    if params["window_num"] > 3:
        warnings.append(
            "The native algorithm overwrites earlier window-pair results; only the last pair is returned and fewer pooled observations are used as window_num grows."
        )
    if params["time_lag"] > 2:
        warnings.append(
            "Native transfer entropy uses the requested time_lag, while causation-entropy pruning retains the original adjacent two lag positions; this is not a uniform general-order lag model."
        )
    return warnings


def read_matrix(path: Path):
    with Path(path).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle, delimiter="\t"))
    if not rows or len(rows[0]) < 2:
        raise ValueError("NIMCE requires a genes-by-observations expression TSV")
    headings, columns = rows[0], rows[0][1:]
    if any(not item or item != item.strip() for item in columns) or len(
        set(columns)
    ) != len(columns):
        raise ValueError("Expression columns must be unique exact nonempty identifiers")
    genes, values = [], []
    for number, row in enumerate(rows[1:], 2):
        if not row:
            continue
        if len(row) != len(headings):
            raise ValueError(f"Malformed expression row {number}")
        if not row[0] or row[0] != row[0].strip() or row[0] in genes:
            raise ValueError(
                "Expression genes must be unique exact nonempty identifiers"
            )
        genes.append(row[0])
        values.append([float(value) for value in row[1:]])
    matrix = np.asarray(values, dtype=np.float64)
    if len(genes) < 2 or not np.isfinite(matrix).all():
        raise ValueError(
            "NIMCE requires at least two genes and finite expression values"
        )
    return genes, columns, matrix


def prepare_data(expression_path: Path, extra_dir: Path, raw_params: dict) -> dict:
    params = parse_params(raw_params)
    genes, columns, expression = read_matrix(expression_path)
    descriptors = load_column_descriptors(
        Path(extra_dir) / "column_descriptors.tsv", columns
    )
    temporal = summarize_temporal_metadata(descriptors)
    if temporal["sampling_design"] != "longitudinal":
        raise ValueError("NIMCE requires longitudinal observed bulk time series")
    groups, observations, stationary = {}, set(), []
    for position, column in enumerate(columns):
        row = descriptors[column]
        if row["observation_type"] == "steady_state":
            stationary.append(column)
            continue
        if row["observation_id"] in observations:
            raise ValueError(
                "NIMCE does not implicitly average time-series technical replicates"
            )
        observations.add(row["observation_id"])
        groups.setdefault(row["series_id"], []).append(
            (row["timepoint"], position, row)
        )
    series, blocks, expected_length, common_interval = [], [], None, None
    for series_id, entries in sorted(groups.items()):
        entries.sort(key=lambda value: value[0])
        times = [entry[0] for entry in entries]
        conditions = {entry[2]["condition"] for entry in entries}
        replicates = {entry[2]["replicate"] for entry in entries}
        if len(conditions) != 1 or len(replicates) != 1:
            raise ValueError(
                "One series must identify one condition and biological replicate"
            )
        if len(times) < params["window_num"]:
            raise ValueError(
                "Each series needs at least window_num observed timepoints"
            )
        intervals = np.diff(times)
        interval = float(intervals[0])
        if (
            interval <= 0
            or not math.isfinite(interval)
            or not all(
                math.isfinite(float(value))
                and value > 0
                and math.isclose(float(value), interval, rel_tol=1e-9, abs_tol=1e-12)
                for value in intervals
            )
        ):
            raise ValueError(
                "NIMCE requires strictly increasing regular observed times"
            )
        if expected_length is not None and len(times) != expected_length:
            raise ValueError(
                "NIMCE requires the same number of timepoints in every series"
            )
        if common_interval is not None and not math.isclose(
            interval, common_interval, rel_tol=1e-9, abs_tol=1e-12
        ):
            raise ValueError("NIMCE requires a common sampling interval across series")
        expected_length, common_interval = len(times), interval
        positions = [entry[1] for entry in entries]
        blocks.append(expression[:, positions].T.copy())
        series.append(
            {
                "series_id": series_id,
                "condition": next(iter(conditions)),
                "replicate": next(iter(replicates)),
                "columns": [columns[index] for index in positions],
                "timepoints": times,
            }
        )
    if not blocks:
        raise ValueError("NIMCE requires at least one time series")
    pooled_rows = len(blocks) * (expected_length - params["window_num"] + 1)
    if pooled_rows < 2:
        raise ValueError(
            "NIMCE requires at least two pooled lag-window observations for its copula transform"
        )
    native_data = np.concatenate(blocks)
    # The native copula assigns distinct ordinal ranks even to a constant
    # vector. Validate each pooled shifted block it actually receives, rather
    # than allowing those invented ranks to become regulatory evidence. Scale
    # each gene before subtraction so the criterion follows floating-point
    # precision, not a fixed threshold in expression units (and cannot overflow).
    window_rows = expected_length - params["window_num"] + 1
    used_positions = {
        position
        for start in range(params["window_num"] - params["time_lag"])
        for position in (start, start + 1, start + 2, start + params["time_lag"])
    }
    for position in sorted(used_positions):
        # createTime_lag numbers shifted blocks from latest to earliest.
        offset = params["window_num"] - 1 - position
        shifted = np.concatenate(
            [block[offset : offset + window_rows] for block in blocks]
        )
        minima, maxima = shifted.min(axis=0), shifted.max(axis=0)
        scales = np.maximum(np.abs(minima), np.abs(maxima))
        spread = np.zeros(len(genes), dtype=np.float64)
        nonzero = scales > 0
        spread[nonzero] = (
            maxima[nonzero] / scales[nonzero]
            - minima[nonzero] / scales[nonzero]
        )
        invalid = [
            gene
            for gene, value in zip(genes, spread)
            if value <= np.finfo(np.float64).eps
        ]
        if invalid:
            raise ValueError(
                "NIMCE requires expression variation beyond floating-point "
                "precision in every pooled lag window; constant or numerically "
                f"degenerate genes in lag-window position {position + 1}: {invalid}"
            )
    ties = {
        gene: int(len(native_data) - len(np.unique(native_data[:, index])))
        for index, gene in enumerate(genes)
    }
    return {
        "genes": genes,
        "columns": columns,
        "descriptors": descriptors,
        "temporal": temporal,
        "series": series,
        "timepoints_per_series": expected_length,
        "sampling_interval": common_interval,
        "native_data": native_data,
        "pooled_window_rows": pooled_rows,
        "unused_steady_state_columns": stationary,
        "repeated_time_series_values_per_gene": ties,
    }


def _write_json(path: Path, payload) -> None:
    Path(path).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def octave_string(value: str | Path) -> str:
    """Quote one literal; identifiers and paths never become Octave code."""
    value = str(value)
    if any(character in value for character in ("\x00", "\r", "\n")):
        raise ValueError("Octave paths cannot contain NUL or line breaks")
    return "'" + value.replace("'", "''") + "'"


def write_driver(
    path: Path,
    *,
    upstream_dir: Path,
    input_path: Path,
    matrix_path: Path,
    resources_path: Path,
    params: dict,
    timepoints_per_series: int,
    threads: int,
) -> None:
    source = f"""% Generated NIMCE invocation; expression/gene IDs are never evaluated.
warning('off', 'Octave:shadowed-function');
pkg load statistics;
addpath({octave_string(upstream_dir)});
rng({params['seed']}, 'twister');
data = load('-ascii', {octave_string(input_path)});
statistics_packages = pkg('list', 'statistics');
resources = struct('pid', getpid(), 'octave_version', version(), 'statistics_version', statistics_packages{{1}}.version, 'blas_config', __octave_config_info__('BLAS_LIBS'), 'blas_version', version('-blas'), 'requested_threads', {threads}, 'OPENBLAS_NUM_THREADS', getenv('OPENBLAS_NUM_THREADS'), 'OMP_NUM_THREADS', getenv('OMP_NUM_THREADS'), 'MKL_NUM_THREADS', getenv('MKL_NUM_THREADS'), 'cpu_status', fileread('/proc/self/status'));
fid = fopen({octave_string(resources_path)}, 'w'); fprintf(fid, '%s\\n', jsonencode(resources)); fclose(fid);
G = NIMCE(data, {timepoints_per_series}, {params['window_num']}, {params['time_lag']}, {float(params['h'])!r}, {float(params['lamda2'])!r});
dlmwrite({octave_string(matrix_path)}, G, '\\t', 'precision', 17);
"""
    Path(path).write_text(source, encoding="utf-8")


def verify_installed_sources(upstream_dir: Path) -> dict:
    receipt = json.loads(
        (Path(upstream_dir) / "nimce-source-receipt.json").read_text(encoding="utf-8")
    )
    if (
        receipt.get("upstream_commit") != UPSTREAM_COMMIT
        or receipt.get("archive_sha256") != UPSTREAM_ARCHIVE_SHA256
        or set(receipt.get("sources", {})) != set(UPSTREAM_SOURCES)
    ):
        raise ValueError(
            "The installed NIMCE source receipt does not match the pinned implementation"
        )
    for name, expected in UPSTREAM_SOURCES.items():
        original = (Path(upstream_dir) / "original" / name).read_bytes()
        installed = (Path(upstream_dir) / name).read_bytes()
        patched = patched_source(name, original)
        if installed != patched or receipt["sources"][name] != {
            "original_sha256": expected,
            "installed_sha256": hashlib.sha256(patched).hexdigest(),
        }:
            raise ValueError(
                f"The installed NIMCE source differs from its verified observation patch: {name}"
            )
    return receipt


def export_network(path: Path, matrix, genes: list[str]) -> int:
    values = np.asarray(matrix, dtype=np.float64)
    if values.shape != (len(genes), len(genes)) or not np.isfinite(values).all():
        raise ValueError(
            "Native NIMCE must return one finite source-by-target score per gene pair"
        )
    edges = [
        (source, target, float(values[i, j]))
        for i, source in enumerate(genes)
        for j, target in enumerate(genes)
        if i != j and values[i, j] > 0
    ]
    edges.sort(key=lambda value: (-value[2], value[0], value[1]))
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["source", "target", "score", "sign", "evidence", "context"])
        writer.writerows(
            (source, target, repr(score), "?", "association", "global")
            for source, target, score in edges
        )
    return len(edges)


def _resource_observation(pid: int) -> dict:
    observation = {"native_pid": pid, "observed_monotonic_ns": time.monotonic_ns()}
    proc = Path("/proc") / str(pid)
    try:
        observation["process_thread_count"] = len(list((proc / "task").iterdir()))
        observation["cpu_affinity"] = sorted(os.sched_getaffinity(pid))
        observation["loaded_blas_libraries"] = sorted(
            {
                line.split()[-1]
                for line in (proc / "maps").read_text().splitlines()
                if "libopenblas" in line or "libblas" in line
            }
        )
        observation["process_thread_count_basis"] = (
            "Observed native process tasks, including library workers; not an OpenBLAS API thread-count query."
        )
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        observation["observation_unavailable"] = (
            "The native process exited before its /proc snapshot completed."
        )
    return observation


def run(
    *,
    input_path,
    params_path,
    extra_dir,
    output_dir,
    threads: int,
    upstream_dir=Path("/app/nimce"),
    octave_executable="octave-cli",
):
    input_path, params_path, extra_dir, output_dir, upstream_dir = map(
        Path, (input_path, params_path, extra_dir, output_dir, upstream_dir)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    provenance = output_dir / "provenance"
    provenance.mkdir(exist_ok=True)
    completed, total, last_percent, warnings = 0, None, 0, []

    def progress(phase, message, percent, *, status="running", error=None):
        nonlocal last_percent
        last_percent = max(last_percent, percent)
        write_progress(
            output_dir / "progress.json",
            status=status,
            phase=phase,
            message=message,
            percent=last_percent,
            completed=completed,
            total=total,
            error=error,
            warnings=warnings,
        )
        event = json.loads((output_dir / "progress.json").read_text())
        event["observed_monotonic_ns"] = time.monotonic_ns()
        with (provenance / "progress-events.jsonl").open(
            "a", encoding="utf-8"
        ) as handle:
            handle.write(json.dumps(event, allow_nan=False) + "\n")

    progress("validate", "Validating NIMCE temporal inputs", 0)
    try:
        validate_runtime_inputs(
            input_path=input_path,
            params_path=params_path,
            extra_dir=extra_dir,
            threads=threads,
            required_paths=[extra_dir / "column_descriptors.tsv"],
        )
        if threads > MAX_THREADS:
            raise ValueError(
                f"NIMCE supports at most {MAX_THREADS} threads, the pinned OpenBLAS backend's intrinsic MAX_THREADS bound"
            )
        load_execution_mode(params_path, supported_modes={"global"})
        params = parse_params(load_params(params_path))
        warnings = parameter_warnings(params)
        data = prepare_data(input_path, extra_dir, params)
        if any(data["repeated_time_series_values_per_gene"].values()):
            warnings.append(
                "Tied expression values retain the original native sort-based copula ranks; NIMCE does not apply tie averaging. Counts are recorded in provenance/resolved.json."
            )
        receipt = verify_installed_sources(upstream_dir)
        genes = data["genes"]
        calls = params["window_num"] - params["time_lag"]
        total = 2 * calls * len(genes)
        native_input = provenance / "native-timeseries.tsv"
        np.savetxt(native_input, data["native_data"], delimiter="\t", fmt="%.17g")
        driver, matrix_path = (
            provenance / "run-nimce.m",
            provenance / "native-score-matrix.tsv",
        )
        write_driver(
            driver,
            upstream_dir=upstream_dir,
            input_path=native_input,
            matrix_path=matrix_path,
            resources_path=provenance / "resources-native.json",
            params=params,
            timepoints_per_series=data["timepoints_per_series"],
            threads=threads,
        )
        resolved = {key: value for key, value in data.items() if key != "native_data"}
        resolved.update(
            {
                "params": params,
                "threads": threads,
                "native_pair_calls_per_phase": calls,
                "native_scoring_orientation": "source rows, target columns",
                "warnings": warnings,
                "input_sha256": {
                    path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in (
                        input_path,
                        params_path,
                        extra_dir / "column_descriptors.tsv",
                    )
                },
                "native_transfer_entropy_sample_offsets": {
                    "target": params["time_lag"],
                    "regulator": [params["time_lag"] - 1, 0],
                    "target_history": [params["time_lag"] - 1, 0],
                },
                "native_pruning_sample_offsets": {
                    "target": params["time_lag"],
                    "regulator": [params["time_lag"] - 1, params["time_lag"] - 2],
                    "mediator": [params["time_lag"] - 1, params["time_lag"] - 2],
                },
                "offset_reference": "Forward sample offsets from the start of each length-window_num native window in the final overwritten pair. Multiply by sampling_interval for physical duration; these are method inputs, not inferred edge-specific delays.",
                "steady_state_handling": "Excluded before native inference; independent time courses remain separate when native windows are formed.",
                "normalization": "Unchanged native per-variable rank copula transform; no wrapper scaling, imputation or averaging.",
                "fixed_clustering": {"k": 2, "start": "uniform", "replicates": 3},
                "source_receipt": receipt,
            }
        )
        _write_json(provenance / "resolved.json", resolved)
        environment = os.environ.copy()
        environment.update(
            {
                name: str(threads)
                for name in (
                    "OPENBLAS_NUM_THREADS",
                    "OMP_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "BLIS_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS",
                    "NUMEXPR_NUM_THREADS",
                )
            }
        )
        _write_json(
            provenance / "session.json",
            {
                "upstream_commit": UPSTREAM_COMMIT,
                "python": sys.version,
                "numpy": np.__version__,
                "source_receipt": receipt,
                "octave_executable": octave_executable,
                "native_thread_environment": {
                    name: environment[name]
                    for name in (
                        "OPENBLAS_NUM_THREADS",
                        "OMP_NUM_THREADS",
                        "MKL_NUM_THREADS",
                        "BLIS_NUM_THREADS",
                    )
                },
            },
        )
        progress("inference", "Computing native kernel transfer entropy", 5)
        counts = {"te": 0, "ce": 0}
        with (provenance / "native.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                [
                    octave_executable,
                    "--quiet",
                    "--no-gui",
                    "--no-history",
                    str(driver.resolve()),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=environment,
                bufsize=1,
            )
            try:
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    if not line.startswith(PROGRESS_PREFIX):
                        continue
                    fields = line.strip().split("\t")
                    if fields[1:] == ["clustering"]:
                        if counts["te"] != calls * len(genes):
                            raise ValueError(
                                "Native clustering began before transfer-entropy observations completed"
                            )
                        progress(
                            "clustering",
                            "Clustering native edge confidence before pruning",
                            5 + 85 * completed // total,
                        )
                        continue
                    if len(fields) != 4 or fields[1] not in counts:
                        raise ValueError("Invalid native NIMCE completion observation")
                    phase, local, size = fields[1], int(fields[2]), int(fields[3])
                    if (
                        size != len(genes)
                        or local != counts[phase] % len(genes) + 1
                        or counts[phase] >= calls * len(genes)
                        or phase == "ce"
                        and counts["te"] != calls * len(genes)
                    ):
                        raise ValueError(
                            "Inconsistent native NIMCE completion observations"
                        )
                    counts[phase] += 1
                    completed += 1
                    with (provenance / "resources-inference.jsonl").open(
                        "a", encoding="utf-8"
                    ) as resources:
                        resources.write(
                            json.dumps(
                                {
                                    **_resource_observation(process.pid),
                                    "phase": phase,
                                    "completed": completed,
                                }
                            )
                            + "\n"
                        )
                    progress(
                        "inference" if phase == "te" else "pruning",
                        f"Completed native {'target' if phase == 'te' else 'source'} {local}/{len(genes)} in {'transfer entropy' if phase == 'te' else 'causation entropy'}",
                        5 + 85 * completed // total,
                    )
                returncode = process.wait()
            except BaseException:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                raise
        if returncode:
            tail = (
                (provenance / "native.log")
                .read_text(encoding="utf-8")
                .splitlines()[-20:]
            )
            raise RuntimeError(
                f"Native NIMCE exited with status {returncode}: " + "\n".join(tail)
            )
        if completed != total:
            raise ValueError(
                "Native NIMCE returned without every completed target/source observation"
            )
        progress("export", "Exporting native NIMCE directed confidence", 95)
        matrix = np.loadtxt(matrix_path, delimiter="\t", ndmin=2)
        resolved["network_edge_count"] = export_network(
            output_dir / "network.csv", matrix, genes
        )
        _write_json(provenance / "resolved.json", resolved)
        progress("done", "NIMCE network exported", 100, status="completed")
        return resolved
    except Exception as error:
        progress(
            "failed", "NIMCE failed", last_percent, status="failed", error=str(error)
        )
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--params", required=True, type=Path)
    parser.add_argument("--extra", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--threads", required=True, type=int)
    arguments = parser.parse_args()
    run(
        input_path=arguments.input,
        params_path=arguments.params,
        extra_dir=arguments.extra,
        output_dir=arguments.output_dir,
        threads=arguments.threads,
    )


if __name__ == "__main__":
    main()
