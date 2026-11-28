import json
import os
import subprocess
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from naniwa.inference import InferenceResult, _completed_run, catalog, infer


class InferenceTests(unittest.TestCase):
    def setUp(self):
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.manifest = self.root / "dataset manifest.json"
        self.manifest.write_text("{}")
        self.selected = ["dyngenie3", "swing"]

    def test_external_contracts(self):
        panel = catalog()
        for modern in (False, True):
            out = self.root / str(modern)
            help_text = "--custom-tools" + (" --output-profile" if modern else "")
            responses = [subprocess.CompletedProcess([], 0, help_text), subprocess.CompletedProcess([], 0)]
            with self.subTest(modern=modern), patch.dict(os.environ, COLUMNS="40", FORCE_COLOR="1"), \
                    patch("subprocess.run", side_effect=responses) as run, \
                    patch("naniwa.inference._completed_run") as completed:
                result = infer(self.manifest, out, self.selected, andrea="andrea-test", max_cores=2, max_ram_gb=8)
            completed.assert_called_once_with(out, self.selected)
            self.assertIs(result, completed.return_value.network)
            probe, execute = run.call_args_list
            self.assertEqual(probe.args[0], ["andrea-test", "infer-network", "execute", "--help"])
            self.assertEqual(probe.kwargs["env"]["TERM"], "dumb")
            self.assertGreaterEqual(int(probe.kwargs["env"]["COLUMNS"]), 120)
            custom_path, params_path = out / "custom_tools.json", out / "tools_params.json"
            expected = ["andrea-test", "infer-network", "execute", "--dataset-manifest", str(self.manifest),
                        "--custom-tools", str(custom_path), "--tools-params", str(params_path), "--output-dir", str(out)]
            if modern:
                expected += ["--output-profile", "canonical"]
            self.assertEqual(execute.args[0], expected + ["--max-cores", "2", "--max-ram-gb", "8"])
            fields = ["name", "docker_image", "extra_inputs"] + (["outputs", "runtime_resources"] if modern else [])
            descriptors = [{"run_id": t, "execution_mode": "global", **{k: panel[t][k] for k in fields}}
                           for t in self.selected]
            self.assertEqual(json.loads(custom_path.read_text()), {"tools": descriptors})
            runs = [{"run_id": t, "tool_id": f"custom_{t}", "execution": {"mode": "global"},
                     "params": panel[t]["params"]} for t in self.selected]
            self.assertEqual(json.loads(params_path.read_text()), {"runs": runs})

    def test_verify_report_and_csv_before_returning_network(self):
        root = self.root / "dataset_001"
        root.mkdir()
        raw = root / "merged_network_raw.csv"
        raw.write_text("tool_id,source,target,score\ndyngenie3,A,B,1\nswing,B,A,2\n")
        tools = {"selected": self.selected, "completed": self.selected, "tool_origins": dict.fromkeys(self.selected, "custom")}
        outputs = {"merged_network_raw": raw.name, "rows_per_tool": dict.fromkeys(self.selected, 1)}
        snapshot = root / "inputs" / "frozen-dataset.json"
        snapshot.parent.mkdir()
        snapshot.write_text("{}")
        report = {"status": "executed", "tools": tools, "outputs": outputs,
                  "inputs": {"dataset_manifest_path": str(snapshot.relative_to(root))}}
        report_file = root / "run_report.json"
        report_file.write_text(json.dumps(report))
        self.assertEqual(_completed_run(self.root, self.selected), InferenceResult(raw, snapshot))
        valid_csv = raw.read_text()
        for invalid, message in [("swing,B", "number of fields"), ("swing,B,A,nan", "finite")]:
            raw.write_text(valid_csv.replace("swing,B,A,2", invalid))
            with self.subTest(row=invalid), self.assertRaisesRegex(ValueError, message):
                _completed_run(self.root, self.selected)
        raw.write_text(valid_csv)
        for tools_change, outputs_change in [
            ({"failed": {"swing": "timeout"}}, {}),
            ({"completed": ["dyngenie3"]}, {}),
            ({"tool_origins": dict.fromkeys(self.selected, "catalog")}, {}),
            ({}, {"rows_per_tool": {"dyngenie3": 1, "swing": 0}}),
        ]:
            with self.subTest(tools_change=tools_change, outputs_change=outputs_change):
                report_file.write_text(json.dumps({**report, "tools": {**tools, **tools_change},
                                                   "outputs": {**outputs, **outputs_change}}))
                with self.assertRaises(ValueError):
                    _completed_run(self.root, self.selected)

    def test_nonzero_exit_and_reuse_preserve_diagnostics(self):
        out = self.root / "failed run"
        responses = [subprocess.CompletedProcess([], 0, "--custom-tools"), subprocess.CompletedProcess([], 2)]
        with patch("subprocess.run", side_effect=responses), self.assertRaisesRegex(ValueError, "exited with code 2"):
            infer(self.manifest, out, self.selected)
        before = (out / "tools_params.json").read_bytes()
        with self.assertRaisesRegex(ValueError, "empty output directory"):
            infer(self.manifest, out, self.selected)
        self.assertEqual((out / "tools_params.json").read_bytes(), before)

    def test_invalid_planner_budget_fails_before_launch(self):
        with patch("subprocess.run") as command:
            for options in ({"max_cores": 0}, {"max_ram_gb": -1}, {"max_ram_gb": float("nan")},
                            {"max_ram_gb": float("inf")}):
                with self.subTest(options=options), self.assertRaisesRegex(ValueError, "--max-"):
                    infer(self.manifest, self.root / "output", **options)
            command.assert_not_called()
