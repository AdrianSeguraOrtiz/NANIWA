"""Exercise the CLI input, output and failure contracts."""
import csv
import json
import shutil
import subprocess
import sys
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from naniwa.cli import main
from naniwa.consensus import STAGES, ConsensusOptions
from naniwa.inference import InferenceResult


def write_dataset(root):
    (root / "expression.tsv").write_text("gene\ts0\ts1\ts2\nA\t1\t3\t2\nB\t3\t2\t1\nC\t2\t1\t3\n")
    (root / "samples.tsv").write_text("column\tseries_id\ttimepoint\ns2\tS\t2\ns0\tS\t0\ns1\tS\t1\n")
    (root / "regulators.txt").write_text("A\nB\n")


class CommandTests(unittest.TestCase):
    def test_infer_and_run_use_the_same_standalone_dataset(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_dataset(root)
            samples = root / "samples.tsv"
            lines = samples.read_text().splitlines()
            samples.write_text(lines[0] + "\ttime_unit\tsampling_design\n" +
                               "".join(line + "\th\tlongitudinal\n" for line in lines[1:]))
            common = ["--expression", str(root / "expression.tsv"), "--samples", str(samples),
                      "--regulators", str(root / "regulators.txt")]
            manifests = []

            def infer_from_manifest(dataset, out, *options):
                manifests.append(json.loads(dataset.read_text()))
                self.assertTrue((dataset.parent / manifests[-1]["extras"]["column_descriptors"]).is_file())
                snapshot = out / "input" / dataset.name
                shutil.copytree(dataset.parent, snapshot.parent)
                raw = out / "merged_network_raw.csv"
                raw.write_text("source,target,score,tool_id\nA,B,2,swing\nB,C,1,swing\nA,C,3,nimce\n")
                return InferenceResult(raw, snapshot)

            with patch("naniwa.inference._run_andrea", new=infer_from_manifest):
                main(["infer", *common, "--tools", "swing", "nimce", "--out", str(root / "infer")])
                main(["run", *common, "--tools", "swing", "nimce", "--out", str(root / "full")])
            main(["consensus", *common, "--stage", "naniwa", "--networks", str(root / "infer/merged_network_raw.csv"),
                  "--out", str(root / "separate")])
            self.assertEqual(manifests[0], manifests[1])
            for name in ("consensus.csv", "consensus_options.json"):
                self.assertEqual((root / "full/consensus" / name).read_bytes(), (root / "separate" / name).read_bytes())
            recorded = json.loads((root / "separate/consensus_options.json").read_text())
            self.assertEqual(recorded["options"], {"stage": "naniwa", "vote": "families",
                                                   "evidence": "joint", "intensity": "global"})

    def test_stage_and_component_arguments_publish_identical_results(self):
        with TemporaryDirectory() as tmp, redirect_stdout(StringIO()):
            root = Path(tmp)
            write_dataset(root)
            network = root / "panel.csv"
            network.write_text("source,target,score,tool_id\nA,B,2,first\nB,C,1,first\nA,C,1,second\n")
            common = ["consensus", "--expression", str(root / "expression.tsv"), "--samples", str(root / "samples.tsv"),
                      "--regulators", str(root / "regulators.txt"), "--networks", str(network)]
            for stage in STAGES:
                params = ConsensusOptions(stage).parameters()
                args = [arg for name in ("vote", "evidence", "intensity") for arg in ("--" + name, params[name])]
                by_name, by_components = root / stage, root / (stage + "-components")
                main([*common, "--stage", stage, "--out", str(by_name)])
                main([*common, *args, "--out", str(by_components)])
                for filename in ("consensus.csv", "consensus_options.json"):
                    self.assertEqual((by_name / filename).read_bytes(), (by_components / filename).read_bytes(), stage)
                recorded = json.loads((by_components / "consensus_options.json").read_text())
                self.assertEqual(recorded["options"], params)

            custom = root / "custom"
            main([*common, "--vote", "all", "--evidence", "marginal", "--out", str(custom)])
            recorded = json.loads((custom / "consensus_options.json").read_text())["options"]
            self.assertEqual(recorded, {"stage": None, "vote": "all", "evidence": "marginal", "intensity": "global"})
            preset = root / "preset"
            main([*common, "--stage", "mean_minmax", "--evidence", "marginal", "--out", str(preset)])
            for filename in ("consensus.csv", "consensus_options.json"):
                self.assertEqual((custom / filename).read_bytes(), (preset / filename).read_bytes())

    def test_incompatible_evidence_and_intensity_fail_before_inference(self):
        with patch("naniwa.cli.run") as run:
            for evidence, intensity in (("none", "global"), ("none", "full"), ("joint", "none"), ("marginal", "none")):
                with self.subTest(evidence=evidence, intensity=intensity), redirect_stderr(StringIO()) as errors:
                    with self.assertRaises(SystemExit) as failure:
                        main(["run", "--dataset", "study.json", "--out", "results",
                              "--evidence", evidence, "--intensity", intensity])
                    self.assertEqual(failure.exception.code, 2)
                    self.assertIn("--intensity none with --evidence none", errors.getvalue())
            run.assert_not_called()
            selections = [
                (["--vote", "all"], ConsensusOptions("mean_minmax")),
                (["--vote", "distinct"], ConsensusOptions("exact_mean")),
                (["--evidence", "marginal"], ConsensusOptions("marginal_global")),
                (["--intensity", "full"], ConsensusOptions("joint_full")),
                (["--intensity", "none"], ConsensusOptions("family_mean")),
                (["--stage", "naniwa", "--evidence", "none"], ConsensusOptions("family_mean")),
                (["--stage", "mean_minmax", "--intensity", "full"],
                 ConsensusOptions(vote="all", evidence="joint", intensity="full")),
            ]
            for flags, expected in selections:
                with self.subTest(flags=flags):
                    main(["run", "--dataset", "study.json", "--out", "results", *flags])
                    self.assertEqual(run.call_args.kwargs["options"], expected)

    def test_individual_and_joint_are_interchangeable(self):
        cli = Path(sys.executable).parent / "naniwa"
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_dataset(root)
            (root / "first.csv").write_text("source,target,score\nB,C,4\nA,B,1\n")
            (root / "second.csv").write_text("source,target,score\nA,C,2\nA,B,5\n")
            common = [str(cli), "consensus", "--expression", "expression.tsv", "--samples", "samples.tsv",
                      "--regulators", "regulators.txt"]

            def run(*args):
                return subprocess.run([*common, *args], cwd=root, text=True, capture_output=True)

            first = run("--network", "b=first.csv", "--network", "a=second.csv", "--out", "first")
            self.assertEqual(first.returncode, 0, first.stderr)
            second = run("--networks", "first/networks.csv", "--out", "second")
            self.assertEqual(second.returncode, 0, second.stderr)
            self.assertEqual((root / "first/consensus.csv").read_bytes(), (root / "second/consensus.csv").read_bytes())
            with (root / "second/consensus.csv").open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual({(r["source"], r["target"]) for r in rows},
                             {("A", "B"), ("A", "C"), ("B", "A"), ("B", "C")})
            self.assertTrue(all(0 <= float(r["score"]) <= 1 for r in rows))
            self.assertEqual(rows, sorted(rows, key=lambda r: (-float(r["score"]), r["source"], r["target"])))
            baseline = run("--networks", "first/networks.csv", "--stage", "mean_minmax", "--out", "mean")
            self.assertEqual(baseline.returncode, 0, baseline.stderr)
            with (root / "mean/consensus.csv").open(newline="") as stream:
                self.assertEqual({(r["source"], r["target"]): float(r["score"]) for r in csv.DictReader(stream)},
                                 {("A", "B"): 0.625, ("A", "C"): 0.2, ("B", "A"): 0, ("B", "C"): 0.5})
            failed = run("--network", "b=first.csv", "--network", "b=second.csv", "--out", "bad")
            self.assertEqual(failed.returncode, 2)
            self.assertIn("Duplicate tool ID", failed.stderr)
            self.assertFalse((root / "bad").exists())

    def test_run_rejects_unknown_stage_before_inference(self):
        with redirect_stderr(StringIO()) as errors, self.assertRaises(SystemExit) as failure:
            main(["run", "--dataset", "study.json", "--out", "results", "--stage", "unknown"])
        self.assertEqual(failure.exception.code, 2)
        self.assertIn("--stage", errors.getvalue())

    def test_run_preserves_inference_if_consensus_fails(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_dataset(root)
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({
                "dataset": {"expression_matrix": "expression.tsv"},
                "extras": {"column_descriptors": "samples.tsv", "tf_list": "regulators.txt"},
            }))
            out = root / "run"
            inference_dir = out / "inference"
            raw = inference_dir / "andrea-created-directory" / "native.csv"

            def fake_infer(dataset, out, tools, andrea, max_cores, max_ram_gb):
                self.assertEqual((dataset, out, tools), (manifest, inference_dir, ["swing"]))
                self.assertEqual((andrea, max_cores, max_ram_gb), ("custom-andrea", 2, 4.5))
                raw.parent.mkdir(parents=True)
                raw.write_text("source,target,score,tool_id\nA,B,0,swing\n")
                # ANDREA has already frozen the inputs. Moving the original
                # manifest must not affect which measurements consensus uses.
                snapshot = manifest.rename(root / "frozen.json")
                return InferenceResult(raw, snapshot)

            errors = StringIO()
            with (
                patch("naniwa.inference._run_andrea", side_effect=fake_infer),
                redirect_stderr(errors),
                self.assertRaises(SystemExit) as failure,
            ):
                main(["run", "--dataset", str(manifest), "--out", str(out), "--tools", "swing",
                      "--andrea", "custom-andrea", "--max-cores", "2", "--max-ram-gb", "4.5"])
            self.assertEqual(failure.exception.code, 2)
            self.assertIn("no score variation", errors.getvalue())
            self.assertTrue(raw.is_file())
            self.assertFalse((out / "consensus" / "consensus.csv").exists())
