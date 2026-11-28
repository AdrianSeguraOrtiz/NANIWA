import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from naniwa.consensus.scale import minmax
from naniwa.inputs import load_dataset, load_networks, write_merged_networks


class InputTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.expr = self.root / "expression.tsv"
        self.expr.write_text("gene\ts3\ts2\ts4\ts1\n"
                             "01\t30\t20\t40\t10\n"
                             "02\t-3\t-2\t-4\t-1\n"
                             "03\t3\t2\t4\t1\n")
        self.desc = self.root / "samples.tsv"
        self.desc.write_text("column\tseries_id\ttimepoint\n"
                             "s2\ta\t2\ns1\ta\t0\ns4\tb\t10\ns3\tb\t4\n")
        self.regs = self.root / "regulators.txt"
        self.regs.write_text("02\n01\n")
        self.manifest = self.root / "dataset-manifest.json"
        self.manifest.write_text(json.dumps({
            "dataset": {"expression_matrix": self.expr.name},
            "extras": {"column_descriptors": self.desc.name, "tf_list": self.regs.name},
        }))
        self.dataset = load_dataset(self.manifest)

    def network(self, name, columns, rows):
        path = self.root / name
        with path.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(columns)
            writer.writerows(rows)
        return path

    def test_manifest_and_standalone_align_samples_by_name_and_time(self):
        direct = load_dataset(expression=self.expr, samples=self.desc, regulators=self.regs)
        self.assertEqual(direct.genes, ("01", "02", "03"))
        self.assertEqual(direct.regulators, ("02", "01"))
        np.testing.assert_array_equal(direct.mask, [[True, False], [False, True], [True, True]])
        np.testing.assert_array_equal(direct.times, [[0, 2], [4, 10]])
        np.testing.assert_array_equal(direct.X[:, :, 0], [[10, 20], [30, 40]])
        np.testing.assert_array_equal(direct.X, self.dataset.X)

    def test_standalone_paths_must_be_complete_and_exclusive(self):
        with self.assertRaisesRegex(ValueError, "together"):
            load_dataset(expression=self.expr)
        with self.assertRaisesRegex(ValueError, "either"):
            load_dataset(self.manifest, regulators=self.regs)

    def test_duplicate_time_mismatched_sample_and_unequal_lengths(self):
        original = self.desc.read_text()
        for old, new, why in [("s1\ta\t0", "s1\ta\t2", "unique timepoints"),
                              ("s2\ta", "missing\ta", "match"),
                              ("s3\tb", "s3\ta", "at least two observations")]:
            with self.subTest(new=new):
                self.desc.write_text(original.replace(old, new))
                with self.assertRaisesRegex(ValueError, why):
                    load_dataset(self.manifest)

    def test_consensus_honors_declared_temporal_metadata(self):
        original = self.desc.read_text().splitlines()
        annotations = ["time_unit", "sampling_design", "condition", "replicate", "observation_type", "observation_id"]
        rows = [line.split("\t") + ["h", "longitudinal", "control", line.split("\t")[1], "time_series", f"obs{i}"]
                for i, line in enumerate(original[1:])]
        header = original[0].split("\t") + annotations

        def load():
            with self.desc.open("w", newline="") as stream:
                csv.writer(stream, delimiter="\t").writerows([header, *rows])
            return load_dataset(self.manifest)

        np.testing.assert_array_equal(load().X, self.dataset.X)
        for field, substitute, message in [
            ("time_unit", "min", "time_unit"), ("sampling_design", "independent_samples", "longitudinal"),
            ("condition", "treatment", "condition"), ("replicate", "other", "replicate"),
            ("observation_type", "steady_state", "time_series"), ("observation_id", "obs1", "observation_id"),
        ]:
            with self.subTest(field=field):
                idx = header.index(field)
                previous, rows[0][idx] = rows[0][idx], substitute
                with self.assertRaisesRegex(ValueError, message):
                    load()
                rows[0][idx] = previous

    def test_unknown_regulator_or_duplicate_gene_fails(self):
        self.regs.write_text("02\nstranger\n")
        with self.assertRaisesRegex(ValueError, "stranger"):
            load_dataset(self.manifest)
        self.regs.write_text("02\n")
        self.expr.write_text(self.expr.read_text().replace("03\t", "02\t"))
        with self.assertRaisesRegex(ValueError, "Duplicate gene"):
            load_dataset(self.manifest)

    def test_one_regulator_keeps_eligible_edges_on_other_targets(self):
        self.regs.write_text("02\n")
        data = load_dataset(self.manifest)
        np.testing.assert_array_equal(data.mask[:, 0], [True, False, True])
        path = self.network("network.csv", ["source", "target", "score"], [["02", "01", 2]])
        _, raw = load_networks(data, individual={"tool": path})
        np.testing.assert_array_equal(minmax(raw, data.mask)[:, data.mask], [[1, 0]])

    def test_both_formats_align_filter_and_reduce_duplicate_edges_identically(self):
        columns = ("source", "target", "score")
        swing = [("01", "02", 3), ("01", "02", 8), ("02", "02", 99), ("03", "01", 100)]
        dyngenie = [("01", "03", 4), ("02", "03", 8)]
        parts = {"swing__01": self.network("s.csv", columns, swing),
                 "dyngenie3": self.network("d.csv", columns, dyngenie)}
        merged = self.network("together.csv", (*columns, "tool_id", "context"),
                              [(*row, "swing__01", "global") for row in reversed(swing)] +
                              [(*row, "dyngenie3", "global") for row in dyngenie])
        tools, raw = load_networks(self.dataset, individual=parts)
        joint_tools, joint_raw = load_networks(self.dataset, merged=merged)
        self.assertEqual(tools, ("dyngenie3", "swing__01"))
        self.assertEqual(joint_tools, tools)
        np.testing.assert_array_equal(joint_raw, raw)
        np.testing.assert_array_equal(raw[:, self.dataset.mask], [[0, 0, 8, 4], [0, 8, 0, 0]])
        self.assertTrue(np.isnan(raw[:, ~self.dataset.mask]).all())
        converted = self.root / "converted.csv"
        write_merged_networks(converted, self.dataset, tools, raw)
        restored_tools, restored = load_networks(self.dataset, merged=converted)
        self.assertEqual(restored_tools, tools)
        np.testing.assert_array_equal(restored, raw)

    def test_global_minmax_uses_full_nonzero_dense_range_per_tool(self):
        mask = self.dataset.mask
        raw = np.full((2, *mask.shape), np.nan)
        raw[:, mask] = [[2, 3, 6, 4], [104, 106, 100, 102]]
        scaled = minmax(raw, mask)
        np.testing.assert_allclose(scaled[:, mask], [[0, 0.25, 1, 0.5], [2/3, 1, 0, 1/3]])
        np.testing.assert_array_equal(raw[0, mask], [2, 3, 6, 4])
        self.assertTrue(np.isnan(scaled[:, ~mask]).all())

    def test_bad_scores_context_and_conflicting_identity_fail_even_on_excluded_edges(self):
        columns = ("source", "target", "score", "context", "tool_id")
        for score, context, name, why in [
            (float("inf"), "global", "t", "finite"), (-1, "global", "t", "nonnegative"),
            (1, "time_5", "t", "global"), (1, "global", "other", "tool_id")
        ]:
            with self.subTest(score=score, context=context, name=name):
                path = self.network("bad.csv", columns, [("01", "01", score, context, name)])
                with self.assertRaisesRegex(ValueError, why):
                    load_networks(self.dataset, individual={"t": path})

    def test_raw_constant_networks_survive_sparse_export(self):
        columns = ("source", "target", "score")
        edges = [(self.dataset.regulators[r], self.dataset.genes[t], 4.0)
                 for t, r in np.argwhere(self.dataset.mask)]
        parts = {"constant": self.network("constant.csv", columns, edges),
                 "zero": self.network("zero.csv", columns, [("01", "03", 0)])}
        tools, raw = load_networks(self.dataset, individual=parts, require_variation=False)
        np.testing.assert_array_equal(raw[:, self.dataset.mask], [[4] * 4, [0] * 4])
        joint = self.root / "joint.csv"
        write_merged_networks(joint, self.dataset, tools, raw)
        restored_tools, restored = load_networks(self.dataset, merged=joint, require_variation=False)
        self.assertEqual(restored_tools, tools)
        np.testing.assert_array_equal(restored, raw)
        with self.assertRaisesRegex(ValueError, "constant.*no score variation"):
            load_networks(self.dataset, merged=joint)

    def test_source_must_report_an_eligible_edge(self):
        for rows in ([], [("01", "01", 9)]):
            with self.subTest(rows=rows):
                path = self.network("empty.csv", ("source", "target", "score"), rows)
                with self.assertRaisesRegex(ValueError, "empty-tool.*no eligible"):
                    load_networks(self.dataset, individual={"empty-tool": path}, require_variation=False)
