import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from naniwa.docker.greynet import adapter


class GreyNetSeriesTests(unittest.TestCase):
    def test_partition_preserves_columns_and_absolute_times(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            matrix, descriptors = root / "expression.tsv", root / "descriptors.tsv"
            matrix.write_text("gene\tb1\ta0\tss\tb0\na\t0.25\t5\t9\t6\n")
            descriptors.write_text(
                "column\tobservation_type\tseries_id\ttimepoint\n"
                "ss\tsteady_state\t\t\nb0\ttime_series\tB\t1.0\n"
                "a0\ttime_series\tA\t2.00\nb1\ttime_series\tB\t4.00\n"
            )
            with descriptors.open() as file:
                metadata = {row["column"]: row for row in csv.DictReader(file, delimiter="\t")}
            # The image supplies validation. Exercise the materialization on its result.
            validator = Mock()
            validator.load_column_descriptors.return_value = metadata
            with patch.dict(sys.modules, {"_column_metadata": validator}):
                first, second = adapter.partition_inputs(matrix, descriptors, root)
            self.assertEqual((first / "expression.tsv").read_text(), "gene\ta0\na\t5\n")
            self.assertEqual((second / "expression.tsv").read_text(), "gene\tb1\tb0\na\t0.25\t6\n")
            self.assertEqual(
                (second / "extra/column_descriptors.tsv").read_text().splitlines()[1:],
                ["b1\ttime_series\tB\t4.00", "b0\ttime_series\tB\t1.0"],
            )

    def test_mean_counts_a_missing_edge_and_an_empty_network_as_zero(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = []
            for index, scores in enumerate(([("s", "a", 0.9), ("s", "b", 0.3)], [("s", "b", 0.6)], [])):
                path = root / str(index)
                with path.open("w", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow(adapter.FIELDS)
                    writer.writerows((*edge, "?", "association", "global") for edge in scores)
                paths.append(path)
            adapter.aggregate(paths, root / "out.csv")
            with (root / "out.csv").open() as file:
                rows = list(csv.DictReader(file))
            self.assertEqual([r["target"] for r in rows], ["a", "b"])
            self.assertEqual([float(r["score"]) for r in rows], [0.9 / 3, (0.3 + 0.6) / 3])
            self.assertTrue(all(r["evidence"] == "andrea_series_agg_mean_effect" for r in rows))

    def test_native_failure_leaves_no_partial_parent_network(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            network = output / "network.csv"
            network.write_text("previous")
            children = [output / "a", output / "b"]
            failure = subprocess.CalledProcessError(1, "/app/run_tool.sh")
            with patch.object(adapter, "partition_inputs", return_value=children), \
                    patch.object(subprocess, "run", side_effect=[None, failure]) as native:
                with self.assertRaises(subprocess.CalledProcessError):
                    adapter.run(output / "expression.tsv", output / "params.json", output, output, 3)
                self.assertEqual(native.call_count, 2)
                for call in native.call_args_list:
                    arguments = call.args[0]
                    self.assertEqual(arguments[0], "/app/run_tool.sh")
                    self.assertEqual(arguments[arguments.index("--params") + 1], str(output / "params.json"))
                    self.assertEqual(arguments[arguments.index("--threads") + 1], "3")
            self.assertFalse(network.exists())
