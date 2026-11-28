import csv
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from naniwa.manifest import write_manifest


class ManifestTests(unittest.TestCase):
    def setUp(self):
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.expression = self.root / "measurements.tsv"
        self.expression.write_text("gene\ts0\ts1\ts2\nA\t1\t3\t2\nB\t4\t2\t6\n")
        self.samples = self.root / "samples.tsv"
        self.regulators = self.root / "regulators.txt"
        self.regulators.write_text("B\nA\n")
        self.bundle = self.root / "bundle"
        self.bundle.mkdir()

    def write(self):
        return write_manifest(self.bundle, self.expression, self.samples, self.regulators)

    def test_compact_samples_produce_a_portable_andrea_dataset(self):
        self.samples.write_text(
            "column\tseries_id\ttimepoint\ttime_unit\tsampling_design\n"
            "s2\tP1\t12\thour\tlongitudinal\n"
            "s0\tP1\t0\thour\tlongitudinal\n"
            "s1\tP1\t6\thour\tlongitudinal\n"
        )
        path = self.write()
        destination = self.root / "relocated"
        self.bundle.rename(destination)
        manifest = json.loads((destination / path.name).read_text())
        spec = manifest["dataset"]["spec"]
        self.assertEqual((manifest["schema_version"], spec["schema_version"]), ("1.0", "1.0"))
        self.assertTrue(spec["id"])
        self.assertEqual(spec["expression"], {
            "genes": 2, "columns": 3, "column_kind": "timepoints", "expression_profile": "bulk",
        })
        self.assertEqual(spec["organism"], {"taxonomic_group": "unknown", "ncbi_taxon_id": None})
        for key, original in [(manifest["dataset"]["expression_matrix"], self.expression),
                              (manifest["extras"]["tf_list"], self.regulators)]:
            content = original.read_bytes()
            original.unlink()
            self.assertFalse(Path(key).is_absolute())
            self.assertEqual((destination / key).read_bytes(), content)
        self.samples.unlink()
        descriptors = Path(manifest["extras"]["column_descriptors"])
        self.assertFalse(descriptors.is_absolute())
        with (destination / descriptors).open(newline="") as stream:
            rows = list(csv.DictReader(stream, delimiter="\t"))
        expected_times = {"s0": "0", "s1": "6", "s2": "12"}
        self.assertCountEqual([row["column"] for row in rows], expected_times)
        for row in rows:
            column = row["column"]
            self.assertEqual(row, {
                "column": column, "series_id": "P1", "timepoint": expected_times[column],
                "time_unit": "hour", "sampling_design": "longitudinal", "observation_type": "time_series",
                "condition": "default", "replicate": "P1", "observation_id": column, "technical_replicate": "1",
            })

    def test_full_descriptors_preserve_metadata_and_extra_columns(self):
        self.samples.write_bytes(
            b"column\tobservation_type\tseries_id\tcondition\treplicate\tobservation_id\ttechnical_replicate"
            b"\ttimepoint\ttime_unit\tsampling_design\tbatch\r\n"
            + b"".join(
                f"s{i}\ttime_series\tP1\tcontrol\tR7\tobs{i}\t2\t{i}\thour\tlongitudinal\tA-1\r\n".encode()
                for i in range(3)
            )
        )
        path = self.write()
        descriptors = json.loads(path.read_text())["extras"]["column_descriptors"]
        self.assertEqual((path.parent / descriptors).read_bytes(), self.samples.read_bytes())

    def test_inference_requires_time_unit_and_sampling_design(self):
        self.samples.write_text("column\tseries_id\ttimepoint\ns0\tP1\t0\ns1\tP1\t1\ns2\tP1\t2\n")
        with self.assertRaises(ValueError) as failure:
            self.write()
        self.assertIn("time_unit", str(failure.exception))
        self.assertIn("sampling_design", str(failure.exception))

    def test_manifest_scan_rejects_nonfinite_values_at_the_original_path(self):
        self.expression.write_text(self.expression.read_text().replace("B\t4", "B\tnan"))
        with self.assertRaisesRegex(ValueError, "expression values must be finite") as failure:
            self.write()
        self.assertIn(str(self.expression), str(failure.exception))
