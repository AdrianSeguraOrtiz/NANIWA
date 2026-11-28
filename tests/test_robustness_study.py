import csv
import hashlib
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np

from experimentation import study


class RobustnessTruthTests(unittest.TestCase):
    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.data = SimpleNamespace(genes=("A", "B", "C"), regulators=("B", "A"),
                                    mask=np.array([[1, 0], [0, 1], [1, 1]], dtype=bool))
        self.row = dict(id="test", dataset_manifest=str(self.root / "dataset.json"),
                        ground_truth_manifest=str(self.root / "truth.json"))
        (self.root / "tfs").write_text("A\nB\n")
        (self.root / "genes").write_text("C\nA\nB\n")
        (self.root / "networks.csv").write_text("source,target,score,context\nA,C,1,global\nB,A,0,global\n")
        (self.root / "expression").write_text("gene\ts1\ts2\nA\t2\t1\nB\t3\t4\nC\t2\t5\n")
        (self.root / "samples").write_text("column\tseries_id\ttimepoint\ns1\t1\t0\ns2\t1\t1\n")
        view = dict(dataset=dict(spec=dict(id="test"), expression_matrix="expression"),
                    extras=dict(tf_list="tfs", column_descriptors="samples"))
        (self.root / "dataset.json").write_text(json.dumps(view))
        files = {"expression": "expression", "extra:tf_list": "tfs", "extra:column_descriptors": "samples"}
        hashes = {k: hashlib.sha256((self.root / v).read_bytes()).hexdigest() for k, v in files.items()}
        digest = hashlib.sha256(json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        self.manifest = dict(dataset_id="test", dataset_fingerprint=dict(algorithm="sha256", value=digest),
                             candidate_space=dict(sources="tfs", targets="genes", allow_self_edges=False),
                             outputs=dict(gene_universe="genes", networks="networks.csv"))
        (self.root / "truth.json").write_text(json.dumps(self.manifest))

    def test_labels_follow_dataset_axes_and_truth_assets_enter_robustness_receipts(self):
        labels = study.robustness_truth(self.row, self.data)
        np.testing.assert_array_equal(labels, [[0, 0], [0, 0], [0, 1]])
        original = study.input_hashes(self.row, with_truth=True)
        with (self.root / "networks.csv").open("a") as stream:
            stream.write("A,B,1,global\n")
        corrected = study.input_hashes(self.row, with_truth=True)
        self.assertNotEqual(original["truth_network"], corrected["truth_network"])
        self.assertNotIn("truth_network", study.input_hashes(self.row))

    def test_reject_wrong_view_content_and_truth_universe(self):
        for change, message in [
            (dict(dataset_id="other"), "different datasets"),
            (dict(dataset_fingerprint=dict(algorithm="sha256", value="0" * 64)), "fingerprint"),
            (dict(candidate_space=self.manifest["candidate_space"] | dict(targets="tfs")), "targets"),
        ]:
            with self.subTest(change=change):
                (self.root / "truth.json").write_text(json.dumps(self.manifest | change))
                with self.assertRaisesRegex(ValueError, message):
                    study.robustness_truth(self.row, self.data)

    def test_reject_self_edges_and_duplicate_directed_labels(self):
        for invalid in ("A,A,1,global", "A,C,0,global"):
            with self.subTest(invalid=invalid):
                with (self.root / "networks.csv").open("w", newline="") as stream:
                    writer = csv.writer(stream)
                    writer.writerows([["source", "target", "score", "context"],
                                      ["A", "C", "1", "global"], invalid.split(",")])
                with self.assertRaisesRegex(ValueError, "invalid directed ground-truth edge"):
                    study.robustness_truth(self.row, self.data)
