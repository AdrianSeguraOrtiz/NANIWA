import unittest

import numpy as np

from experimentation.robustness_diagnostics import SourceDiagnostics
from experimentation.robustness_panels import Panel


class SourceDiagnosticsTests(unittest.TestCase):
    def test_tie_groups_fractional_top1_and_sequential_coverage(self):
        mask = np.array([[1, 1, 1], [1, 1, 0], [1, 1, 0], [0, 0, 0]], dtype=bool)
        truth = np.array([[1, 0, 1], [0, 1, 0], [0, 0, 0], [0, 0, 0]], dtype=bool)
        a = np.array([[2, 2, 0], [3, 4, 8], [0, 1, 9], [1, 2, 3]], dtype=float)
        a[~mask] = np.nan
        tools = ("a", "tied")
        raw = np.stack([a, np.zeros_like(a)])
        diagnostics = SourceDiagnostics(tools, raw, mask, truth)
        clean = list(diagnostics.rows(Panel("clean", tools, raw)))
        self.assertEqual([r["scope"] for r in clean], ["all", "all"])
        self.assertEqual([r["source"] for r in clean], list(tools))
        first, tied = clean
        self.assertEqual(first["n_informative_targets"], 2)
        self.assertEqual(first["n_unique_true_top1"], 1)
        self.assertEqual(first["n_new_unique_true_top1"], 1)
        self.assertAlmostEqual(first["aupr"], (1 + 2/4 + 3/7) / 3)
        self.assertAlmostEqual(first["top1_by_target"], 0.75)
        self.assertAlmostEqual(tied["aupr"], 3/7)
        self.assertAlmostEqual(tied["top1_by_target"], (2/3 + 1/2) / 2)
        self.assertEqual(tied["n_unique_true_top1"], 0)
        self.assertIsNone(tied["mean_error_rank_correlation_previous"])

        source = a.copy()
        source[0, 0] = 3
        panel = Panel("copies", (*tools, "s", "s_copy"), np.stack([*raw, source, source]), repeat=1)
        new, copy = list(diagnostics.rows(panel))[-2:]
        self.assertEqual(new["n_unique_true_top1"], 2)
        self.assertEqual(new["n_new_unique_true_top1"], 1)
        self.assertEqual(copy["n_unique_true_top1"], 2)
        self.assertEqual(copy["n_new_unique_true_top1"], 0)
        self.assertEqual(copy["n_error_correlations_previous_added"], 1)
        self.assertAlmostEqual(copy["mean_error_rank_correlation_previous_added"], 1)

    def test_error_rank_correlation_controls_for_target_and_truth(self):
        mask = np.ones((1, 4), dtype=bool)
        truth = np.array([[True, True, False, False]])
        raw = np.array([[[4, 3, 2, 1]], [[4, 4, 1, 1]]])
        diagnostics = SourceDiagnostics(("a", "constant-within-class"), raw, mask, truth)
        # The first added source swaps the classes but retains their internal order.
        # The second reverses both internal orders. One real tool has no residual ranks.
        sources = np.concatenate((raw, [[[1, 0, 3, 2]], [[3, 4, 1, 2]]]))
        panel = Panel("contrasts", ("a", "constant-within-class", "b", "c"), sources)
        b, c = list(diagnostics.rows(panel))[-2:]
        self.assertAlmostEqual(b["mean_error_rank_correlation_previous"], 1)
        self.assertEqual(b["n_error_correlations_previous"], 1)
        self.assertAlmostEqual(c["mean_error_rank_correlation_previous"], -1)
        self.assertAlmostEqual(c["mean_error_rank_correlation_previous_added"], -1)
        self.assertEqual(c["n_error_correlations_previous"], 2)

    def test_specialist_scopes_restrict_candidates_and_informative_targets(self):
        mask = np.ones((2, 3), dtype=bool)
        truth = np.array([[True, False, False], [False, False, False]])
        raw = np.array([[[0, 0, 0], [0, 0, 0]]])
        diag = SourceDiagnostics(("real",), raw, mask, truth)
        scores = np.concatenate((raw, [[[3, 1, 0], [4, 2, 1]]]))
        panel = Panel("specialist", ("real", "s"), scores, repeat=2,
                      specialist_targets=np.array([True, False]), focus_source="s")
        rows = {r["scope"]: r for r in diag.rows(panel) if r["source"] == "s"}
        self.assertAlmostEqual(rows["all"]["aupr"], 0.5)
        self.assertAlmostEqual(rows["specialist"]["aupr"], 1)
        self.assertEqual(rows["specialist"]["top1_by_target"], 1)
        self.assertEqual(rows["other"]["n_candidates"], 3)
        self.assertEqual(rows["other"]["n_informative_targets"], 0)
        self.assertIsNone(rows["other"]["aupr"])
        self.assertIsNone(rows["other"]["top1_by_target"])
