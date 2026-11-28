import csv
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from experimentation import robustness_report as report


def write_table(path, rows, fields):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def metadata(name, tools, repeat="", reference="clean", founder="", matched="", focus="[]"):
    return [dict(method=f"{policy}__{name}", panel=name, policy=policy, scenario=name.split("-r")[0],
                 tools=tools, rank_correlation_to_clean="", repeat=repeat, reference_panel=reference,
                 founder_panel=founder, matched_panel=matched, specialist_targets=focus)
            for policy in report.POLICIES]


class RobustnessReportTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "methods.tsv"
        self.targets = {"x", "y", "z"}
        self.metadata = (metadata("clean", "a,b") + metadata("leave-b", "a")
                         + metadata("null-1-r1", "a,b,null1", "1")
                         + metadata("null-2-r1", "a,b,null1,null2", "1", founder="null-1-r1")
                         + metadata("copies-r1", "a,b,null1,clone", "1", founder="null-1-r1", matched="null-2-r1")
                         + metadata("specialist-r1", "a,b,specialist", "1", focus='["x"]'))

    def plan(self, rows=None):
        rows = self.metadata if rows is None else rows
        write_table(self.path, rows, rows[0])
        return report.PanelPlan(self.path, self.targets)

    def test_missing_or_inconsistent_panel_metadata(self):
        with self.assertRaisesRegex(ValueError, "missing required robustness policy"):
            self.plan(self.metadata[:-1])
        with self.assertRaisesRegex(ValueError, "unexpected robustness policy"):
            self.plan(self.metadata + [dict(self.metadata[0], method="extra__clean", policy="extra")])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.plan(self.metadata + self.metadata[-1:])
        with self.assertRaisesRegex(ValueError, "differ across policies"):
            self.plan([*self.metadata[:-1], dict(self.metadata[-1], specialist_targets='["y"]')])
        with self.assertRaisesRegex(ValueError, "invalid specialist targets"):
            self.plan([*self.metadata[:-1], dict(self.metadata[-1], specialist_targets='["foreign"]')])
        with self.assertRaisesRegex(ValueError, "unknown or unmatched"):
            self.plan(self.metadata + metadata("bad-r2", "a,b,null1,clone", "2", founder="null-1-r1"))

    def test_fractional_and_zero_ties_scoped_truth_and_policy_contrasts(self):
        plan = self.plan()
        cache = SimpleNamespace(truth_keys={("a", "x"), ("b", "y")}, candidate_space=SimpleNamespace(sources={"a", "b", "c"}))
        observed_scopes = []

        def average_precision(*, truth_keys, prediction_scores, n_candidates):
            observed_scopes.append((truth_keys, set(prediction_scores), n_candidates))
            return len(truth_keys) / n_candidates

        def edge(src, dst, score):
            return SimpleNamespace(source=src, target=dst, score=score)

        for (panel, policy), method in plan.methods.items():
            # For y, all scores are zero and one of the three eligible regulators is true.
            # For x, NANIWA on the specialist panel shares first place with one false source.
            positives = [edge("a", "x", 1), edge("c", "z", 1)]
            if panel == "specialist-r1" and policy == "naniwa":
                positives.append(edge("b", "x", 1))
            plan.score(method, {"aupr": 0.5}, positives, cache, average_precision)
        rows = plan.observations({"id": "d", "graph_id": "g"}, plan.methods.values())
        pick = lambda scenario, contrast, scope, metric: next(
            r for r in rows if (r["scenario"], r["contrast"], r["scope"], r["metric"]) == (scenario, contrast, scope, metric))
        top = pick("specialist", "reference", "all", "top1_by_target")
        self.assertEqual((top["n_targets"], top["n_informative_targets"], top["n_candidates"], top["n_truth_edges"]), (3, 2, 9, 2))
        self.assertAlmostEqual(top["naniwa"], (0.5 + 1 / 3) / 2)
        self.assertAlmostEqual(top["comparator_score"], (1 + 1 / 3) / 2)
        focused = pick("specialist", "reference", "specialist", "top1_by_target")
        self.assertEqual([focused[k] for k in report.EFFECTS], [-0.5, -0.5, 0, -0.5, 0.5])
        other = pick("specialist", "reference", "other", "top1_by_target")
        self.assertEqual([other[k] for k in report.EFFECTS], [0, 0, 0, 0, 0])
        self.assertEqual(other["naniwa"], 1 / 3)
        self.assertIn(({("a", "x")}, {("a", "x"), ("b", "x")}, 3), observed_scopes)
        self.assertIn(({("b", "y")}, {("c", "z")}, 6), observed_scopes)
        self.assertEqual({r["comparator"] for r in rows}, {"mean_minmax"})
        self.assertEqual(len(rows), len({(r["panel"], r["reference_panel"], r["contrast"], r["scope"], r["metric"])
                                         for r in rows}))
        self.assertEqual(pick("reintroduce-b", "reintroduction", "all", "aupr")["reference_panel"], "leave-b")
        self.assertEqual(pick("copies", "matched", "all", "aupr")["reference_panel"], "null-2-r1")
        self.assertEqual(pick("copies", "founder", "all", "aupr")["reference_panel"], "null-1-r1")
        with self.assertRaisesRegex(ValueError, "identical methods"):
            plan.observations({"id": "d"}, [])

    def test_paired_averaging_and_bootstrap_use_graphs_as_units(self):
        factors = ("id", "graph_id")
        rows = []
        # Repetitions, datasets and graphs have deliberately unequal sample sizes.
        for dataset, graph, values in (("a", "g", [0, 0.6]), ("b", "g", [1]), ("c", "h", [0.2])):
            for index, value in enumerate(values):
                for metric in ("aupr", "top1_by_target"):
                    row = dict(id=dataset, graph_id=graph, scenario="test", contrast="reference", scope="all",
                               metric=metric, repeat=index, panel="test", reference_panel="clean", comparator="mean_minmax")
                    for field in report.VALUES:
                        row[field] = None if metric == "top1_by_target" and graph == "h" else value
                    rows.append(row)
        path = Path(self.directory.name)
        report.write_report(path, rows, factors, {}, 2, write_table)
        with (path / "robustness_summary.tsv").open() as stream:
            aupr, top = list(csv.DictReader(stream, delimiter="\t"))
        self.assertAlmostEqual(float(aupr["naniwa"]), ((0.3 + 1) / 2 + 0.2) / 2)
        self.assertEqual((aupr["n_repeats"], aupr["n_datasets"], aupr["n_graphs"]), ("4", "3", "2"))
        self.assertEqual((float(aupr["change_gap_ci_low"]), float(aupr["change_gap_ci_high"])), (0.2, 0.65))
        self.assertEqual((top["n_valid_graphs"], top["change_gap_ci_high"]), ("1", ""))
        self.assertAlmostEqual(float(top["naniwa"]), 0.65)
        with (path / "robustness_counts.tsv").open() as stream:
            counts = list(csv.DictReader(stream, delimiter="\t"))
        self.assertEqual([(r["comparator"], r["effect"], r["n_scenarios"]) for r in counts],
                         [("mean_minmax", "gap", "1"), ("mean_minmax", "change_gap", "1")])

    def test_strongest_source_uses_only_other_graphs_and_breaks_ties_by_name(self):
        rows = [dict(graph_id=g, contrast="reintroduction", scenario=f"reintroduce-{t}", reference_panel=f"leave-{t}")
                for g in ("g1", "g2", "g3") for t in ("a", "b")]
        diag = {"d1": ("g1", {"a": 1, "b": 0}), "d2": ("g2", {"a": 0, "b": 1}),
                "d3": ("g2", {"a": 0, "b": 1}), "d4": ("g3", {"a": 0.5, "b": 0.5})}
        result = report.best_reintroductions(rows, diag)
        self.assertEqual([r["reference_panel"] for r in result], ["leave-b", "leave-a", "leave-a"])
        self.assertEqual(report.best_reintroductions(rows[:2], {"d1": diag["d1"]}), [])
