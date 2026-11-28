from collections import defaultdict
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np

from experimentation import study
from experimentation.construction_report import construction_contrasts, registered_recipe
from naniwa.consensus import ConsensusOptions
from naniwa.inputs import Dataset


class ConstructionStudyTests(unittest.TestCase):
    def test_recipe_covers_the_operating_path_and_matched_controls(self):
        recipe = study.read_json(study.CONSTRUCTION)
        path = recipe["path"]
        self.assertEqual(path, ["mean_minmax", "family_mean", "marginal_global", "naniwa"])
        self.assertEqual(recipe["primary"], path[-1])
        self.assertEqual(set(path) | set(recipe["controls"]), set(study.variants()))
        pairs = {(method, reference) for label, method, reference in recipe["contrasts"]}
        self.assertEqual(pairs, set(zip(path[1:], path)) | {
            ("naniwa", "mean_minmax"), ("naniwa", "joint_full"), ("naniwa", "flat_joint_global"),
            ("joint_full", "marginal_full"), ("marginal_global", "marginal_full")})

    def test_each_phase_scores_its_registered_methods(self):
        protocol = study.read_json(study.FINAL)
        tools = protocol["individual_tools"]
        genes = tuple("ABCDE")
        mask = np.not_equal.outer(genes, genes)
        rng = np.random.default_rng(42)
        data = Dataset(genes, genes, rng.normal(size=(1, 6, len(genes))), np.arange(6)[None, :], mask)
        raw = rng.random((len(tools), *mask.shape))
        for phase in ("construction", "final"):
            args = SimpleNamespace(phase=phase)
            methods = [m["method"] for m, values in study.predictions(args, data, tools, raw, {}, protocol)]
            if phase == "construction":
                self.assertEqual(set(methods), set(tools) | set(study.variants()))
                self.assertNotIn("exact_mean", methods)
            else:
                self.assertEqual(set(methods), {"naniwa", *tools, *protocol["controls"], *protocol["baselines"]})
                self.assertIn("exact_mean", methods)
        truth = (raw[0] > 0.5) & mask
        with TemporaryDirectory() as directory:
            folder = Path(directory)
            by_panel = defaultdict(list)
            for row, values in study.robustness_predictions({"id": "toy"}, data, tools, raw, folder, truth):
                by_panel[row["panel"]].append(row["policy"])
                if row["policy"] == "naniwa" and row["scenario"].startswith(("copy-", "null-exact-")):
                    self.assertLessEqual(row["max_abs_change_from_anchor"], 1e-12)
                    self.assertEqual(row["score_anchor"], row["founder_panel"] or "clean")
            self.assertEqual(len(by_panel), 107)
            for policies in by_panel.values():
                self.assertEqual(policies, ["naniwa", "mean_minmax"])
            weights = study.table(folder / "specialist_weights.tsv")
            self.assertEqual({row["policy"] for row in weights}, {"naniwa", "mean_minmax"})

    def test_phase_recipes_resolve_the_public_module_parameters(self):
        expected = {
            "mean_minmax": ("all", "none", "none"),
            "exact_mean": ("distinct", "none", "none"),
            "family_mean": ("families", "none", "none"),
            "marginal_global": ("families", "marginal", "global"),
            "naniwa": ("families", "joint", "global"),
            "marginal_full": ("families", "marginal", "full"),
            "joint_full": ("families", "joint", "full"),
            "flat_joint_global": ("distinct", "joint", "global"),
        }
        recipes = {phase: study.variants(phase) for phase in study.PHASE_RECIPES}
        self.assertEqual(set(recipes["construction"]), expected.keys() - {"exact_mean"})
        self.assertEqual(set(recipes["robustness"]), {"naniwa", "mean_minmax"})
        self.assertEqual(set(recipes["final"]), expected.keys() - {"marginal_full"})
        for phase, stages in recipes.items():
            for stage, options in stages.items():
                with self.subTest(phase=phase, stage=stage):
                    vote, evidence, intensity = expected[stage]
                    public = ConsensusOptions.resolve(vote=vote, evidence=evidence, intensity=intensity)
                    self.assertEqual(public.parameters(), options.parameters())

    def test_stage_contrasts_are_paired_and_graph_balanced(self):
        # Two positive paired differences in one graph and a negative difference in the other.
        rows = []
        for graph, view, base, full in [("g", "a", 0.1, 0.2), ("g", "b", 0.3, 0.8), ("h", "c", 0.5, 0.3)]:
            for name, value in [("base", base), ("full", full)]:
                rows.append(dict(graph_id=graph, id=view, method=name, aupr=value))
        contrast = [("layer", "full", "base")]
        result, = construction_contrasts(rows, contrast, 20261126)
        self.assertAlmostEqual(result["delta_aupr"], 0.05)
        self.assertAlmostEqual(result["ci_low"], -0.2)
        self.assertAlmostEqual(result["ci_high"], 0.3)
        self.assertEqual((result["n_views"], result["n_graphs"], result["graphs_higher"], result["graphs_lower"]),
                         (3, 2, 1, 1))
        with self.assertRaisesRegex(ValueError, "missing registered"):
            construction_contrasts(rows[:-1], contrast, 20261126)
        one, = construction_contrasts(rows[:2], contrast, 20261126)
        self.assertIsNone(one["ci_low"])

    def test_freeze_records_the_production_settings(self):
        frozen = study.frozen()
        self.assertEqual(frozen["options"], dict(stage="naniwa", vote="families", evidence="joint", intensity="global"))
        self.assertEqual(frozen["options"], study.variants()[frozen["variant"]].parameters())
        self.assertEqual(frozen["construction_sha256"], study.digest(study.CONSTRUCTION))

    def test_evaluation_checks_stage_coverage_and_originating_recipe(self):
        recipe = study.read_json(study.CONSTRUCTION)
        tools = set(study.read_json(study.FINAL)["individual_tools"])
        methods = {*tools, *recipe["path"], *recipe["controls"]}
        with TemporaryDirectory() as directory:
            protocol = Path(directory) / "protocol.json"
            loaded, provenance = registered_recipe(study.CONSTRUCTION, protocol, methods, tools)
            self.assertEqual(loaded, recipe)
            self.assertEqual(provenance, {"construction_recipe_sha256": study.digest(study.CONSTRUCTION)})
            for mismatched in [methods | {"exact_mean"}, methods - {"mean_minmax"}, methods - {min(tools)}]:
                with self.subTest(mismatched=mismatched), self.assertRaisesRegex(ValueError, "method IDs"):
                    registered_recipe(study.CONSTRUCTION, protocol, mismatched, tools)
            for digest in [study.digest(study.CONSTRUCTION), "different"]:
                protocol.write_text(json.dumps({"phase": "construction", "recipes": {study.CONSTRUCTION.name: digest}}))
                if digest == "different":
                    with self.assertRaisesRegex(ValueError, "scoring protocol"):
                        registered_recipe(study.CONSTRUCTION, protocol, methods, tools)
                else:
                    _, verified = registered_recipe(study.CONSTRUCTION, protocol, methods, tools)
                    self.assertEqual(verified["scoring_protocol_sha256"], study.digest(protocol))

    def test_registry_verifies_identity_before_scoring(self):
        with TemporaryDirectory() as temp:
            index = Path(temp) / "empty.tsv"
            index.write_text("id\tgraph_id\n")
            with self.assertRaisesRegex(ValueError, "registered benchmark"):
                study.verify_benchmark([], index, "construction")
            with self.assertRaisesRegex(ValueError, "registered benchmark"):
                study.verify_benchmark([], index, "final")
