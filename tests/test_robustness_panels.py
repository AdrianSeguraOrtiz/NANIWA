import hashlib
import json
import unittest
from pathlib import Path

import numpy as np
from scipy.special import ndtri

from experimentation.robustness_panels import panels


SPEC = Path(__file__).resolve().parents[1] / "experimentation/02-robustness/panels.json"


class RobustnessPanelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = json.loads(SPEC.read_text())
        cls.tools = tuple(f"tool{i}" for i in range(7))
        rng = np.random.default_rng(7411)
        cls.mask = rng.random((24, 20)) > .15
        cls.mask[0] = False
        cls.mask[1, 1:] = False
        cls.raw = rng.integers(0, 20, size=(len(cls.tools), *cls.mask.shape)).astype(float)
        cls.raw[:, ~cls.mask] = 0
        cls.truth = cls.mask & (rng.random(cls.mask.shape) < .28)
        cls.by_id = {p.id: p for p in panels(cls.tools, cls.raw, cls.mask, "fixture", cls.spec, cls.truth)}

    def test_registered_panels_and_seeded_draws(self):
        self.assertEqual(len(self.by_id), 107)
        self.assertEqual(sum(p.exact_copy for p in self.by_id.values()), 13)
        for p in self.by_id.values():
            self.assertEqual(len(set(p.names)), len(p.names))
            self.assertEqual(p.sources.shape, (len(p.names), *self.mask.shape))
            self.assertEqual(p.reference, "clean")
            self.assertEqual(p.exact_copy, p.scenario.startswith(("copy-", "null-exact-")))
            if p.matched is not None:
                self.assertIn(p.matched, self.by_id)
            if p.founder is not None:
                self.assertIn(p.founder, self.by_id)
        # Check the registered baseline perturbations, including source order and scores.
        panel_digest = hashlib.sha256()
        for p in list(self.by_id.values())[:68]:
            panel_digest.update(json.dumps([p.id, list(p.names)]).encode())
            panel_digest.update(p.sources.astype("<f8").tobytes())
        self.assertEqual(panel_digest.hexdigest(), "df0f028285d1db07379d553e936b20d3530a5a51d58ae47a9d3460ea378ed8ba")

    def test_null_relatives_share_the_independent_founder_and_form_prefixes(self):
        for repeat in self.spec["repeat_ids"]:
            founder = self.by_id[f"null-1-r{repeat}"]
            for count in (3, 7):
                independent = self.by_id[f"null-{count}-r{repeat}"]
                np.testing.assert_array_equal(independent.sources[:8], founder.sources)
                self.assertEqual(independent.founder, founder.id)
                exact = self.by_id[f"null-exact-{count}-r{repeat}"]
                self.assertEqual(exact.matched, independent.id)
                self.assertEqual(exact.founder, founder.id)
                self.assertEqual(exact.names[:8], founder.names)
                for copy in exact.sources[7:]:
                    np.testing.assert_array_equal(copy, founder.sources[-1])

                for frac in self.spec["near_copy_fractions"]:
                    near = self.by_id[f"null-near-{count}-{frac:g}-r{repeat}"]
                    self.assertEqual(near.matched, independent.id)
                    self.assertEqual(near.founder, founder.id)
                    self.assertEqual(near.names, exact.names)
                    np.testing.assert_array_equal(near.sources[:8], founder.sources)
                    prefix = self.by_id[f"null-near-3-{frac:g}-r{repeat}"]
                    np.testing.assert_array_equal(near.sources[:10], prefix.sources)
                    for relative in near.sources[8:]:
                        self.assertTrue(np.any(relative != founder.sources[-1]))
                        for t, eligible in enumerate(self.mask):
                            x, y = relative[t, eligible], founder.sources[-1, t, eligible]
                            np.testing.assert_array_equal(np.sort(x), np.sort(y))
                            self.assertLessEqual(np.count_nonzero(x != y), max(2, round(len(x) * frac)))

    def test_specialist_assigns_strengths_before_observing_the_reference(self):
        no_edges = {p.id: p for p in panels(self.tools, self.raw, self.mask, "fixture",
                                            self.spec, np.zeros_like(self.truth)) if p.focus_source}
        for repeat in self.spec["repeat_ids"]:
            panel = self.by_id[f"specialist-r{repeat}"]
            control = no_edges[panel.id]
            group = panel.specialist_targets
            np.testing.assert_array_equal(group, control.specialist_targets)
            self.assertFalse(np.any(group & ~self.mask.any(axis=1)))
            self.assertEqual(group.sum(), round(self.mask.any(axis=1).sum() * .5))
            self.assertEqual(panel.names[-1], panel.focus_source)
            np.testing.assert_array_equal(panel.sources[:-1], self.raw)
            shifts = np.where(group, 2., .5)[:, None] * self.truth
            delta = ndtri(panel.sources[-1, self.mask]) - ndtri(control.sources[-1, self.mask])
            np.testing.assert_allclose(delta, shifts[self.mask], atol=1e-12)

    def test_growth_preserves_quality_and_controls_error_dependence(self):
        for repeat in self.spec["repeat_ids"]:
            previous = {arm: self.by_id["clean"] for arm in ("complementary", "shared")}
            for size in self.spec["panel_growth"]["sizes"][1:]:
                a = self.by_id[f"scale-complementary-{size}-r{repeat}"]
                b = self.by_id[f"scale-shared-{size}-r{repeat}"]
                self.assertEqual(b.matched, a.id)
                self.assertEqual(a.names, b.names)
                for arm, p in (("complementary", a), ("shared", b)):
                    smaller = previous[arm]
                    self.assertEqual(p.names[:len(smaller.names)], smaller.names)
                    np.testing.assert_array_equal(p.sources[:len(smaller.names)], smaller.sources)
                    previous[arm] = p
                for t, eligible in enumerate(self.mask):
                    for label in (False, True):
                        cells = eligible & (self.truth[t] == label)
                        np.testing.assert_array_equal(np.sort(a.sources[7:, t, cells], axis=1),
                                                      np.sort(b.sources[7:, t, cells], axis=1))
            # Absolute source quality is paired above. Here compare only negative-edge errors.
            negative = self.mask & ~self.truth
            pairs = np.triu_indices(len(a.names) - 7, 1)
            independent_corr = np.corrcoef(a.sources[7:][:, negative])[pairs].mean()
            shared_corr = np.corrcoef(b.sources[7:][:, negative])[pairs].mean()
            self.assertGreater(shared_corr, independent_corr + .45)
            self.assertEqual(len(np.unique(b.sources[7:], axis=0)), len(b.names) - 7)

    def test_labels_must_match_candidate_axes(self):
        with self.assertRaisesRegex(ValueError, "boolean matrix"):
            list(panels(self.tools, self.raw, self.mask, "fixture", self.spec, self.truth.T))
