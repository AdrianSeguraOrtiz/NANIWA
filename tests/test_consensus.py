import unittest
from dataclasses import replace

import numpy as np

from naniwa.consensus import ConsensusOptions, STAGES, TemporalConsensus
from naniwa.consensus.correction import correct_vote
from naniwa.consensus.scale import minmax
from naniwa.inputs import Dataset


class ConsensusTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(129873)
        genes = tuple(f"g{i}" for i in range(13))
        regulators = genes[:11]
        expression = rng.normal(size=(3, 13, len(genes)))
        times = np.cumsum(rng.uniform(.6, 1.7, size=expression.shape[:2]), axis=1)
        self.data = Dataset(genes, regulators, expression, times, np.not_equal.outer(genes, regulators))
        self.raw = rng.random((6, len(genes), len(regulators)))
        self.model = TemporalConsensus(self.data)

    def test_default_panel_and_source_shares(self):
        np.testing.assert_array_equal(self.model.score(self.raw), self.model.score(self.raw, ConsensusOptions("naniwa")))
        order = [2, 4, 3, 1, 5, 0]
        scaled = minmax(self.raw, self.data.mask)
        clone = np.concatenate((self.raw, self.raw[[2, 2]]))
        for stage in STAGES:
            options = ConsensusOptions(stage)
            shares = np.empty(self.raw.shape[:2])
            scores = self.model.score(self.raw, options, shares)
            self.assertTrue(np.isfinite(scores[self.data.mask]).all())
            self.assertTrue(np.isnan(scores[~self.data.mask]).all())
            self.assertTrue(np.all((scores[self.data.mask] >= 0.) & (scores[self.data.mask] <= 1.)))
            np.testing.assert_allclose(shares.sum(axis=0), 1., rtol=0., atol=2e-15)
            reordered = np.empty_like(shares)
            np.testing.assert_array_equal(self.model.score(self.raw[order], options, reordered), scores)
            np.testing.assert_array_equal(reordered, shares[order])
            if stage in ("mean_minmax", "exact_mean", "family_mean"):
                np.testing.assert_allclose((scaled * shares[:, :, None]).sum(axis=0), scores, atol=2e-15)
            extended_shares = np.empty(clone.shape[:2])
            extended = self.model.score(clone, options, extended_shares)
            if stage == "mean_minmax":
                self.assertFalse(np.array_equal(extended[self.data.mask], scores[self.data.mask]))
            else:
                np.testing.assert_array_equal(scores, extended)
                np.testing.assert_allclose(extended_shares[[2, 6, 7]].sum(axis=0), shares[2], atol=2e-15)

    def test_flat_single_series_ties_and_withholds_mergers(self):
        flat = replace(self.data, X=np.ones_like(self.data.X[:1, :2]), times=self.data.times[:1, :2])
        model = TemporalConsensus(flat)
        for stage in STAGES:
            np.testing.assert_allclose(model.score(self.raw, ConsensusOptions(stage)), minmax(self.raw, flat.mask).mean(axis=0))

    def test_each_vote_combines_with_each_temporal_update(self):
        raw = np.concatenate((self.raw, self.raw[[2, 2]]))
        base_stages = {"all": "mean_minmax", "distinct": "exact_mean", "families": "family_mean"}
        registered = {ConsensusOptions(stage).components: stage for stage in STAGES}
        tested = 0
        for vote, base_stage in base_stages.items():
            base_shares = np.empty(raw.shape[:2])
            base = self.model.score(raw, ConsensusOptions(base_stage), base_shares)
            evidence_scores = {"marginal": self.model.evidence.marginal,
                               "joint": self.model.evidence.joint_percentiles(base)}
            for evidence, intensities in (("none", ("none",)), ("marginal", ("full", "global")), ("joint", ("full", "global"))):
                for intensity in intensities:
                    with self.subTest(vote=vote, evidence=evidence, intensity=intensity):
                        options = ConsensusOptions(vote=vote, evidence=evidence, intensity=intensity)
                        self.assertEqual(options.stage, registered.get(options.components))
                        self.assertEqual(ConsensusOptions.resolve(**options.parameters()), options)
                        self.assertEqual(ConsensusOptions(**options.parameters()), options)
                        expected = base if evidence == "none" else correct_vote(
                            base, evidence_scores[evidence], attenuate=intensity == "global")
                        shares = np.empty(raw.shape[:2])
                        np.testing.assert_array_equal(self.model.score(raw, options, shares), expected)
                        np.testing.assert_array_equal(shares, base_shares)
                        tested += 1
        self.assertEqual(tested, 15)

    def test_single_source_extremes_and_zero_eligible_target(self):
        mask = self.data.mask.copy()
        mask[1, :] = False
        data = replace(self.data, mask=mask)
        result = TemporalConsensus(data).score(self.raw[:1])
        original = minmax(self.raw[:1], mask)[0]
        self.assertTrue(np.isnan(result[1]).all())
        np.testing.assert_array_equal(result[original == 0.], 0.)
        np.testing.assert_array_equal(result[original == 1.], 1.)

    def test_reordering_genes_and_regulators(self):
        rng = np.random.default_rng(2478)
        rows, cols = rng.permutation(len(self.data.genes)), rng.permutation(len(self.data.regulators))
        order = np.ix_(rows, cols)
        model = TemporalConsensus(Dataset(tuple(np.array(self.data.genes)[rows]), tuple(np.array(self.data.regulators)[cols]),
                                           self.data.X[:, :, rows], self.data.times, self.data.mask[order]))
        np.testing.assert_allclose(model.score(self.raw[:, rows][:, :, cols]), self.model.score(self.raw)[order],
                                   rtol=0., atol=2e-14)

    def test_rejects_invalid_inputs_and_nonvarying_networks(self):
        for raw in (self.raw[0], self.raw[:0], self.raw[:, :-1], self.raw * np.nan, -self.raw, np.ones_like(self.raw)):
            with self.assertRaises(ValueError):
                self.model.score(raw)
        readonly = np.zeros(self.raw.shape[:2])
        readonly.flags.writeable = False
        for buffer in (np.zeros(1), np.zeros(self.raw.shape[:2], dtype=int), readonly, []):
            with self.assertRaisesRegex(ValueError, "source_vote_weights"):
                self.model.score(self.raw, source_vote_weights=buffer)
        with self.assertRaises(ValueError):
            ConsensusOptions(stage="unknown")
        for key in ("vote", "evidence", "intensity"):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, f"Unknown {key}"):
                ConsensusOptions(**{key: "unknown"})
