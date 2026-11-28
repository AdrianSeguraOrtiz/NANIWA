import unittest

import numpy as np
from scipy.optimize import minimize

from naniwa.consensus import ConsensusOptions, TemporalConsensus
from naniwa.consensus.hierarchy import pair_costs
from naniwa.consensus.simplex import fit_simplex
from naniwa.inputs import Dataset


class HierarchyTests(unittest.TestCase):
    def test_predictive_mergers_keep_a_complementary_family(self):
        rng = np.random.default_rng(12932)
        genes = tuple(f"g{i}" for i in range(18))
        n, length = 12, 70
        expression = rng.normal(size=(n, length, len(genes)))
        for i in range(length - 1):
            expression[:, i+1, 2:] = (.22 * expression[:, i, 2:] + expression[:, i, np.repeat([0, 1], 8)]
                                      + .08 * rng.normal(size=(n, 16)))
        data = Dataset(genes, genes[:2], expression, np.tile(np.arange(length), (n, 1)),
                       np.not_equal.outer(genes, genes[:2]))
        raw = np.zeros((3, len(genes), 2))
        raw[0, :, 0] = raw[1, :, 1] = 1.
        raw[2, :, 0] = np.linspace(.1, 1., len(genes))
        shares = np.empty(raw.shape[:2])
        TemporalConsensus(data).score(raw, ConsensusOptions("family_mean"), shares)
        np.testing.assert_array_equal(shares, np.broadcast_to([[.25], [.5], [.25]], shares.shape))

    def test_simplex_and_pair_costs_match_independent_optimizations_and_residuals(self):
        rng = np.random.default_rng(674)
        for n, width in [(31, 5), (8, 13)]:
            predictors = rng.normal(size=(5, n, width))
            predictors[0, :, 0] = predictors[0, :, 1]  # Rank-deficient Gram matrix.
            y = rng.normal(size=predictors.shape[:2])
            gram = predictors.swapaxes(1, 2) @ predictors
            xy = np.einsum("tni,tn->ti", predictors, y)
            objective = (y * y).sum(axis=1)
            active = np.ones(len(y), dtype=bool)
            weights = fit_simplex(gram, xy, active, objective)
            for target, (design, observed) in enumerate(zip(predictors, y, strict=True)):
                def residual(w):
                    return np.sum((design @ w - observed)**2)
                ref = minimize(residual, np.full(width, 1 / width), method="SLSQP", bounds=[(0., 1.)] * width,
                               constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.}],
                               options={"ftol": 1e-12, "maxiter": 400})
                self.assertAlmostEqual(residual(weights[target]), ref.fun, places=8)
            sizes = rng.integers(1, 4, size=width)
            left, right, cost = pair_costs(gram, xy, objective, weights, sizes, active, n)
            rss = ((np.einsum("tni,ti->tn", predictors, weights) - y)**2).sum(axis=1)
            for i, j, estimated in zip(left, right, cost, strict=True):
                merged = weights.copy()
                total = weights[:, i] + weights[:, j]
                merged[:, i] = total * sizes[i] / (sizes[i] + sizes[j])
                merged[:, j] = total * sizes[j] / (sizes[i] + sizes[j])
                forced = ((np.einsum("tni,ti->tn", predictors, merged) - y)**2).sum(axis=1)
                self.assertAlmostEqual(estimated, n * np.log(forced/rss).mean() - np.log(n), places=8)
