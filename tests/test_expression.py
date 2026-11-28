import unittest
from dataclasses import replace

import numpy as np
from scipy.stats import rankdata

from naniwa.consensus.expression import TemporalEvidence, _leaders
from naniwa.inputs import Dataset


class ExpressionTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(21)
        genes = tuple(f"g{i}" for i in range(12))
        expression = rng.normal(size=(2, 21, len(genes)))
        times = np.cumsum(rng.uniform(.2, 1.6, size=expression.shape[:2]), axis=1)
        self.dataset = Dataset(genes, genes[:11], expression, times, np.not_equal.outer(genes, genes[:11]))
        self.votes = np.where(self.dataset.mask, rng.uniform(.01, .99, size=self.dataset.mask.shape), np.nan)

    def test_leader_limit_removes_entire_cutoff_tie(self):
        votes = np.array([[9., 8., 7., 6., 5., 4., 1., 1., 1., np.nan], [0.] * 10])
        selected, widths = _leaders(votes)
        np.testing.assert_array_equal(widths, [6, 0])
        np.testing.assert_array_equal(selected[0, :widths[0]], np.arange(6))

    def test_marginal_lag_respects_intervals_series_and_ties(self):
        data = self.dataset
        x = data.X.copy()
        x[:, :, 3] = x[:, :, 4]
        data = replace(data, X=x)
        stats = TemporalEvidence(data)
        z = (x - x.mean(axis=(0, 1))) / x.std(axis=(0, 1))
        previous = z[:, :-1] - z[:, :-1].mean(axis=1, keepdims=True)
        change = np.diff(z, axis=1) / np.diff(data.times, axis=1)[:, :, None]
        change -= change.mean(axis=1, keepdims=True)
        t = 11
        own, y = previous[:, :, t].ravel(), change[:, :, t].ravel()
        residual = y - own * (own @ y) / (own @ own + .05 * len(own))
        evidence = [abs(np.corrcoef(previous[:, :, r].ravel(), residual)[0, 1]) for r in range(11)]
        np.testing.assert_array_equal(stats.marginal[t], (rankdata(evidence) - .5) / 11)

    def test_ridge_forecast_and_contribution_against_direct_regressions(self):
        data, base = self.dataset, self.votes
        stats = TemporalEvidence(data)
        forecast, joint = stats.forecast(base), stats.joint_percentiles(base)
        n = data.times.size - len(data.times)
        z = (data.X - data.X.mean(axis=(0, 1))) / data.X.std(axis=(0, 1))
        def design(values):
            values = values - values.mean(axis=1, keepdims=True)
            return (values / np.sqrt((values**2).mean(axis=(0, 1)))).reshape(n, -1)
        own = design(z[:, :-1])
        response = design(np.diff(z, axis=1) / np.diff(data.times, axis=1)[:, :, None])
        for t in range(len(base)):
            eligible = np.flatnonzero(data.mask[t])
            ids = eligible[np.argsort(-base[t, eligible])[:8]]
            penalties = np.r_[.05 * n, .5 * n / (1 + base[t, ids] / base[t, ids].max())]
            predictors = own[:, np.r_[t, ids]]
            aug = np.vstack([predictors, np.diag(np.sqrt(penalties))])
            y = np.r_[response[:, t], np.zeros(len(penalties))]
            coeff, rss, *_ = np.linalg.lstsq(aug, y, rcond=None)
            np.testing.assert_allclose(forecast.prediction[t], predictors[:, 1:] @ coeff[1:], atol=2e-12)
            gains = [np.linalg.lstsq(np.delete(aug, j+1, axis=1), y, rcond=None)[1][0] - rss[0] for j in range(8)]
            np.testing.assert_array_equal(joint[t, ids[np.argsort(gains)]], np.sort(stats.marginal[t, ids]))
            other = np.setdiff1d(eligible, ids)
            np.testing.assert_array_equal(joint[t, other], stats.marginal[t, other])

    def test_invalid_times_and_masks(self):
        d = self.dataset
        with self.assertRaisesRegex(ValueError, "increase strictly"):
            TemporalEvidence(replace(d, times=d.times[:, ::-1]))
        with self.assertRaisesRegex(ValueError, "mask"):
            TemporalEvidence(replace(d, mask=np.zeros_like(d.mask)))
