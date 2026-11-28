import unittest

import numpy as np
from scipy.special import expit, logit

from naniwa.consensus.correction import correct_vote


class CorrectionTests(unittest.TestCase):
    def test_one_intensity_matches_the_total_targetwise_movement(self):
        base = np.array([[.49, .50, .51, 1.], [.05, .15, .85, .95], [0., .4, np.nan, 1.]])
        q = np.array([[.25, .5, .75, 0.], [.125, .375, .625, .875], [0., .9, .5, 1.]])
        full = correct_vote(base, q, attenuate=False)
        inside = (base > 0) & (base < 1)
        movement = np.log(2.) * (2 * q - 1.)
        applied = np.zeros(len(base))
        totals = np.zeros(len(base))
        for t, mask in enumerate(inside):
            u = movement[t, mask]
            spread = np.diff(np.quantile(logit(base[t, mask]), [.25, .75]))[0]
            strength = min(1., spread / np.ptp(u)) if np.ptp(u) > 0 else 1.
            totals[t] = np.abs(u).sum()
            applied[t] = strength * totals[t]
        panel_strength = applied.sum() / totals.sum()
        self.assertGreater(panel_strength, .3)
        self.assertLess(panel_strength, 1.)
        expected = base.copy()
        expected[inside] = expit(logit(base[inside]) + panel_strength * movement[inside])
        np.testing.assert_allclose(correct_vote(base, q, attenuate=True), expected, rtol=0, atol=3e-16)
        np.testing.assert_allclose(logit(full[inside]) - logit(base[inside]), movement[inside], atol=6e-15)

    def test_zero_full_and_neutral_intensity(self):
        same_votes = np.array([[.2, .2, .2, .2]])
        q = np.array([[.125, .375, .625, .875]])
        np.testing.assert_array_equal(correct_vote(same_votes, q, attenuate=True), same_votes)
        spread_votes = np.array([[.01, .02, .98, .99]])
        np.testing.assert_array_equal(correct_vote(spread_votes, q, attenuate=True),
                                      correct_vote(spread_votes, q, attenuate=False))
        np.testing.assert_array_equal(correct_vote(spread_votes, np.full_like(q, .5), attenuate=True), spread_votes)

    def test_boundaries_masks_and_floating_point_extremes(self):
        base = np.array([[0., 1., np.nan], [np.nextafter(0., 1.), np.nextafter(1., 0.), .8]])
        result = correct_vote(base, np.array([[.1, .8, .5], [.1, .99, .5]]), attenuate=True)
        np.testing.assert_array_equal(result[0], base[0])
        self.assertTrue(np.all((result[1] >= 0.) & (result[1] <= 1.)))
