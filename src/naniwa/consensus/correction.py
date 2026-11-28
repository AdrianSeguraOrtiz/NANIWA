"""Temporal adjustment in log odds with one intensity for a complete panel."""

import numpy as np
from scipy.special import expit, logit


def _global_intensity(log_odds, movement, interior):
    """Match the sum of absolute movements obtained under each target's IQR bound."""
    counts = interior.sum(axis=1)
    ordered = np.sort(np.where(interior, log_odds, np.inf), axis=1)
    rows = np.arange(len(counts))

    def quantile(p):
        position = np.maximum(counts - 1, 0) * p
        lower = position.astype(int)
        low = np.where(counts > 0, ordered[rows, lower], 0.)
        high = np.where(counts > 0, ordered[rows, np.ceil(position).astype(int)], 0.)
        return low + (high - low) * (position - lower)

    spread = quantile(.75) - quantile(.25)
    largest = np.max(movement, axis=1, where=interior, initial=-np.inf)
    smallest = np.min(movement, axis=1, where=interior, initial=np.inf)
    ranges = np.where(counts > 0, largest - smallest, 0.)
    strengths = np.ones_like(spread)
    limited = (counts >= 2) & (ranges > spread)
    strengths[limited] = spread[limited] / ranges[limited]
    volume = np.abs(movement).sum(axis=1)
    total = volume.sum()
    return float(np.clip(strengths @ volume / total, 0., 1.)) if total else 1.


def correct_vote(base, percentiles, *, attenuate):
    """Adjust interior votes by 2**(2*q-1) in odds, then optionally attenuate globally."""
    interior = (base > 0) & (base < 1)
    full = base.copy()
    votes = base[interior]
    factor = np.exp2(2. * percentiles[interior] - 1.)
    adjusted = np.clip(votes * factor / (1. + votes * (factor - 1.)), 0., 1.)
    full[interior] = adjusted
    if not attenuate or not interior.any():
        return full

    log_odds = np.zeros_like(base)
    movement = np.zeros_like(base)
    log_odds[interior] = logit(votes)
    # Use the exact odds factor if an interior update rounds to a floating-point endpoint.
    steps = np.log(factor)
    represented = (adjusted > 0.) & (adjusted < 1.)
    steps[represented] = logit(adjusted[represented]) - logit(votes[represented])
    movement[interior] = steps
    intensity = _global_intensity(log_odds, movement, interior)
    if intensity == 1.:
        return full
    if intensity == 0.:
        return base.copy()
    shifted = interior & (movement != 0.)
    full[shifted] = expit(log_odds[shifted] + intensity * movement[shifted])
    return full
