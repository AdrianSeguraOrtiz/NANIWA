"""Agglomerate distinct tools by the predictive cost of sharing a vote."""

import numpy as np

from .simplex import fit_simplex


def pair_costs(gram, xy, objective, weights, sizes, informative, n_eff):
    """Compare every forced merger at the current fit, holding all other family weights fixed."""
    left, right = np.triu_indices(len(sizes), 1)
    diagonal = np.diagonal(gram, axis1=1, axis2=2)
    fitted = np.einsum("tij,tj->ti", gram, weights)
    gradient = fitted - xy
    rss = objective + (weights * (fitted - 2. * xy)).sum(axis=1)
    mass = weights[:, left] + weights[:, right]
    shift = mass * (sizes[left] / (sizes[left] + sizes[right])) - weights[:, left]
    variance = np.maximum(diagonal[:, left] + diagonal[:, right] - 2. * gram[:, left, right], 0.)
    increment = shift * (2. * (gradient[:, left] - gradient[:, right]) + shift * variance)
    if np.any(increment[informative] < -2e-7 * np.maximum(objective[informative, None], 1e-12)):
        raise ArithmeticError("Forced family merger improves on the simplex optimum")
    floor = np.maximum(objective * 1e-12, 1e-12)[:, None]
    ratio = (np.maximum(rss[:, None] + np.maximum(increment, 0.), floor)
             / np.maximum(rss[:, None], floor))
    costs = n_eff * np.log(ratio[informative]).mean(axis=0) - np.log(n_eff)
    return left, right, costs


def predictive_families(forecasts, evidence):
    """Build one partition across targets. Refit the simplex after each accepted merger."""
    members = [(i,) for i in range(len(forecasts))]
    if len(members) == 1 or evidence.n_eff < 2 or not evidence.informative.any():
        return members
    predictions = np.stack([f.prediction for f in forecasts], axis=1)
    own = np.stack([f.own_projection for f in forecasts], axis=1)
    xy = np.stack([f.cross_response for f in forecasts], axis=1)
    gram = predictions @ predictions.swapaxes(1, 2) - own[:, :, None] * own[:, None, :]
    while len(members) > 1:
        weights = fit_simplex(gram, xy, evidence.informative, evidence.objective)
        sizes = np.array([len(group) for group in members])
        left, right, costs = pair_costs(gram, xy, evidence.objective, weights, sizes,
                                        evidence.informative, evidence.n_eff)
        # Members are in canonical order; resolve roundoff ties lexicographically.
        costs = np.round(costs, 12)
        best = np.argmin(costs)
        if costs[best] >= 0.:
            break
        i, j = left[best], right[best]
        fraction = sizes[i] / (sizes[i] + sizes[j])
        other = 1. - fraction
        diagonal = (fraction**2 * gram[:, i, i] + 2. * fraction * other * gram[:, i, j]
                    + other**2 * gram[:, j, j])
        cross = fraction * gram[:, i, :] + other * gram[:, j, :]
        gram[:, i, :] = gram[:, :, i] = cross
        gram[:, i, i] = diagonal
        xy[:, i] = fraction * xy[:, i] + other * xy[:, j]
        keep = np.arange(len(members)) != j
        gram = gram[:, keep][:, :, keep]
        xy = xy[:, keep]
        members[i] = tuple(sorted(members[i] + members[j]))
        del members[j]
    return members
