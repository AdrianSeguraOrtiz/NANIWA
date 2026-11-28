"""Least-squares fits on per-target simplices for evaluating family mergers."""

import numpy as np


def _face_optimum(gram, xy, support):
    """Solve the sum-to-one normal equations on each target's current support."""
    width = xy.shape[1]
    allowed = support[:, :, None] & support[:, None, :]
    # A relative 1e-12 ridge resolves numerical null-space ties towards equal mass.
    regularizer = np.where(support, 1e-12, 1.)
    system = gram * allowed + np.eye(width) * regularizer[:, :, None]
    rhs = np.stack((xy * support, support), axis=-1)
    solution = np.linalg.solve(system, rhs)
    multiplier = (solution[:, :, 0].sum(axis=1) - 1.) / solution[:, :, 1].sum(axis=1)
    weights = solution[:, :, 0] - solution[:, :, 1] * multiplier[:, None]
    residual = np.einsum("tij,tj->ti", system, weights) - rhs[:, :, 0] + multiplier[:, None] * support
    inaccurate = ((np.abs(weights.sum(axis=1) - 1.) > 1e-9)
                  | (np.abs(residual * support).max(axis=1) > 1e-9))
    if inaccurate.any():
        # A bordered solve avoids cancellation of the two normal-equation solutions.
        count = int(inaccurate.sum())
        kkt = np.zeros((count, width + 1, width + 1))
        kkt[:, :width, :width] = system[inaccurate]
        kkt[:, width, :width] = kkt[:, :width, width] = support[inaccurate]
        target = np.column_stack((rhs[inaccurate, :, 0], np.ones(count)))
        weights[inaccurate] = np.linalg.solve(kkt, target[..., None])[:, :width, 0]
    return weights


def fit_simplex(gram, xy, informative, baseline):
    """Fit nonnegative weights summing to one. Certify the unregularized duality gap."""
    n_targets, width = xy.shape
    weights = np.full_like(xy, 1. / width)
    if width == 1 or not informative.any():
        return weights
    diag = np.diagonal(gram, axis1=1, axis2=2)
    scale = np.maximum.reduce((diag.max(axis=1), np.abs(xy).max(axis=1), baseline, np.full(n_targets, 1e-12)))
    gram = gram / scale[:, None, None]
    xy = xy / scale[:, None]
    gradient = np.einsum("tij,tj->ti", gram, weights) - xy
    gap = (weights * gradient).sum(axis=1) - gradient.min(axis=1)
    pending = informative & (gap > 2e-11)
    active = np.ones_like(weights, dtype=bool)
    for _ in range(6 * width + 40):
        rows = np.flatnonzero(pending)
        if not len(rows):
            break
        proposed = _face_optimum(gram[rows], xy[rows], active[rows])
        outside = proposed < -1e-11
        blocked = outside.any(axis=1)
        if blocked.any():
            # Move to the first boundary on the line from the feasible point to the face optimum.
            targets = rows[blocked]
            current = weights[targets]
            direction = proposed[blocked] - current
            ratios = np.divide(current, -direction, where=outside[blocked], out=np.full_like(current, np.inf))
            leaving = np.argmin(ratios, axis=1)
            step = ratios[np.arange(len(targets)), leaving]
            updated = np.maximum(current + step[:, None] * direction, 0.)
            updated[np.arange(len(targets)), leaving] = 0.
            weights[targets] = updated / updated.sum(axis=1, keepdims=True)
            active[targets, leaving] = False
        if (~blocked).any():
            targets = rows[~blocked]
            updated = np.maximum(proposed[~blocked], 0.)
            updated /= updated.sum(axis=1, keepdims=True)
            weights[targets] = updated
            gradient = np.einsum("tij,tj->ti", gram[targets], updated) - xy[targets]
            entering = np.argmin(gradient, axis=1)
            gap = (gradient * updated).sum(axis=1) - gradient[np.arange(len(targets)), entering]
            expand = (gap > 2e-10) & ~active[targets, entering]
            active[targets[expand], entering[expand]] = True
            pending[targets[~expand]] = False
    gradient = np.einsum("tij,tj->ti", gram, weights) - xy
    gap = (weights * gradient).sum(axis=1) - gradient.min(axis=1)
    if np.any(informative & (gap > 2e-8)):
        raise ArithmeticError(f"Simplex fit failed its optimality certificate (relative gap {gap[informative].max()})")
    return weights
