"""Observed lag evidence and sufficient statistics for temporal source forecasts."""

from dataclasses import dataclass

import numpy as np
from scipy.stats import rankdata


def _observations(data):
    """Validate aligned observations and return globally standardized expression and intervals."""
    expression = np.asarray(data.X, dtype=float)
    times = np.asarray(data.times, dtype=float)
    genes, regulators = tuple(data.genes), tuple(data.regulators)
    if expression.ndim != 3:
        raise ValueError("Expression requires axes (series, observation, gene)")
    n_series, n_times, n_genes = expression.shape
    if not n_series or n_times < 2 or times.shape != (n_series, n_times):
        raise ValueError("Each series needs a time for every observation and at least two observations")
    if not np.isfinite(expression).all() or not np.isfinite(times).all():
        raise ValueError("Expression and sampling times must be finite")
    intervals = np.diff(times, axis=1)
    if np.any(intervals <= 0):
        raise ValueError("Sampling times must increase strictly within each series")
    if len(genes) != n_genes or len(set(genes)) != n_genes:
        raise ValueError("Expression requires a unique name for each gene")
    if len(set(regulators)) != len(regulators) or not set(regulators).issubset(genes):
        raise ValueError("Regulators must be unique members of the expression genes")
    mask = np.asarray(data.mask)
    if mask.dtype != bool or mask.shape != (n_genes, len(regulators)) or not mask.any():
        raise ValueError("Expected a boolean target-by-regulator mask with at least one eligible edge")
    standardized = (expression - expression.mean(axis=(0, 1))) / np.maximum(expression.std(axis=(0, 1)), 1e-9)
    gene_index = {gene: i for i, gene in enumerate(genes)}
    return standardized, intervals, [gene_index[r] for r in regulators]


def _distinct_columns(matrix):
    """Reuse exactly equal profiles so equivalent associations receive identical ranks."""
    unique, groups = np.unique(matrix.T, axis=0, return_inverse=True)
    if len(unique) == matrix.shape[1]:
        return matrix, np.arange(len(unique))
    return unique.T, groups.reshape(-1)


def _marginal_percentiles(previous, change, regulator_ids, mask):
    axes = (0, 1)
    own = previous - previous.mean(axis=1, keepdims=True)
    change = change - change.mean(axis=1, keepdims=True)
    n = own.shape[0] * own.shape[1]
    slope = (own * change).sum(axis=axes) / ((own * own).sum(axis=axes) + .05 * n)
    residual = change - slope * own
    response = residual / np.maximum(residual.std(axis=axes, keepdims=True), 1e-9)
    predictors = previous[:, :, regulator_ids]
    predictors = predictors - predictors.mean(axis=1, keepdims=True)
    predictors /= np.maximum(predictors.std(axis=axes, keepdims=True), 1e-9)
    x, regulators = _distinct_columns(predictors.reshape(n, len(regulator_ids)))
    y, targets = _distinct_columns(response.reshape(n, response.shape[-1]))
    x = x - x.mean(axis=0)
    y = y - y.mean(axis=0)
    norm = np.linalg.norm(x, axis=0)[:, None] * np.linalg.norm(y, axis=0)
    evidence = np.abs((x.T @ y / np.maximum(norm, 1e-12)).T)[np.ix_(targets, regulators)]
    ranks = rankdata(np.where(mask, evidence, np.inf), axis=1, method="average")
    return np.where(mask, (ranks - .5) / np.maximum(mask.sum(axis=1, keepdims=True), 1), .5)


def _center_and_scale(values):
    centered = values - values.mean(axis=1, keepdims=True)
    return centered / np.maximum(np.sqrt(np.mean(centered**2, axis=(0, 1))), 1e-10)


def _leaders(scores):
    """Select up to eight positive scores per target, excluding a tie that crosses rank eight."""
    values = np.where(np.isfinite(scores), scores, -np.inf)
    order = np.argsort(-values, axis=1, kind="stable")
    selected = order[:, :8]
    ordered = np.take_along_axis(values, selected, axis=1)
    include = ordered > 0.
    if values.shape[1] > 8:
        include &= ordered > np.take_along_axis(values, order[:, 8:9], axis=1)
    return selected, include.sum(axis=1)


@dataclass(frozen=True)
class Forecast:
    prediction: np.ndarray  # target × transition, regulatory contribution
    own_projection: np.ndarray  # target, projection onto penalized own history
    cross_response: np.ndarray  # target, inner product with residualized response


class TemporalEvidence:
    """Prepare all adjacent within-series transitions once per dataset."""

    def __init__(self, data):
        expression, intervals, regulator_ids = _observations(data)
        previous = expression[:, :-1]
        change = np.diff(expression, axis=1) / intervals[:, :, None]
        self.marginal = _marginal_percentiles(previous, change, regulator_ids, data.mask)
        n = intervals.size
        own = _center_and_scale(previous).reshape(n, len(data.genes))
        response = _center_and_scale(change).reshape(own.shape)
        self.predictors = own[:, regulator_ids]
        self.n_eff = max(1, n - len(expression) - 1)
        self.penalty = .5 * n

        # Eliminate the penalized own-history coefficient by a Schur complement.
        precision = 1. / ((own**2).sum(axis=0) + .05 * n)
        own_response = (own * response).sum(axis=0)
        self.objective = np.maximum((response**2).sum(axis=0) - own_response**2 * precision, 0.)
        self.informative = (self.objective > 1e-10) & data.mask.any(axis=1)
        self.gram = self.predictors.T @ self.predictors
        self.own_projection = (own.T @ self.predictors) * np.sqrt(precision[:, None])
        self.cross_response = response.T @ self.predictors - self.own_projection * (own_response * np.sqrt(precision))[:, None]

    def _systems(self, scores, minimum_size):
        """Yield batched normal equations for targets with equal neighbourhood sizes."""
        selected, widths = _leaders(scores)
        for width in np.unique(widths[widths >= minimum_size]):
            rows = np.flatnonzero((widths == width) & self.informative)
            if not len(rows):
                continue
            ids = selected[rows, :width]
            region = (rows[:, None], ids)
            votes = scores[region]
            projection = self.own_projection[region]
            system = self.gram[ids[:, :, None], ids[:, None, :]] - projection[:, :, None] * projection[:, None, :]
            diag = np.arange(width)
            system[:, diag, diag] += self.penalty / (1. + votes / votes.max(axis=1, keepdims=True))
            yield rows, ids, system, projection, self.cross_response[region]

    def forecast(self, scores):
        """Predict rates of expression change from a source's nominated regulators."""
        n_targets = len(scores)
        prediction = np.zeros((n_targets, len(self.predictors)))
        own_projection = np.zeros(n_targets)
        cross_response = np.zeros(n_targets)
        for rows, ids, system, projection, rhs in self._systems(scores, minimum_size=1):
            coefficients = np.linalg.solve(system, rhs[..., None])[..., 0]
            prediction[rows] = np.einsum("ntk,tk->tn", self.predictors[:, ids], coefficients)
            own_projection[rows] = np.einsum("tk,tk->t", coefficients, projection)
            cross_response[rows] = np.einsum("tk,tk->t", coefficients, rhs)
        return Forecast(prediction, own_projection, cross_response)

    def joint_percentiles(self, votes):
        """Redistribute the leaders' marginal percentiles by unique ridge contribution."""
        percentiles = self.marginal.copy()
        for rows, ids, system, _, rhs in self._systems(votes, minimum_size=2):
            inverse = np.linalg.inv(system)
            coefficients = np.einsum("tij,tj->ti", inverse, rhs)
            gain = coefficients**2 / np.diagonal(inverse, axis1=1, axis2=2)
            largest = gain.max(axis=1, keepdims=True)
            relative = np.divide(gain, largest, out=np.zeros_like(gain), where=largest > 0)
            # Algebraically equal contributions must tie after reordering regulators.
            strength = np.round(relative, 12)
            ordered = np.sort(strength, axis=1)
            equal = strength[:, :, None] == ordered[:, None, :]
            region = (rows[:, None], ids)
            ordered_percentiles = np.sort(self.marginal[region], axis=1)
            assigned = np.einsum("tij,tj->ti", equal, ordered_percentiles) / equal.sum(axis=2)
            percentiles[region] = np.where(np.ptp(strength, axis=1, keepdims=True) > 0., assigned, self.marginal[region])
        return percentiles
