"""Measure individual quality, shared ranking errors and incremental true first choices."""
from dataclasses import dataclass

import numpy as np
from scipy.stats import rankdata


def _average_precision(scores, labels):
    """ANDREA directed AP over complete score groups, including the bottom zero group."""
    positives = labels.sum()
    if positives == 0 or positives == len(labels):
        return None
    order = np.argsort(-scores, kind="stable")
    ranked = scores[order]
    ends = np.r_[np.flatnonzero(ranked[1:] != ranked[:-1]), len(ranked) - 1]
    hits = np.cumsum(labels[order])[ends]
    recall = hits / positives
    precision = hits / (ends + 1)
    return float(np.cumsum(np.diff(recall, prepend=0.0) * precision)[-1])


def _error_ranks(scores, mask, truth):
    """Standardize midranks inside each target and truth class, then pool eligible edges."""
    residual = np.zeros(mask.shape)
    for label in (False, True):
        stratum = mask & (truth == label)
        count = stratum.sum(axis=1, keepdims=True)
        ranks = rankdata(np.where(stratum, scores, np.inf), axis=1, method="average")
        centered = np.where(stratum, ranks - (count + 1) / 2, 0.0)
        deviation = np.sqrt((centered ** 2).sum(axis=1, keepdims=True) / np.maximum(count, 1))
        residual += np.divide(centered, deviation, out=np.zeros_like(centered), where=deviation > 0)
    eligible = residual[mask]
    norm = np.sqrt(np.sum(eligible ** 2))
    return eligible / norm if norm > 0 else None


@dataclass
class _Profile:
    aupr: float | None
    top1_by_target: float | None
    unique_true: np.ndarray
    error_ranks: np.ndarray | None


class _Scope:
    def __init__(self, mask, truth):
        self.mask = mask
        self.truth = truth
        positives = truth.sum(axis=1)
        self.informative = (positives > 0) & (positives < mask.sum(axis=1))
        self.counts = dict(n_targets=mask.shape[0], n_candidates=int(mask.sum()), n_truth_edges=int(positives.sum()),
                           n_informative_targets=int(self.informative.sum()))

    def profile(self, scores):
        mask, truth = self.mask, self.truth
        scores = np.asarray(scores, dtype=float)
        values = scores[mask]
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("Individual source scores must be finite and nonnegative on candidate edges")
        maxima = np.max(scores, axis=1, keepdims=True, where=mask, initial=-np.inf)
        top = mask & (scores == maxima)
        ties = top.sum(axis=1)
        top_truth = top & truth
        idx = self.informative
        top1 = float(np.mean(top_truth.sum(axis=1)[idx] / ties[idx])) if idx.any() else None
        unique_true = top_truth & (idx & (ties == 1))[:, None]
        return _Profile(_average_precision(values, truth[mask]), top1, unique_true, _error_ranks(scores, mask, truth))


class SourceDiagnostics:
    """Profile the real tools once and measure every panel in its registered source order."""
    def __init__(self, tools, raw, mask, truth):
        if mask.dtype != bool or truth.dtype != bool or truth.shape != mask.shape or np.any(truth & ~mask):
            raise ValueError("Expected boolean directed truth on eligible candidate edges")
        self.all_targets = _Scope(mask, truth)
        self.real = {name: self.all_targets.profile(scores) for name, scores in zip(tools, raw, strict=True)}

    def rows(self, panel):
        yield from self._rows(panel, self.all_targets, "all", None)
        if panel.specialist_targets is not None:
            for label, selected in (("specialist", panel.specialist_targets), ("other", ~panel.specialist_targets)):
                scope = _Scope(self.all_targets.mask[selected], self.all_targets.truth[selected])
                yield from self._rows(panel, scope, label, selected)

    def _rows(self, panel, scope, label, selected):
        previous = []
        covered = np.zeros_like(scope.mask)
        for index, (name, scores) in enumerate(zip(panel.names, panel.sources, strict=True), start=1):
            real = name in self.real
            if selected is None:
                profile = self.real[name] if real else scope.profile(scores)
            else:
                profile = scope.profile(scores[selected])
            row = dict(panel=panel.id, scenario=panel.scenario, repeat=panel.repeat, source=name, source_index=index,
                       source_type="real" if real else "added", scope=label, **scope.counts,
                       aupr=profile.aupr, top1_by_target=profile.top1_by_target,
                       n_unique_true_top1=int(profile.unique_true.sum()),
                       n_new_unique_true_top1=int(np.count_nonzero(profile.unique_true & ~covered)))
            covered |= profile.unique_true
            # Unit length makes the dot product the pooled correlation. Rows with
            # no variation inside any target and truth class have no defined pair.
            pairs = [(float(np.clip(np.einsum("i,i->", profile.error_ranks, ranks), -1, 1)), added)
                     for ranks, added in previous if profile.error_ranks is not None and ranks is not None]
            for suffix, values in (("previous", [rho for rho, _ in pairs]),
                                   ("previous_added", [rho for rho, added in pairs if added])):
                row["n_error_correlations_" + suffix] = len(values)
                row["mean_error_rank_correlation_" + suffix] = float(np.mean(values)) if values else None
            previous.append((profile.error_ranks, not real))
            yield row
