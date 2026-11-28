"""Place each tool's scores on the same scale and identify distinct sources."""

import numpy as np


def minmax(raw, mask):
    """Normalize each tool across all eligible edges. Ineligible entries become NaN."""
    raw = np.asarray(raw, dtype=float)
    mask = np.asarray(mask)
    if (mask.dtype != bool or raw.ndim != 3 or raw.shape[1:] != mask.shape
            or not mask.any() or len(raw) == 0):
        raise ValueError("Expected tool matrices aligned to a nonempty candidate mask")
    values = raw[:, mask]
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Eligible scores must be finite and nonnegative")
    minimum = values.min(axis=1, keepdims=True)
    span = values.max(axis=1, keepdims=True) - minimum
    if (span == 0).any():
        raise ValueError("Every tool needs score variation across eligible edges for min-max scaling")
    scaled = np.full_like(raw, np.nan)
    scaled[:, mask] = (values - minimum) / span
    return scaled


def canonical_edges(data):
    """Use gene and regulator identifiers to break ties independently of input order."""
    targets, regulators = np.nonzero(data.mask)
    order = np.lexsort((np.asarray(data.regulators)[regulators], np.asarray(data.genes)[targets]))
    return targets[order], regulators[order]


def distinct_sources(scaled, edges):
    """Return the distinct normalized networks in canonical order and the tool-to-source map."""
    by_scores = {}
    for tool, matrix in enumerate(scaled):
        values = matrix[edges].astype(">f8", copy=False)
        values[values == 0] = 0.0  # Signed zeros are equivalent.
        by_scores.setdefault(values.tobytes(), []).append(tool)
    groups = [by_scores[key] for key in sorted(by_scores)]
    inverse = np.empty(len(scaled), dtype=int)
    for source, tools in enumerate(groups):
        inverse[tools] = source
    return scaled[[tools[0] for tools in groups]], inverse


def mean_scores(scores):
    """Equal multisets of votes give exactly tied means, regardless of source order."""
    if len(scores) > 2:
        scores = scores.copy(order="F")
        scores.sort(axis=0)
    return scores.mean(axis=0)
