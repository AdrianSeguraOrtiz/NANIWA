"""Consensus of normalized source networks conditioned on observed expression time courses."""

from collections import OrderedDict
import hashlib
from threading import Lock

import numpy as np

from .correction import correct_vote
from .expression import TemporalEvidence
from .hierarchy import predictive_families
from .options import ConsensusOptions
from .scale import canonical_edges, distinct_sources, mean_scores, minmax


class TemporalConsensus:
    """Score multiple tool panels against the same target-by-regulator space and trajectories."""

    def __init__(self, data):
        self.evidence = TemporalEvidence(data)
        self.mask = np.asarray(data.mask, dtype=bool).copy()
        self.edges = canonical_edges(data)
        self._cache_lock = Lock()
        self._forecasts = OrderedDict()
        self._partition_key = None
        self._partition = None

    def _families(self, sources):
        # A study repeatedly inserts or removes a few tools from the same panel.
        # Cache source-specific fits independently and retain only the most recent partition.
        keys = tuple(hashlib.blake2b(np.ascontiguousarray(s).view(np.uint8), digest_size=24).digest()
                     for s in sources)
        if keys == self._partition_key:
            return self._partition
        forecasts = []
        for key, source in zip(keys, sources, strict=True):
            if key not in self._forecasts:
                self._forecasts[key] = self.evidence.forecast(source)
            self._forecasts.move_to_end(key)
            forecasts.append(self._forecasts[key])
            if len(self._forecasts) > 48:
                self._forecasts.popitem(last=False)
        self._partition = predictive_families(forecasts, self.evidence)
        self._partition_key = keys
        return self._partition

    def _base_vote(self, scaled, vote, source_vote_weights):
        if vote == "all":
            if source_vote_weights is not None:
                source_vote_weights.fill(1. / len(scaled))
            return mean_scores(scaled)

        sources, inverse = distinct_sources(scaled, self.edges)
        if vote == "distinct":
            base = mean_scores(sources)
            shares = np.full(len(sources), 1. / len(sources))
        else:
            with self._cache_lock:
                families = self._families(sources)
            base = mean_scores(np.stack([mean_scores(sources[list(family)]) for family in families]))
            shares = np.empty(len(sources))
            for family in families:
                shares[list(family)] = 1. / (len(families) * len(family))
        if source_vote_weights is not None:
            counts = np.bincount(inverse)
            source_vote_weights[:] = (shares[inverse] / counts[inverse])[:, None]
        return base

    def score(self, raw, options=ConsensusOptions(), source_vote_weights=None):
        """Return target × regulator confidence scores and optionally record pre-correction source shares."""
        scaled = minmax(raw, self.mask)
        if source_vote_weights is not None and (
            not isinstance(source_vote_weights, np.ndarray) or source_vote_weights.shape != scaled.shape[:2]
            or source_vote_weights.dtype.kind != "f" or not source_vote_weights.flags.writeable
        ):
            raise ValueError("source_vote_weights must be a writable floating-point array of shape (tools, targets)")
        components = options.components
        base = self._base_vote(scaled, components.vote, source_vote_weights)
        if components.evidence == "none":
            return base
        percentiles = (self.evidence.marginal if components.evidence == "marginal"
                       else self.evidence.joint_percentiles(base))
        return correct_vote(base, percentiles, attenuate=components.intensity == "global")
