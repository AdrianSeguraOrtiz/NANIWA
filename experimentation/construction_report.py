"""Predeclared, paired construction contrasts aggregated by generating topology."""
from collections import defaultdict
import hashlib
import json

import numpy as np


def registered_recipe(recipe_path, protocol_path, methods, individuals):
    """Match the scored methods and, when present, the originating study protocol to the recipe."""
    raw = recipe_path.read_bytes()
    recipe = json.loads(raw)
    expected = {*recipe["path"], *recipe["controls"], *individuals}
    if methods != expected:
        raise ValueError(f"Construction method IDs differ from the recipe: missing {sorted(expected - methods)}, "
                         f"unexpected {sorted(methods - expected)}")
    provenance = {"construction_recipe_sha256": hashlib.sha256(raw).hexdigest()}
    if protocol_path.is_file():
        recorded = protocol_path.read_bytes()
        protocol = json.loads(recorded)
        if (protocol.get("phase") != "construction" or
                protocol.get("recipes", {}).get(recipe_path.name) != provenance["construction_recipe_sha256"]):
            raise ValueError("The scoring protocol and the registered construction recipe differ")
        provenance["scoring_protocol_sha256"] = hashlib.sha256(recorded).hexdigest()
    return recipe, provenance


def construction_contrasts(metrics, comparisons, seed):
    """Pair scores within views before giving each generating graph equal weight."""
    by_view = defaultdict(dict)
    for row in metrics:
        by_view[row["graph_id"], row["id"]][row["method"]] = row["aupr"]
    graphs = sorted({g for g, _ in by_view})
    draws = np.random.default_rng(seed).integers(len(graphs), size=(5000, len(graphs)))
    results = []
    for label, method, reference in comparisons:
        paired = defaultdict(list)
        for (graph, view), scores in by_view.items():
            if method not in scores or reference not in scores:
                raise ValueError(f"{view}: missing registered construction contrast {method} vs {reference}")
            paired[graph].append((scores[method], scores[reference]))
        means = np.array([np.mean(paired[g], axis=0) for g in graphs])
        diff = means[:, 0] - means[:, 1]
        lo, hi = np.quantile(diff[draws].mean(axis=1), (.025, .975)) if len(graphs) > 1 else (None, None)
        results.append(dict(contrast=label, method=method, reference=reference, n_views=len(by_view),
                            n_graphs=len(graphs), aupr=means[:, 0].mean(), reference_aupr=means[:, 1].mean(),
                            delta_aupr=diff.mean(), ci_low=lo, ci_high=hi,
                            graphs_higher=int((diff > 1e-12).sum()), graphs_tied=int((abs(diff) <= 1e-12).sum()),
                            graphs_lower=int((diff < -1e-12).sum())))
    return results
