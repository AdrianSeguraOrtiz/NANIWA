"""Generate the registered source panels for the robustness study."""
import hashlib
import itertools
import json
from dataclasses import dataclass

import numpy as np
from scipy.special import ndtr

from scipy.stats import rankdata


@dataclass(frozen=True, eq=False)
class Panel:
    scenario: str
    names: tuple[str, ...]
    sources: np.ndarray
    repeat: int | None = None
    reference: str = "clean"
    founder: str | None = None
    matched: str | None = None
    exact_copy: bool = False
    specialist_targets: np.ndarray | None = None
    focus_source: str | None = None

    @property
    def id(self):
        return self.scenario if self.repeat is None else f"{self.scenario}-r{self.repeat}"


def rng_for(seed, *parts):
    key = json.dumps([seed, *parts], ensure_ascii=False).encode()
    return np.random.Generator(np.random.PCG64(int.from_bytes(hashlib.sha256(key).digest()[:16], "big")))


def shuffled(matrix, mask, rng, fraction=None):
    copy = matrix.copy()
    for target, eligible in enumerate(mask):
        columns = np.flatnonzero(eligible)
        perm = rng.permutation(columns)
        if fraction is None:
            copy[target, columns] = matrix[target, perm]
        elif len(columns) >= 2:
            # A cyclic shift moves each selected position. Tied scores can stay equal.
            moved = perm[:max(2, int(round(len(columns) * fraction)))]
            copy[target, moved] = matrix[target, np.roll(moved, 1)]
    return copy


def _null_families(tools, raw, mask, dataset_id, spec, founders):
    for repeat, (name, founder_scores) in founders.items():
        founder_id = f"null-1-r{repeat}"
        for count in spec["null_counts"]:
            if count == 1:
                continue
            matched = f"null-{count}-r{repeat}"
            names = (*tools, name, *(f"{name}_copy{i}" for i in range(2, count + 1)))
            yield Panel(f"null-exact-{count}", names, np.concatenate((raw, [founder_scores] * count)),
                        repeat, founder=founder_id, matched=matched, exact_copy=True)
            for fraction in spec["near_copy_fractions"]:
                relatives = [founder_scores]
                for i in range(2, count + 1):
                    rng = rng_for(spec["seed"], dataset_id, "null-relative", repeat, i)
                    relatives.append(shuffled(founder_scores, mask, rng, fraction))
                yield Panel(f"null-near-{count}-{fraction:g}", names, np.concatenate((raw, relatives)),
                            repeat, founder=founder_id, matched=matched)


def _specialist(tools, raw, mask, truth, dataset_id, spec, repeat):
    settings = spec["specialist"]
    targets = np.flatnonzero(mask.any(axis=1))
    count = max(1, int(round(len(targets) * settings["target_fraction"])))
    if len(targets) >= 2:
        count = min(len(targets) - 1, count)
    draw = rng_for(spec["seed"], dataset_id, "specialist-targets", repeat).permutation(targets)
    selected = np.zeros(len(mask), dtype=bool)
    selected[draw[:count]] = True
    shift = np.where(selected, settings["specialist_shift"], settings["background_shift"])
    noise = rng_for(spec["seed"], dataset_id, "specialist-errors", repeat).standard_normal(mask.shape)
    scores = np.where(mask, ndtr(noise + shift[:, None] * truth), 0.0)
    return Panel("specialist", (*tools, "specialist"), np.concatenate((raw, scores[None])), repeat,
                 specialist_targets=selected, focus_source="specialist")


def _growth(tools, raw, mask, truth, dataset_id, spec, repeat):
    settings = spec["panel_growth"]
    sizes = settings["sizes"]
    if sizes[0] != len(tools) or sizes != sorted(set(sizes)) or sizes[-1] <= len(tools):
        raise ValueError("Panel growth sizes must start with the real panel and increase")
    total = sizes[-1] - len(tools)
    order = rng_for(spec["seed"], dataset_id, "growth-order", repeat).permutation(total)
    shared = rng_for(spec["seed"], dataset_id, "growth-common", repeat).standard_normal(mask.shape)
    rho = settings["shared_error_correlation"]
    # Each class retains the exact scores of the paired source at this target.
    strata = [(t, np.flatnonzero(row & (truth[t] == label)))
              for t, row in enumerate(mask) for label in (False, True)]
    independent_sources, shared_sources, names = [], [], list(tools)
    for source in order:
        label = f"synthetic-{source + 1}"
        private = rng_for(spec["seed"], dataset_id, "growth-errors", repeat, label).standard_normal(mask.shape)
        scores = np.where(mask, ndtr(private + settings["positive_shift"] * truth), 0.0)
        shared_order = np.sqrt(rho) * shared + np.sqrt(1 - rho) * private
        reordered = np.zeros_like(scores)
        for t, cols in strata:
            positions = cols[np.argsort(shared_order[t, cols], kind="stable")]
            reordered[t, positions] = np.sort(scores[t, cols])
        independent_sources.append(scores)
        shared_sources.append(reordered)
        names.append(label)
        if len(names) in sizes:
            yield Panel(f"scale-complementary-{len(names)}", tuple(names),
                        np.concatenate((raw, independent_sources)), repeat)
            yield Panel(f"scale-shared-{len(names)}", tuple(names),
                        np.concatenate((raw, shared_sources)), repeat, matched=f"scale-complementary-{len(names)}-r{repeat}")


def panels(tools, raw, mask, dataset_id, spec, truth):
    """Generate all registered panels from predictions and boolean benchmark labels."""
    if truth.shape != mask.shape or truth.dtype != np.bool_:
        raise ValueError("Robustness truth must be a boolean matrix on the candidate axes")
    tools = tuple(tools)
    yield Panel("clean", tools, raw)
    for i, tool in enumerate(tools):
        names = (*tools, tool + "_copy")
        yield Panel(f"copy-{tool}", names, np.concatenate((raw, raw[[i]])), exact_copy=True)
        for repeat, fraction in itertools.product(spec["repeat_ids"], spec["near_copy_fractions"]):
            rng = rng_for(spec["seed"], dataset_id, "near", tool, repeat)
            clone = shuffled(raw[i], mask, rng, fraction)
            yield Panel(f"near-{tool}-{fraction:g}", names, np.concatenate((raw, clone[None])), repeat)
        kept = [j for j in range(len(tools)) if j != i]
        yield Panel(f"leave-{tool}", tuple(tools[j] for j in kept), raw[kept])

    founders = {}
    for repeat in spec["repeat_ids"]:
        donors = rng_for(spec["seed"], dataset_id, "donors", repeat).permutation(len(tools))
        nulls, labels = [], []
        for k in range(1, max(spec["null_counts"]) + 1):
            donor = donors[(k - 1) % len(donors)]
            rng = rng_for(spec["seed"], dataset_id, "null", k, repeat)
            nulls.append(shuffled(raw[donor], mask, rng))
            labels.append(f"null{k}-{tools[donor]}")
            if k == 1:
                founders[repeat] = (labels[0], nulls[0])
            if k in spec["null_counts"]:
                yield Panel(f"null-{k}", (*tools, *labels), np.concatenate((raw, nulls)), repeat,
                            founder=f"null-1-r{repeat}" if k > 1 else None)

    # Pooled correlation of within-target standardized ranks. Ties follow input order.
    midranks = rankdata(np.where(mask, raw, np.inf), axis=-1, method="average")
    counts = mask.sum(axis=-1, keepdims=True)
    centered = np.where(mask, midranks - (counts + 1) / 2, 0.)
    deviation = np.sqrt(np.sum(centered ** 2, axis=-1, keepdims=True) / np.maximum(counts, 1))
    ranks = np.divide(centered, deviation, out=np.zeros_like(centered), where=deviation > 1e-14)[:, mask]
    norms = np.linalg.norm(ranks, axis=1, keepdims=True)
    unit = np.divide(ranks, norms, out=np.zeros_like(ranks), where=norms > 0)
    sim = unit @ unit.T
    subsets = list(itertools.combinations(range(len(tools)), spec["subset_size"]))
    agreement = [np.round(np.mean([sim[a, b] for a, b in itertools.combinations(s, 2)]), 14) for s in subsets]
    for label, index in (("low", np.argmin(agreement)), ("high", np.argmax(agreement))):
        indices = list(subsets[index])
        yield Panel(f"agreement-{label}", tuple(tools[j] for j in indices), raw[indices])

    yield from _null_families(tools, raw, mask, dataset_id, spec, founders)
    for repeat in spec["repeat_ids"]:
        yield _specialist(tools, raw, mask, truth, dataset_id, spec, repeat)
        yield from _growth(tools, raw, mask, truth, dataset_id, spec, repeat)
