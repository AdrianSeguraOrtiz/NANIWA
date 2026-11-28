"""Paired robustness comparisons across repeated panels, datasets and graph topologies."""
import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from statistics import fmean

import numpy as np


COMPARATOR = "mean_minmax"
POLICIES = ("naniwa", COMPARATOR)
ENDPOINTS = ("aupr", "top1_by_target")
EFFECTS = ("gap", "naniwa_change", "comparator_change", "change_gap", "abs_change_gap")
VALUES = ("naniwa", "comparator_score", "reference_naniwa", "reference_comparator", *EFFECTS)
GROUP = ("scenario", "contrast", "scope", "metric", "comparator")
INTERVALS = tuple(f"{field}_ci_{bound}" for field in EFFECTS for bound in ("low", "high"))


@dataclass(frozen=True)
class Panel:
    name: str
    scenario: str
    repeat: str
    reference: str
    founder: str
    matched: str
    specialist: frozenset
    tools: tuple


@dataclass(frozen=True)
class Comparison:
    scenario: str
    repeat: str
    panel: str
    reference: str
    contrast: str
    specialist: frozenset


class PanelPlan:
    def __init__(self, path, targets):
        self.methods = {}
        self.panels = {}
        self.targets = frozenset(targets)
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream, delimiter="\t")
            required = {"method", "panel", "policy", "tools", "rank_correlation_to_clean", "scenario", "repeat",
                        "reference_panel", "founder_panel", "matched_panel", "specialist_targets"}
            if missing := required.difference(reader.fieldnames or ()):
                raise ValueError(f"{path}: missing panel fields {sorted(missing)}")
            for row in reader:
                if row["policy"] not in POLICIES:
                    raise ValueError(f"{path}: unexpected robustness policy {row['policy']!r} for {row['panel']}")
                focus = json.loads(row["specialist_targets"] or "[]")
                if (not isinstance(focus, list) or not all(isinstance(t, str) for t in focus)
                        or len(set(focus)) != len(focus) or not set(focus) <= self.targets):
                    raise ValueError(f"{path}: invalid specialist targets for {row['panel']}")
                members = tuple(row["tools"].split(","))
                panel = Panel(row["panel"], row["scenario"], row["repeat"], row["reference_panel"] or "clean",
                              row["founder_panel"], row["matched_panel"], frozenset(focus), members)
                key = (panel.name, row["policy"])
                if (not all((panel.name, panel.scenario, row["method"], row["policy"])) or key in self.methods
                        or "" in members or len(set(members)) != len(members)):
                    raise ValueError(f"{path}: duplicate or invalid panel/policy {key}")
                if self.panels.setdefault(panel.name, panel) != panel:
                    raise ValueError(f"{path}: panel metadata differ across policies for {panel.name}")
                self.methods[key] = row["method"]
        if len(set(self.methods.values())) != len(self.methods) or "clean" not in self.panels:
            raise ValueError(f"{path}: expected distinct method IDs and a clean panel")
        if len({(p.scenario, p.repeat) for p in self.panels.values()}) != len(self.panels):
            raise ValueError(f"{path}: repeated scenario/repetition")
        self.comparisons = []
        self.omissions = {}
        for p in self.panels.values():
            if any((p.name, policy) not in self.methods for policy in POLICIES):
                raise ValueError(f"{path}: missing required robustness policy for {p.name}")
            if (p.scenario == "specialist" and not p.specialist) or p.specialist == self.targets:
                raise ValueError(f"{path}: specialist and other strata must both contain targets for {p.name}")
            for kind, ref in (("reference", p.reference), ("founder", p.founder), ("matched", p.matched)):
                if not ref:
                    continue
                if ref not in self.panels or (kind != "reference" and self.panels[ref].repeat != p.repeat):
                    raise ValueError(f"{path}: unknown or unmatched {kind} panel {ref} for {p.name}")
                if kind == "matched" and len(self.panels[ref].tools) != len(p.tools):
                    raise ValueError(f"{path}: matched panels must contain the same number of tools for {p.name}")
                if kind == "founder" and not set(self.panels[ref].tools) < set(p.tools):
                    raise ValueError(f"{path}: {p.name} must contain the founder panel and additional tools")
                self.comparisons.append(Comparison(p.scenario, p.repeat, p.name, ref, kind, p.specialist))
            if p.name.startswith("leave-"):
                clean, kept = set(self.panels["clean"].tools), set(p.tools)
                omitted = p.name.removeprefix("leave-")
                if clean != kept | {omitted} or omitted in kept:
                    raise ValueError(f"{path}: {p.name} must omit only {omitted} from clean")
                self.omissions[omitted] = f"reintroduce-{omitted}"
                self.comparisons.append(Comparison(self.omissions[omitted], p.repeat, "clean", p.name,
                                                   "reintroduction", frozenset()))
        self.needed = defaultdict(set)
        for c in self.comparisons:
            for _, targets_in_scope in self.scopes(c):
                for panel in (c.panel, c.reference):
                    for policy in POLICIES:
                        self.needed[self.methods[panel, policy]].add(targets_in_scope)
        self.scores = {}

    def scopes(self, comparison):
        yield "all", None
        if comparison.specialist:
            yield "specialist", comparison.specialist
            yield "other", self.targets - comparison.specialist

    def signature(self):
        """Only dataset-specific tool composition and the identities of specialist targets may vary."""
        return (self.methods, {(c.scenario, c.repeat, c.panel, c.reference, c.contrast, bool(c.specialist))
                               for c in self.comparisons})

    def score(self, method, result, positive, cache, average_precision):
        if method not in self.needed:
            raise ValueError(f"Ranking absent from robustness panel metadata: {method}")
        counts = Counter(dst for _, dst in cache.truth_keys)
        sources = cache.candidate_space.sources
        candidates = {dst: len(sources) - int(dst in sources) for dst in self.targets}
        # A target with all-zero scores starts with all eligible candidates tied for first.
        best = {dst: (0.0, counts[dst], n) for dst, n in candidates.items() if 0 < counts[dst] < n}
        for edge in positive:
            dst = edge.target
            if dst not in best:
                continue
            high, hits, ties = best[dst]
            hit = (edge.source, dst) in cache.truth_keys
            if edge.score > high:
                best[dst] = (edge.score, int(hit), 1)
            elif edge.score == high:
                best[dst] = (high, hits + hit, ties + 1)
        for subset in self.needed[method]:
            targets = self.targets if subset is None else subset
            n_candidates = sum(candidates[t] for t in targets)
            n_truth = sum(counts[t] for t in targets)
            eligible = best.keys() & targets
            top1 = fmean(best[t][1] / best[t][2] for t in sorted(eligible)) if eligible else None
            ap = result["aupr"] if subset is None else average_precision(
                truth_keys={edge for edge in cache.truth_keys if edge[1] in targets},
                prediction_scores={(r.source, r.target): r.score for r in positive if r.target in targets},
                n_candidates=n_candidates,
            )
            self.scores[method, subset] = dict(aupr=ap, top1_by_target=top1, n_targets=len(targets),
                                               n_informative_targets=len(eligible), n_candidates=n_candidates,
                                               n_truth_edges=n_truth)

    def observations(self, factors, predicted_methods):
        if set(self.methods.values()) != set(predicted_methods):
            raise ValueError(f"{factors['id']}: predictions and methods.tsv do not cover identical methods")
        rows = []
        for c in self.comparisons:
            for scope, targets in self.scopes(c):
                current = {p: self.scores[self.methods[c.panel, p], targets] for p in POLICIES}
                reference = {p: self.scores[self.methods[c.reference, p], targets] for p in POLICIES}
                counts = {k: current["naniwa"][k] for k in ("n_targets", "n_informative_targets",
                                                           "n_candidates", "n_truth_edges")}
                for metric in ENDPOINTS:
                    n, b = current["naniwa"][metric], current[COMPARATOR][metric]
                    n0, b0 = reference["naniwa"][metric], reference[COMPARATOR][metric]
                    scores = (n, b, n0, b0)
                    if any(v is None for v in scores):
                        if not all(v is None for v in scores):
                            raise ValueError(f"{factors['id']}: inconsistent coverage for {c.scenario}/{scope}")
                        effects = (None,) * len(EFFECTS)
                    else:
                        dn, db = n - n0, b - b0
                        effects = (n - b, dn, db, dn - db, abs(dn) - abs(db))
                    rows.append(dict(factors, scenario=c.scenario, repeat=c.repeat, panel=c.panel,
                                     reference_panel=c.reference, contrast=c.contrast, scope=scope, metric=metric,
                                     comparator=COMPARATOR, **counts, **dict(zip(VALUES, (*scores, *effects)))))
        return rows


def read_source_aupr(path, omissions):
    scores = {}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if not {"panel", "source", "aupr"} <= set(reader.fieldnames or ()):
            raise ValueError(f"{path}: expected panel,source,aupr")
        for row in reader:
            if row["panel"] != "clean" or row.get("scope", "all") != "all":
                continue
            source = row["source"]
            score = float(row["aupr"])
            if source in scores or not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError(f"{path}: repeated or invalid clean AP for {source}")
            scores[source] = score
    if not omissions or set(scores) != set(omissions):
        raise ValueError(f"{path}: clean source diagnostics and tool omissions do not match")
    return scores


def best_reintroductions(rows, diagnostics):
    """Select a real source on other graphs only, weighting each graph equally."""
    if not diagnostics:
        return []
    by_graph = defaultdict(list)
    for graph, scores in diagnostics.values():
        by_graph[graph].append(scores)
    if len(by_graph) < 2:
        return []
    tools = sorted(next(iter(diagnostics.values()))[1])
    means = {g: {t: fmean(s[t] for s in scores) for t in tools} for g, scores in by_graph.items()}
    best = {g: min(tools, key=lambda t: (-fmean(means[h][t] for h in means if h != g), t)) for g in means}
    return [dict(row, scenario="reintroduce-strongest") for row in rows
            if row["contrast"] == "reintroduction" and row["scenario"] == "reintroduce-" + best[row["graph_id"]]]


def aggregate(rows, fields, count):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in fields)].append(row)
    averaged = []
    for key, children in sorted(groups.items()):
        valid = [r for r in children if r["naniwa"] is not None]
        result = dict(zip(fields, key))
        result.update({f"n_{count}": len(children), f"n_valid_{count}": len(valid)})
        for child_count in ("repeats", "datasets"):
            if f"n_{child_count}" in children[0]:
                for prefix in ("n_", "n_valid_"):
                    result[prefix + child_count] = sum(r[prefix + child_count] for r in children)
        result.update({k: fmean(r[k] for r in valid) if valid else None for k in VALUES})
        averaged.append(result)
    return averaged


def write_report(out, rows, factors, diagnostics, seed, table):
    rows = rows + best_reintroductions(rows, diagnostics)
    table(out / "robustness_by_panel.tsv", rows, rows[0].keys())
    datasets = aggregate(rows, (*factors, *GROUP), "repeats")
    table(out / "robustness_by_dataset.tsv", datasets, datasets[0].keys())
    graphs = aggregate(datasets, ("graph_id", *GROUP), "datasets")
    table(out / "robustness_by_topology.tsv", graphs, graphs[0].keys())
    summary = aggregate(graphs, GROUP, "graphs")
    groups = defaultdict(list)
    for row in graphs:
        if row["naniwa"] is not None:
            groups[tuple(row[k] for k in GROUP)].append(row)
    draws = {}
    for row in summary:
        group = groups[tuple(row[k] for k in GROUP)]
        n = len(group)
        if n > 1:
            if n not in draws:
                rng = np.random.Generator(np.random.PCG64(seed))
                draws[n] = rng.integers(n, size=(5000, n))
            values = np.array([[g[field] for field in EFFECTS] for g in group])
            low, high = np.quantile(values[draws[n]].mean(axis=1), (0.025, 0.975), axis=0)
            for i, field in enumerate(EFFECTS):
                row[field + "_ci_low"], row[field + "_ci_high"] = low[i], high[i]
        else:
            row.update(dict.fromkeys(INTERVALS))
    table(out / "robustness_summary.tsv", summary, summary[0].keys())
    # Count only the primary all-target comparison against the seven-source panel.
    primary = [r for r in summary if r["metric"] == "aupr" and r["scope"] == "all"
               and r["contrast"] == "reference" and r["scenario"] != "clean" and r["n_valid_graphs"] > 0]
    counts = []
    for effect in ("gap", "change_gap"):
        values = [r[effect] for r in primary]
        counts.append(dict(comparator=COMPARATOR, effect=effect, n_scenarios=len(values),
                           higher=sum(v > 1e-12 for v in values), tie=sum(abs(v) <= 1e-12 for v in values),
                           lower=sum(v < -1e-12 for v in values)))
    table(out / "robustness_counts.tsv", counts, counts[0].keys())
