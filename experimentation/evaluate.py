"""Score a phase's frozen predictions using ANDREA-Osaka's directed evaluator."""
import argparse
import csv
import gzip
import hashlib
import json
import math
import os
from collections import defaultdict
from importlib.metadata import version
from itertools import groupby
from operator import itemgetter
from pathlib import Path
from statistics import fmean
from tempfile import TemporaryDirectory

import numpy as np
from andrea.core.commands.evaluate_inference import evaluation as osaka
from andrea.core.shared.dataset_identity import fingerprint_dataset_content

if __package__:
    from . import construction_report, robustness_report
else:
    import construction_report
    import robustness_report

HERE = Path(__file__).resolve().parent
INDIVIDUALS = set(json.loads((HERE / "03-final-comparison/comparison.json").read_text())["individual_tools"])
FACTORS = ("id", "graph_id", "simulator", "design", "noise", "n_series", "n_timepoints")
METRICS = ("aupr", "auroc", "f1_at_truth_count", "epr_at_truth_count")


def table(path, rows, fields):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def view_truth(view_id, dataset_path, truth_path):
    """Check view identity and the full directed TF → gene universe against the truth package."""
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    truth_by_context, manifest = osaka._load_truth_networks(truth_path)
    if dataset["dataset"]["spec"]["id"] != view_id or manifest["dataset_id"] != view_id:
        raise ValueError(f"{view_id}: dataset and truth must both identify this view")
    expression = dataset_path.parent / dataset["dataset"]["expression_matrix"]
    extras = {k: dataset_path.parent / v for k, v in dataset["extras"].items() if v is not None}
    fingerprint = fingerprint_dataset_content(expression_path=expression, extras=extras)
    if fingerprint != manifest["dataset_fingerprint"]:
        raise ValueError(f"{view_id}: expression or metadata differ from the truth package fingerprint")
    truth = truth_by_context["global"]
    space = truth.candidate_space
    with expression.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream, delimiter="\t")
        next(reader)
        genes = {row[0] for row in reader}
    regulators = set(extras["tf_list"].read_text(encoding="utf-8-sig").splitlines())
    if space.targets != genes or space.sources != regulators or space.gene_universe != genes or space.allow_self_edges:
        raise ValueError(f"{view_id}: ground truth and dataset define different candidate interactions")
    return truth, osaka._build_truth_level_cache(truth=truth, level="directed")


def evaluate_view(factors, predictions, truth, cache, plan=None):
    """Keep one complete method in memory. Zeros share ANDREA's bottom score group."""
    methods = {}
    open_file = gzip.open if predictions.suffix == ".gz" else open
    with open_file(predictions, "rt", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"source", "target", "score", "tool_id"}.issubset(reader.fieldnames or ()):
            raise ValueError(f"{predictions}: expected source,target,score,tool_id")
        for method, group in groupby(reader, key=itemgetter("tool_id")):
            if not method or method in methods:
                raise ValueError(f"{predictions}: each method must occupy one contiguous block")
            seen, positive = set(), []
            for row in group:
                src, dst = row["source"], row["target"]
                score = float(row["score"])
                if (src, dst) in seen or src == dst or src not in truth.candidate_space.sources or dst not in truth.candidate_space.targets:
                    raise ValueError(f"{predictions}: repeated or ineligible edge {src} -> {dst} for {method}")
                if not math.isfinite(score) or score < 0:
                    raise ValueError(f"{predictions}: invalid score for {method}")
                seen.add((src, dst))
                if score > 0:
                    positive.append(osaka.NetworkRow(src, dst, score, "?", "global", method))
            if len(seen) != cache.n_candidates:
                raise ValueError(f"{predictions}: incomplete ranking for {method}: {len(seen)}/{cache.n_candidates}")
            result = osaka._evaluate_pairing(
                tool_id=method, capabilities=osaka.ToolCapabilities(method, "custom", True, False, "none"),
                prediction_rows=positive, excluded_prediction_rows=[], prediction_context="global",
                truth=truth, level="directed", truth_cache=cache,
            )
            if result["status"] != "ok":
                raise ValueError(f"{predictions}: {method}: {result['reason']}")
            methods[method] = dict(factors, method=method, **{k: result[k] for k in (*METRICS, "n_truth_edges", "n_candidates")})
            if plan is not None:
                plan.score(method, result, positive, cache, osaka._sparse_average_precision)
    if not methods:
        raise ValueError(f"Empty predictions: {predictions}")
    return methods


def summarize(out, metrics, baselines, seed):
    grouped = defaultdict(list)
    for row in metrics:
        grouped[row["graph_id"], row["method"]].append(row)
    per_graph = [dict(graph_id=g, method=m, n_views=len(views),
                      **{metric: fmean(v[metric] for v in views) for metric in METRICS})
                 for (g, m), views in sorted(grouped.items())]
    table(out / "by_topology.tsv", per_graph, ("graph_id", "method", "n_views", *METRICS))
    methods, graphs = sorted({r["method"] for r in per_graph}), sorted({r["graph_id"] for r in per_graph})
    by_key = {(r["method"], r["graph_id"]): r for r in per_graph}
    for baseline in baselines:
        if baseline not in methods:
            raise ValueError(f"Baseline absent from predictions: {baseline}")
    if baselines:
        rng = np.random.Generator(np.random.PCG64(seed))
        draws = rng.integers(len(graphs), size=(5000, len(graphs)))
    summary = []
    for m in methods:
        row = dict(method=m, n_graphs=len(graphs),
                   **{metric: fmean(by_key[m, g][metric] for g in graphs) for metric in METRICS})
        if not baselines:
            summary.append(row)
        for baseline in baselines:
            differences = np.array([by_key[m, g]["aupr"] - by_key[baseline, g]["aupr"] for g in graphs])
            low, high = np.quantile(differences[draws].mean(axis=1), (0.025, 0.975))
            summary.append(dict(row, baseline=baseline, delta_aupr=fmean(differences),
                                ci_low=low if len(graphs) > 1 else "", ci_high=high if len(graphs) > 1 else ""))
    table(out / "summary.tsv", summary, summary[0].keys())


def run(index, out, baselines, seed):
    index, out = index.resolve(), out.resolve()
    if out.is_relative_to(Path(__file__).resolve().parents[1]):
        raise ValueError("Use an evaluation directory outside the checkout")
    if out.exists():
        raise ValueError(f"Choose a new output directory: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=out.parent) as staging:
        work = Path(staging) / "evaluation"
        work.mkdir()
        metrics, winners, seen = [], [], set()
        robustness, diagnostics = [], {}
        method_ids = panel_signature = None
        phase = None
        with index.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream, delimiter="\t")
            for row in reader:
                row_phase = row.get("phase")
                if row_phase not in {"construction", "robustness", "final"} or (phase and phase != row_phase):
                    raise ValueError(f"{index}: expected one registered scoring phase")
                phase = row_phase
                factors = {k: row[k] for k in FACTORS}
                view_id = row["id"]
                if view_id in seen:
                    raise ValueError(f"Duplicate dataset: {view_id}")
                seen.add(view_id)
                truth, cache = view_truth(view_id, index.parent / row["dataset_manifest"],
                                          index.parent / row["ground_truth_manifest"])
                plan = None
                if phase == "robustness":
                    if not row.get("panels"):
                        raise ValueError(f"{view_id}: missing panel metadata path")
                    plan = robustness_report.PanelPlan(index.parent / row["panels"], truth.candidate_space.targets)
                    if panel_signature is None:
                        panel_signature = plan.signature()
                    elif panel_signature != plan.signature():
                        raise ValueError(f"{view_id}: panel scenario or repetition coverage differs across the phase")
                    if not row.get("source_diagnostics"):
                        raise ValueError(f"{view_id}: the robustness index needs source diagnostics for real-tool reintroduction")
                    source_aupr = robustness_report.read_source_aupr(index.parent / row["source_diagnostics"], plan.omissions)
                    diagnostics[view_id] = (row["graph_id"], source_aupr)
                methods = evaluate_view(factors, index.parent / row["predictions"], truth, cache, plan)
                if method_ids is None:
                    method_ids = set(methods)
                    if phase == "construction":
                        recipe, construction_provenance = construction_report.registered_recipe(
                            HERE / "01-construction/ablations.json",
                            index.parent / "protocol.json", method_ids, INDIVIDUALS)
                if set(methods) != method_ids:
                    raise ValueError(f"{view_id}: method coverage differs across the phase")
                if plan is not None:
                    robustness.extend(plan.observations(factors, methods))
                if INDIVIDUALS <= methods.keys():
                    best = max(methods[m]["aupr"] for m in INDIVIDUALS)
                    tied = sorted(m for m in INDIVIDUALS if methods[m]["aupr"] == best)
                    winner = dict(factors, best_individuals=",".join(tied), best_aupr=best)
                    if "naniwa" in methods:
                        score = methods["naniwa"]["aupr"]
                        winner.update(naniwa_aupr=score, naniwa_minus_best_aupr=score - best)
                    winners.append(winner)
                metrics.extend(methods.values())
                print(f"{view_id}: {len(methods)} methods, {cache.n_candidates} eligible edges", flush=True)
        if not metrics:
            raise ValueError(f"Empty phase index: {index}")
        table(work / "metrics.tsv", metrics, metrics[0].keys())
        if winners:
            table(work / "best_individual_by_dataset.tsv", winners, winners[0].keys())
        summarize(work, metrics, baselines, seed)
        if phase == "construction":
            contrasts = construction_report.construction_contrasts(metrics, recipe["contrasts"], seed)
            table(work / "construction_contrasts.tsv", contrasts, contrasts[0].keys())
        if robustness:
            robustness_report.write_report(work, robustness, FACTORS, diagnostics, seed, table)
        receipt = {"phase": phase, "andrea_version": version("ANDREA"), "evaluator_sha256": hashlib.sha256(Path(osaka.__file__).read_bytes()).hexdigest(),
                   "predictions_index_sha256": hashlib.sha256(index.read_bytes()).hexdigest(),
                   "n_datasets": len(seen), "n_methods": len(method_ids), "primary_metric": "directed.aupr",
                   "paired_baselines": baselines, "bootstrap_draws": 5000 if baselines else 0, "bootstrap_seed": seed if baselines else None}
        if robustness:
            receipt.update(robustness_report=True, robustness_policies=robustness_report.POLICIES,
                           robustness_bootstrap_draws=5000, robustness_bootstrap_seed=seed,
                           evaluate_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                           robustness_report_sha256=hashlib.sha256(Path(robustness_report.__file__).read_bytes()).hexdigest(),
                           strongest_real_reintroduction=len({graph for graph, _ in diagnostics.values()}) >= 2)
        elif phase == "construction":
            receipt.update(construction_provenance, evaluate_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                           construction_report_sha256=hashlib.sha256(Path(construction_report.__file__).read_bytes()).hexdigest(),
                           construction_bootstrap_draws=5000, construction_bootstrap_seed=seed)
        (work / "evaluation.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        os.rename(work, out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True, help="Frozen predictions.tsv written by study.py")
    parser.add_argument("--out", type=Path, required=True, help="New destination directory outside the checkout")
    parser.add_argument("--baseline", action="append", default=[], help="Method ID for paired AUPR contrasts across topologies (repeatable)")
    parser.add_argument("--seed", type=int, default=20261126, help="Paired topology bootstrap seed (default: 20261126)")
    args = parser.parse_args()
    run(args.predictions, args.out, list(dict.fromkeys(args.baseline)), args.seed)
