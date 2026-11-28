"""Generate paired time series with ANDREA and export the declared temporal views."""

import argparse
import csv
import fcntl
import hashlib
import itertools
import json
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from andrea.core.shared.dataset_identity import fingerprint_dataset_content


if __package__:
    from .benchmark_identity import inventory_hashes
else:
    from benchmark_identity import inventory_hashes


INDEX_FIELDS = ("id", "graph_id", "simulator", "design", "noise", "n_series", "n_timepoints",
                "dataset_manifest", "ground_truth_manifest", "networks")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, data):
    tmp = path.with_suffix(".pending")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def resolve(template, values):
    if isinstance(template, dict):
        return {key: resolve(value, values) for key, value in template.items()}
    if isinstance(template, str) and template.startswith("$"):
        return values[template[1:]]
    return template


def interventions(graph, seed, design):
    regulators = sorted({s for s, _, _ in graph["edges"]})
    count = design["targets_per_group"]
    if len(regulators) <= count:
        raise ValueError(f"{graph['id']}: too few regulators for distinct intervention groups")
    rng = np.random.default_rng(seed)
    pool = rng.permutation(regulators).tolist()
    groups = []
    for _ in range(design["groups"]):
        for _ in range(100):
            group = sorted(map(str, rng.choice(pool, size=count, replace=False)))
            if group not in groups:
                groups.append(group)
                break
        else:
            raise ValueError(f"{graph['id']}: could not draw distinct intervention groups")
    return groups


def view_plan(config, profiles, specs):
    """Resolve the configured views into native simulations before running ANDREA."""
    requested = config.get("views")
    remaining = None if requested is None else set(requested)
    if remaining is not None and len(remaining) != len(requested):
        raise ValueError("Duplicate view IDs in the simulation recipe")
    plan = {spec["id"]: [] for spec in specs}
    if set(config["seeds"]) != plan.keys():
        raise ValueError("Simulation seeds must cover exactly the configured graphs")
    design = config["design"]
    schedules = list(itertools.product(design["series"], design["timepoints"]))
    for graph_id, cases in plan.items():
        seeds = config["seeds"][graph_id]
        if seeds.keys() != {"interventions", *profiles["simulators"]}:
            raise ValueError(f"{graph_id}: provide a seed for interventions and for each simulator")
        for sim, profile in profiles["simulators"].items():
            for noise in profile["noise"]:
                name = f"{graph_id}__{sim}__{design['id']}__{noise}"
                views = []
                for ns, nt in schedules:
                    view_id = f"{name}__s{ns}_t{nt}"
                    if remaining is not None:
                        if view_id not in remaining:
                            continue
                        remaining.remove(view_id)
                    views.append((ns, nt))
                if views:
                    cases.append((sim, design["id"], noise, name, views))
    if remaining:
        raise ValueError(f"Unknown view IDs in the simulation recipe: {sorted(remaining)}")
    cases = [case for graph_cases in plan.values() for case in graph_cases]
    if {"native": len(cases), "views": sum(len(case[-1]) for case in cases)} != config["expected"]:
        raise ValueError("Simulation recipe does not match its expected native and view counts")
    return plan


def requests(config, profiles, specs, out, plan):
    for spec in specs:
        graph_id = spec["id"]
        path = out / "topologies" / graph_id / "networks" / graph_id / "topology.json"
        graph = read(path)
        if graph["id"] != graph_id or len(graph["genes"]) != spec["size"]:
            raise ValueError(f"Topology identity mismatch: {path}")
        seeds = config["seeds"][graph_id]
        targets = interventions(graph, seeds["interventions"], profiles["interventions"])
        steps = [i * graph["stride"] for i in range(21)]
        values = dict(graph_id=graph_id, size=spec["size"], targets=targets,
                      vectors=[[-0.5 if gene in group else 0 for gene in graph["genes"]] for group in targets],
                      steps=steps, last_step=steps[-1])
        for sim, design, noise, name, views in plan[graph_id]:
            profile = profiles["simulators"][sim]
            source = profile["input"]
            input_file = (path.parent / graph[source["topology_file"]]).resolve(strict=True)
            run = dict(run_id=name, simulator_id=sim, replicates=1, base_seed=seeds[sim],
                       params=resolve(profile["params"], values | profile["noise"][noise]))
            if "native_outputs" in profile:
                run["native_outputs"] = profile["native_outputs"]
            scenario = dict(profiles["scenario"], id=name, inputs={
                source["name"]: {"path": str(input_file), "format": source["format"]}})
            yield dict(id=name, graph=graph, simulator=sim, design=design, noise=noise,
                       scenario=scenario, runs={"schema_version": "1.0", "runs": [run]},
                       input_sha256=sha(input_file), profile=profile, views=views)


def andrea(cmd, args, log):
    with log.open("w") as stream:
        result = subprocess.run([*cmd, *map(str, args)], stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"ANDREA exited with code {result.returncode}; see {log}")


def native_data(case, root, cmd):
    root.mkdir(parents=True, exist_ok=True)
    receipt = root / "complete.json"
    signature = dict(runs=case["runs"], scenario={k: v for k, v in case["scenario"].items() if k != "inputs"},
                     input_sha256=case["input_sha256"], image=case["profile"]["image_id"])
    digest = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
    if receipt.exists():
        record = read(receipt)
        if record["recipe_sha256"] != digest or any(sha(root / name) != value for name, value in record["files"].items()):
            raise ValueError(f"Native recipe or output changed: {root}")
        return root / record["dataset"], root / record["truth"]

    config = root / "config"
    config.mkdir(exist_ok=True)
    write(config / "scenario.json", case["scenario"])
    write(config / "runs.json", case["runs"])
    andrea(cmd, ["generate-data", "plan", "--scenario", config / "scenario.json",
                 "--simulator-runs", config / "runs.json", "--out", config / "plan.json",
                 "--max-parallel-tasks", 1, "--max-cores", 2, "--max-ram-gb", 4], root / "plan.log")
    generated = root / "generated"
    if not generated.exists():
        with TemporaryDirectory(prefix="attempt-", dir=root) as attempt:
            temp = Path(attempt) / "generated"
            andrea(cmd, ["generate-data", "run", "--plan", config / "plan.json",
                         "--output-dir", temp], root / "generate.log")
            write(temp / "recipe.json", digest)
            temp.rename(generated)
    if read(generated / "recipe.json") != digest:
        raise ValueError(f"Generation belongs to a different recipe: {root}")
    marker, = generated.glob("*/benchmark-manifest.json")
    result, = read(marker)["datasets"]
    ds, truth = (marker.parent / result[k] for k in ("dataset_manifest", "ground_truth_manifest"))
    dataset = read(ds)
    payload = [ds, truth, ds.parent / dataset["dataset"]["expression_matrix"]]
    payload.extend(ds.parent / value for value in dataset["extras"].values())
    payload.extend(truth.parent / value for value in read(truth)["outputs"].values())
    write(receipt, dict(recipe_sha256=digest, dataset=str(ds.relative_to(root)),
                        truth=str(truth.relative_to(root)), files={str(p.relative_to(root)): sha(p) for p in payload}))
    return ds, truth


def table(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.reader(stream, delimiter="\t"))


def save_table(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        csv.writer(stream, delimiter="\t", lineterminator="\n").writerows(rows)


def validate_truth(path, truth, graph):
    space = truth["candidate_space"]
    sources = (path.parent / space["sources"]).read_text().splitlines()
    targets = (path.parent / space["targets"]).read_text().splitlines()
    expected = {(s, t) for s, t, _ in graph["edges"] if s != t}
    with (path.parent / truth["outputs"]["networks"]).open(newline="") as stream:
        actual = [(r["source"], r["target"]) for r in csv.DictReader(stream)
                  if r["source"] != r["target"] and float(r["score"]) > 0]
    if (space["allow_self_edges"] or sorted(targets) != sorted(graph["genes"])
            or sorted(sources) != sorted({s for s, _ in expected})
            or len(actual) != len(expected) or set(actual) != expected):
        raise ValueError(f"Ground truth or candidate space differs from the input topology: {path}")


def export(case, native, truth_file, out):
    data = read(native)
    matrix = table(native.parent / data["dataset"]["expression_matrix"])
    extras = {k: native.parent / v for k, v in data["extras"].items()}
    descriptors = table(extras["column_descriptors"])
    columns, data_rows = matrix[0][1:], matrix[1:]
    labels, records = descriptors[0], descriptors[1:]
    col, series, time, obs = map(labels.index, ("column", "series_id", "timepoint", "observation_type"))
    genes = [r[0] for r in data_rows]
    regulators = extras["tf_list"].read_text().splitlines()
    expected_tfs = {s for s, _, _ in case["graph"]["edges"]}
    if sorted(genes) != sorted(case["graph"]["genes"]) or sorted(regulators) != sorted(expected_tfs):
        raise ValueError(f"Gene/TF universe mismatch: {native}")
    if any(len(r) != len(columns) + 1 for r in data_rows) or len(set(columns)) != len(columns):
        raise ValueError(f"Invalid native matrix dimensions: {native}")
    values = np.asarray([r[1:] for r in data_rows], dtype=float)
    if not np.isfinite(values).all() or not np.any(np.ptp(values, axis=1) > 1e-12):
        raise ValueError(f"Nonfinite or wholly uninformative native expression: {native}")
    if [r[col] for r in records] != columns or any(r[obs] != "time_series" for r in records):
        raise ValueError(f"Sample descriptors differ from native expression: {native}")
    series_ids = sorted({r[series] for r in records})
    groups = [sorted([i for i, row in enumerate(records) if row[series] == s],
                     key=lambda i: float(records[i][time])) for s in series_ids]
    points = case["profile"]["native_points"]
    if len(groups) != 6 or any(len(g) != points for g in groups):
        raise ValueError(f"Unexpected number of series or timepoints: {native}")
    for group in groups:
        times = np.array([float(records[i][time]) for i in group])
        delta = np.diff(times)
        if not np.isfinite(times).all() or np.any(delta <= 0) or abs(delta - delta.mean()).max() > 0.25 * delta.mean():
            raise ValueError(f"Invalid sampling grid: {native}")
        if case["simulator"] == "genesnake" and not np.array_equal(times, np.arange(102)):
            raise ValueError(f"Expected the GeneSnake grid 0..101: {native}")
    truth = read(truth_file)
    if truth["dataset_id"] != data["dataset"]["spec"]["id"] or truth["simulator_id"] != case["simulator"]:
        raise ValueError(f"Native dataset and truth identities disagree: {truth_file}")
    validate_truth(truth_file, truth, case["graph"])
    if truth["dataset_fingerprint"] != fingerprint_dataset_content(
            expression_path=native.parent / data["dataset"]["expression_matrix"], extras=extras):
        raise ValueError(f"Unmatched native truth: {truth_file}")
    rows = []
    for ns, nt in case["views"]:
        stride = case["profile"]["view_stride"] * (20 // (nt - 1))
        idx = [i for g in groups[:ns] for i in g[:(nt - 1) * stride + 1:stride]]
        if len(idx) != ns * nt:
            raise ValueError("View dimensions disagree with native sampling grid")
        name = f"{case['id']}__s{ns}_t{nt}"
        folder = out / "views" / name
        folder.mkdir(parents=True, exist_ok=True)
        inp = folder / "input"
        if not (folder / "complete.json").exists():
            with TemporaryDirectory(dir=folder) as temp:
                tmp = Path(temp) / "input"
                (tmp / "extras").mkdir(parents=True)
                save_table(tmp / "expression.tsv", [[matrix[0][0], *(columns[i] for i in idx)],
                           *([row[0], *(row[i + 1] for i in idx)] for row in data_rows)])
                save_table(tmp / "extras/column_descriptors.tsv", [labels, *(records[i] for i in idx)])
                shutil.copyfile(extras["tf_list"], tmp / "extras/tf_list.txt")
                derived = json.loads(json.dumps(data))
                derived["id"] = name + "_manifest"
                spec = derived["dataset"]["spec"]
                spec["id"] = spec["name"] = name
                spec["expression"]["columns"] = len(idx)
                derived["dataset"]["expression_matrix"] = "expression.tsv"
                # Keep the simulator-specific order when serializing extras.
                keys = ("column_descriptors", "tf_list")
                if case["simulator"] == "genesnake":
                    keys = ("tf_list", "column_descriptors")
                derived["extras"] = {k: f"extras/{extras[k].name}" for k in keys}
                write(tmp / "dataset-manifest.json", derived)
                bound = dict(truth, dataset_id=name, dataset_fingerprint=fingerprint_dataset_content(
                    expression_path=tmp / "expression.tsv", extras={k: tmp / v for k, v in derived["extras"].items()}))
                candidate_files = {truth["candidate_space"][k] for k in ("sources", "targets")}
                for rel in set(truth["outputs"].values()) | candidate_files:
                    dest = tmp / rel
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    if dest.exists():
                        if sha(dest) != sha(truth_file.parent / rel):
                            raise ValueError(f"View and native truth candidate space differ at {rel}")
                    else:
                        shutil.copyfile(truth_file.parent / rel, dest)
                write(tmp / "ground-truth-manifest.json", bound)
                hashes = {str(p.relative_to(tmp)): sha(p) for p in tmp.rglob("*") if p.is_file()}
                if inp.exists():
                    if any(sha(inp / rel) != value for rel, value in hashes.items()):
                        raise ValueError(f"Conflicting incomplete view: {inp}")
                else:
                    tmp.rename(inp)
                write(folder / "complete.json", dict(native_id=case["id"], series=series_ids[:ns],
                      source_columns=[columns[i] for i in idx], native_truth_sha256=sha(truth_file), files=hashes))
        else:
            verified = read(folder / "complete.json")
            if (verified["native_truth_sha256"] != sha(truth_file) or verified["source_columns"] != [columns[i] for i in idx]
                    or any(sha(inp / rel) != value for rel, value in verified["files"].items())):
                raise ValueError(f"View verification failed: {inp}")
        rows.append(dict(id=name, graph_id=case["graph"]["id"], simulator=case["simulator"], noise=case["noise"],
                         design=case["design"], n_series=ns, n_timepoints=nt,
                         dataset_manifest=str((inp / "dataset-manifest.json").relative_to(out)),
                         ground_truth_manifest=str((inp / "ground-truth-manifest.json").relative_to(out)), networks=""))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--andrea", help="ANDREA-Osaka executable (default: this Python's andrea module).")
    parser.add_argument("--jobs", type=int, default=1, help="Concurrent native simulations.")
    parser.add_argument("--graph", action="append",
                        help="Generate only these graph IDs; repeat as needed. The full run writes datasets.tsv.")
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    cfg = read(args.config)
    profiles = read(args.config.parent / cfg["profiles"])
    graphs = read(args.config.parent / cfg["topologies"])["graphs"]
    plan = view_plan(cfg, profiles, graphs)
    for p in profiles["simulators"].values():
        actual = subprocess.check_output(
            ["docker", "image", "inspect", "--format", "{{.Id}}", p["image"]], text=True).strip()
        if actual != p["image_id"]:
            raise ValueError(f"Unexpected image for {p['image']}: {actual}")
    out = args.out.resolve()
    if out.is_relative_to(Path(__file__).resolve().parents[1]):
        parser.error("--out must be outside the project checkout")
    if args.graph and not set(args.graph).issubset({g["id"] for g in graphs}):
        parser.error("Unknown --graph ID")
    selected = [g for g in graphs if not args.graph or g["id"] in args.graph]
    out.mkdir(parents=True, exist_ok=True)
    cmd = [args.andrea] if args.andrea else [sys.executable, "-m", "andrea"]

    def generate(case):
        native, truth = native_data(case, out / "native" / case["id"], cmd)
        rows = export(case, native, truth, out)
        print(f"{case['id']}: {len(rows)} validated views", flush=True)
        return rows

    with (out / ".simulation.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            results = list(pool.map(generate, requests(cfg, profiles, selected, out, plan)))
        rows = sorted(itertools.chain.from_iterable(results), key=lambda r: r["id"])
        if not args.graph:
            totals = {"native": len(results), "views": len(rows)}
            if totals != cfg["expected"] or len({row["id"] for row in rows}) != len(rows):
                raise ValueError("Incomplete dataset inventory")
            if "data_lock" in cfg:
                observed = inventory_hashes(out, rows)
                lock_path = args.config.parent / cfg["data_lock"]
                if observed != read(lock_path):
                    raise ValueError(f"Simulated data differ from {lock_path}; observed {observed}")
            buf = StringIO()
            writer = csv.DictWriter(buf, fieldnames=INDEX_FIELDS, delimiter="\t", lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
            tmp = out / "datasets.pending"
            tmp.write_text(buf.getvalue(), encoding="utf-8")
            tmp.replace(out / "datasets.tsv")
        print(f"Complete: {len(results)} native simulations, {len(rows)} views.", flush=True)


if __name__ == "__main__":
    main()
