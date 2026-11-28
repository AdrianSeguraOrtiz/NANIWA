"""Run the registered experiments on ANDREA datasets from an external working directory."""
import argparse
import csv
import gzip
import hashlib
import io
import itertools
import json
import os
import platform
from importlib.metadata import version
from importlib.resources import files
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from scipy.stats import rankdata

from naniwa.consensus import ConsensusOptions, STAGES, TemporalConsensus
from naniwa.inference import run_inference
from naniwa.inputs import load_dataset, load_networks

if __package__:
    from .benchmark_identity import inventory_hashes
    from .robustness_diagnostics import SourceDiagnostics
    from .robustness_panels import panels
else:
    from benchmark_identity import inventory_hashes
    from robustness_diagnostics import SourceDiagnostics
    from robustness_panels import panels

HERE = Path(__file__).resolve().parent
CONSTRUCTION = HERE / "01-construction/ablations.json"
ROBUSTNESS = HERE / "02-robustness/panels.json"
FINAL = HERE / "03-final-comparison/comparison.json"
PHASE_RECIPES = {"construction": CONSTRUCTION, "robustness": ROBUSTNESS, "final": FINAL}
INDEX_PATHS = ("dataset_manifest", "ground_truth_manifest", "networks")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def table(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def write_table(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, rows[0], delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def safe_id(name):
    return bool(name) and name not in (".", "..") and Path(name).name == name


def dataset_rows(index):
    rows = table(index)
    if not rows or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("The dataset index must contain unique dataset IDs")
    for row in rows:
        if not safe_id(row["id"]):
            raise ValueError(f"Invalid dataset ID: {row['id']!r}")
        for field in INDEX_PATHS:
            if row.get(field):
                row[field] = str((index.parent / row[field]).resolve())
    return rows


def variants(phase="construction", recipe=None):
    """Resolve the consensus configurations scored in a registered phase."""
    if recipe is None:
        recipe = read_json(PHASE_RECIPES[phase])
    if phase == "construction":
        names = (*recipe["path"], *recipe["controls"])
    elif phase == "robustness":
        names = recipe["every_panel"]
    else:
        baselines = [name for name in recipe["baselines"] if name in STAGES]
        names = ("naniwa", *recipe["controls"], *baselines)
    return {name: ConsensusOptions(stage=name) for name in names}


def ranked_scores(values):
    ranks = rankdata(values, method="average")
    return ranks - ranks.mean()


def correlation(left, right):
    norm = np.linalg.norm(left) * np.linalg.norm(right)
    return float(left @ right / norm) if norm else ""


def prediction(method, panel, policy, tools, stability, values, **details):
    return dict(method=method, panel=panel, policy=policy, tools=",".join(tools),
                rank_correlation_to_clean=stability, **details), values


def truth_assets(row):
    path = Path(row["ground_truth_manifest"])
    spec = read_json(path)
    return {"truth_manifest": path, "truth_network": path.parent / spec["outputs"]["networks"],
            "truth_genes": path.parent / spec["outputs"]["gene_universe"],
            **{f"truth_{key}": path.parent / spec["candidate_space"][key] for key in ("sources", "targets")}}


def robustness_truth(row, data):
    """Read the directed reference for benchmark controls on the inference axes."""
    paths = truth_assets(row)
    manifest = read_json(paths["truth_manifest"])
    view_path = Path(row["dataset_manifest"])
    view = read_json(view_path)
    if manifest["dataset_id"] != row["id"] or view["dataset"]["spec"]["id"] != row["id"]:
        raise ValueError(f"{row['id']}: robustness inputs identify different datasets")
    # ANDREA's content identity is portable across inference and simulation runtimes.
    components = {"expression": digest(view_path.parent / view["dataset"]["expression_matrix"])}
    components.update({f"extra:{key}": digest(view_path.parent / path)
                       for key, path in view["extras"].items() if path is not None})
    fingerprint = hashlib.sha256(json.dumps(components, sort_keys=True, separators=(",", ":")).encode("ascii"))
    if manifest["dataset_fingerprint"] != {"algorithm": "sha256", "value": fingerprint.hexdigest()}:
        raise ValueError(f"{row['id']}: robustness ground truth fingerprint differs from the dataset")
    for field, expected in (("sources", data.regulators), ("targets", data.genes), ("genes", data.genes)):
        labels = paths[f"truth_{field}"].read_text(encoding="utf-8-sig").splitlines()
        if sorted(labels) != sorted(expected):
            raise ValueError(f"{row['id']}: ground truth {field} differ from the dataset")
    if manifest["candidate_space"]["allow_self_edges"] is not False:
        raise ValueError(f"{row['id']}: the registered candidate space excludes self edges")
    targets = {gene: i for i, gene in enumerate(data.genes)}
    sources = {tf: i for i, tf in enumerate(data.regulators)}
    truth = np.zeros(data.mask.shape, dtype=bool)
    seen = set()
    with paths["truth_network"].open(encoding="utf-8-sig", newline="") as stream:
        for edge in csv.DictReader(stream):
            if edge["context"] != "global":
                continue
            t, r = targets.get(edge["target"]), sources.get(edge["source"])
            value = float(edge["score"])
            if t is None or r is None or not data.mask[t, r] or (t, r) in seen or not np.isfinite(value) or value < 0:
                raise ValueError(f"{row['id']}: invalid directed ground-truth edge {edge}")
            seen.add((t, r))
            truth[t, r] = value > 0
    if not truth.any() or truth.sum() == data.mask.sum():
        raise ValueError(f"{row['id']}: the robustness benchmark requires true and false candidate interactions")
    return truth


def robustness_predictions(row, data, tools, raw, directory, truth):
    spec = read_json(ROBUSTNESS)
    policies = variants("robustness", spec)
    score = TemporalConsensus(data).score
    diagnostics = SourceDiagnostics(tools, raw, data.mask, truth)
    clean_ranks, anchors = {}, {}
    source_rows, weight_rows = [], []
    for panel in panels(tools, raw, data.mask, row["id"], spec, truth):
        source_rows.extend(diagnostics.rows(panel))
        specialist = panel.specialist_targets
        target_names = [] if specialist is None else list(np.asarray(data.genes)[specialist])
        anchor = panel.founder or panel.reference
        for policy, options in policies.items():
            weights = np.empty(panel.sources.shape[:2]) if specialist is not None else None
            values = score(panel.sources, options, source_vote_weights=weights)[data.mask]
            if panel.id == "clean" or panel.scenario == "null-1":
                anchors[panel.id, policy] = values
            ranks = ranked_scores(values)
            if panel.id == "clean":
                clean_ranks[policy] = ranks
            max_change = float(np.max(np.abs(values - anchors[anchor, policy])))
            if panel.exact_copy and policy == "naniwa" and max_change > 1e-12:
                raise ValueError(f"{row['id']}/{panel.id}/{policy}: exact-copy invariance failed ({max_change})")
            if weights is not None:
                index = panel.names.index(panel.focus_source)
                for target, gene in enumerate(data.genes):
                    weight_rows.append(dict(panel=panel.id, scenario=panel.scenario, repeat=panel.repeat,
                                            policy=policy, source=panel.focus_source, target=gene,
                                            scope="specialist" if specialist[target] else "other",
                                            source_vote_weight=weights[index, target]))
            yield prediction(f"{policy}__{panel.id}", panel.id, policy, panel.names,
                             correlation(ranks, clean_ranks[policy]), values, scenario=panel.scenario,
                             repeat=panel.repeat, reference_panel=panel.reference, founder_panel=panel.founder,
                             matched_panel=panel.matched, specialist_targets=json.dumps(target_names),
                             score_anchor=anchor, max_abs_change_from_anchor=max_change)
    write_table(directory / "source_diagnostics.tsv", source_rows)
    write_table(directory / "specialist_weights.tsv", weight_rows)


def predictions(args, data, tools, raw, external, protocol):
    score = TemporalConsensus(data).score
    mask = data.mask
    for tool, matrix in zip(tools, raw, strict=True):
        yield prediction(tool, "clean", "individual", (tool,), "", matrix[mask])
    if args.phase == "final":
        configs = variants("final", protocol)
        for name in protocol["baselines"]:
            if name == "mean_rank":
                ranks = rankdata(raw[:, mask], axis=1, method="average")
                values = (ranks.mean(axis=0) - 0.5) / mask.sum()
            elif name == "raw_mean":
                values = raw[:, mask].mean(axis=0)
            else:
                continue
            yield prediction(name, "clean", name, tools, "", values)
    else:
        configs = variants()
    for name, opts in configs.items():
        yield prediction(name, "clean", name, tools, "", score(raw, opts)[mask])
    for method, path in external.items():
        _, scores = load_networks(data, individual={method: path}, require_variation=False)
        yield prediction(method, "clean", "external", (), "", scores[0, mask])


def score_dataset(args, row, directory, external, protocol):
    data = load_dataset(row["dataset_manifest"])
    tools, raw = load_networks(data, merged=row["networks"])
    if set(tools) != set(protocol["individual_tools"]):
        raise ValueError(f"{row['id']} needs the seven registered source tools; found {tools}")
    t, r = np.nonzero(data.mask)
    regulators, targets = np.asarray(data.regulators)[r], np.asarray(data.genes)[t]
    if args.phase == "robustness":
        truth = robustness_truth(row, data)
        rankings = robustness_predictions(row, data, tools, raw, directory, truth)
    else:
        rankings = predictions(args, data, tools, raw, external, protocol)
    methods = []
    with gzip.GzipFile(directory / "predictions.csv.gz", "wb", compresslevel=3, mtime=0) as zipped, \
            io.TextIOWrapper(zipped, encoding="utf-8", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["source", "target", "score", "tool_id"])
        for method, values in rankings:
            name = method["method"]
            if not np.isfinite(values).all():
                raise ValueError(f"Nonfinite predictions for {row['id']}/{name}")
            writer.writerows(zip(regulators, targets, map(float, values), itertools.repeat(name)))
            methods.append(method)
    write_table(directory / "methods.tsv", methods)


def external_files(index, protocol, ids, reserved):
    if bool(index) != bool(protocol):
        raise ValueError("Supply --external and --external-protocol together")
    if index is None:
        return {}
    registered = read_json(protocol)
    if not isinstance(registered, dict) or not registered:
        raise ValueError("The external protocol must name at least one comparator")
    for method, spec in registered.items():
        if not safe_id(method) or not method.isprintable() or method in reserved:
            raise ValueError(f"Invalid or reserved external method ID: {method!r}")
        if (not isinstance(spec, dict) or not isinstance(spec.get("version"), str) or not spec["version"].strip()
                or not isinstance(spec.get("parameters"), dict) or not isinstance(spec.get("seeds"), list)
                or not all(type(seed) is int for seed in spec["seeds"])
                or not isinstance(spec.get("selection_rule"), str) or not spec["selection_rule"].strip()):
            raise ValueError(f"{method}: specify version, parameters, seeds and selection_rule in the external protocol")
    files = {key: {} for key in ids}
    for row in table(index):
        key, method = row["id"], row["method"]
        if key not in files or method not in registered or method in files[key]:
            raise ValueError(f"Unexpected or repeated external prediction: {key}/{method}")
        path = (index.parent / row["network"]).resolve(strict=True)
        files[key][method] = path
    if any(methods.keys() != registered.keys() for methods in files.values()):
        raise ValueError("External methods must provide one network for every indexed dataset")
    return files


def software():
    return {"python": platform.python_version(), **{name: version(name) for name in ("naniwa", "numpy", "scipy")}}


def frozen():
    name = read_json(CONSTRUCTION)["primary"]
    return {"variant": name, "options": variants()[name].parameters(), "naniwa": version("naniwa"),
            "construction_sha256": digest(CONSTRUCTION)}


def save_once(path, data):
    text = json.dumps(data, sort_keys=True, indent=2, allow_nan=False) + "\n"
    if path.is_file():
        if path.read_text() != text:
            raise ValueError(f"Configuration or inputs changed; choose a new output path: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f:
        f.write(text)


def input_hashes(row, *, with_truth=False):
    path = Path(row["dataset_manifest"])
    spec = read_json(path)
    assets = {"manifest": path, "expression": path.parent / spec["dataset"]["expression_matrix"],
              "samples": path.parent / spec["extras"]["column_descriptors"], "regulators": path.parent / spec["extras"]["tf_list"]}
    if row.get("networks"):
        assets["networks"] = Path(row["networks"])
    if with_truth:
        assets.update(truth_assets(row))
        for key, value in spec["extras"].items():
            if key not in ("column_descriptors", "tf_list") and value is not None:
                assets[f"extra:{key}"] = path.parent / value
    return {name: digest(file) for name, file in assets.items()}


def run(args):
    import fcntl   # ANDREA/Docker experiment host is Linux.
    args.out = args.out.resolve()
    if args.out.is_relative_to(HERE.parent):
        raise ValueError("Use a result path outside the repository")
    if args.phase == "freeze":
        save_once(args.out, frozen())
        return
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        process(args)


def verify_benchmark(rows, index, phase):
    folder = FINAL.parent if phase == "final" else CONSTRUCTION.parent
    design = read_json(folder / "simulation.json")
    lock = read_json(folder / design["data_lock"])
    if inventory_hashes(index.parent, rows) != lock:
        raise ValueError(f"Inputs do not match the registered benchmark in {folder.name}/data.lock.json")
    for row in rows:
        identity = (f"{row['graph_id']}__{row['simulator']}__{row['design']}__{row['noise']}"
                    f"__s{row['n_series']}_t{row['n_timepoints']}")
        if identity != row["id"]:
            raise ValueError(f"Experimental factors do not identify view {row['id']}")
    return lock


def process(args):
    rows = dataset_rows(args.index)
    protocol = read_json(FINAL)
    benchmark = verify_benchmark(rows, args.index, args.phase) if args.phase != "infer" else None
    if args.config and read_json(args.config) != frozen():
        raise ValueError("The configuration, method version or construction recipe differs from the freeze")
    reserved = {*protocol["individual_tools"], *protocol["baselines"], *STAGES}
    external = external_files(args.external, args.external_protocol, [r["id"] for r in rows], reserved)
    recipes = {file.name: digest(file) for file in [CONSTRUCTION, ROBUSTNESS, FINAL]}
    if args.phase == "robustness":
        for filename in ("robustness_panels.py", "robustness_diagnostics.py"):
            recipes[filename] = digest(HERE / filename)
    settings = {"phase": args.phase, "versions": software(), "runner_sha256": digest(__file__),
        "index_sha256": digest(args.index), "benchmark": benchmark, "recipes": recipes, "method": read_json(args.config) if args.config else None,
        "external_index_sha256": digest(args.external) if args.external else None,
        "external_protocol": read_json(args.external_protocol) if args.external_protocol else None,
        "inference": {"andrea": args.andrea, "max_cores": args.max_cores, "max_ram_gb": args.max_ram_gb,
                      "catalog_sha256": hashlib.sha256(files("naniwa").joinpath("catalog/catalog.json").read_bytes()).hexdigest()}
                     if args.phase == "infer" else None}
    if args.phase in PHASE_RECIPES:
        settings[f"{args.phase}_stages"] = {name: options.parameters() for name, options in variants(args.phase).items()}
    save_once(args.out / "protocol.json", settings)
    completed = []
    for row in rows:
        directory = args.out / row["id"]
        receipt = directory / "receipt.json"
        extra = external.get(row["id"], {})
        inputs = input_hashes(row, with_truth=args.phase == "robustness")
        inputs.update({f"external:{method}": digest(path) for method, path in extra.items()})
        if not directory.exists():
            with TemporaryDirectory(dir=args.out, prefix=".partial-") as tmp:
                stage = Path(tmp) / "result"
                stage.mkdir()
                if args.phase == "infer":
                    result = run_inference(row["dataset_manifest"], stage / "andrea", protocol["individual_tools"],
                                           andrea=args.andrea, max_cores=args.max_cores, max_ram_gb=args.max_ram_gb)
                    outputs = {"inference_dataset_manifest": str(result.dataset.relative_to(stage)), "networks": str(result.network.relative_to(stage))}
                else:
                    score_dataset(args, row, stage, extra, protocol)
                    outputs = {"predictions": "predictions.csv.gz"}
                    if args.phase == "robustness":
                        outputs.update(panels="methods.tsv", source_diagnostics="source_diagnostics.tsv",
                                       specialist_weights="specialist_weights.tsv")
                signatures = {key: digest(stage / value) for key, value in outputs.items()}
                if args.phase not in ("infer", "robustness"):
                    signatures["methods"] = digest(stage / "methods.tsv")
                save_once(stage / "receipt.json", {"inputs": inputs, "outputs": outputs, "sha256": signatures})
                stage.rename(directory)
        record = read_json(receipt)
        if record["inputs"] != inputs:
            raise ValueError(f"Source inputs differ from completed result: {receipt}")
        for field, signature in record["sha256"].items():
            path = directory / ("methods.tsv" if field == "methods" else record["outputs"][field])
            if digest(path) != signature:
                raise ValueError(f"Completed output changed: {path}")
        updated = {key: str((directory / value).relative_to(args.out)) for key, value in record["outputs"].items()}
        completed.append((row | updated) if args.phase == "infer" else dict(row | updated, phase=args.phase))
        print(f"{len(completed)}/{len(rows)}  {row['id']}", flush=True)
    with TemporaryDirectory(dir=args.out) as tmp:
        index = Path(tmp) / ("datasets.tsv" if args.phase == "infer" else "predictions.tsv")
        write_table(index, completed)
        os.replace(index, args.out / index.name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="phase", required=True)
    listing = commands.add_parser("variants", help="Print a study phase's consensus configurations and their component parameters.")
    listing.add_argument("--phase", dest="study_phase", choices=PHASE_RECIPES, default="construction",
                         help="Study phase to inspect (default: construction).")
    freeze = commands.add_parser("freeze", help="Record the NANIWA stage, its component settings and the construction recipe.")
    freeze.add_argument("--out", required=True, type=Path)
    for phase in ("infer", "construction", "robustness", "final"):
        p = commands.add_parser(phase)
        p.add_argument("--index", required=True, type=Path)
        p.add_argument("--out", required=True, type=Path)
        p.set_defaults(config=None, external=None, external_protocol=None, andrea=None, max_cores=None, max_ram_gb=None)
        if phase == "infer":
            p.add_argument("--andrea", default="andrea")
            p.add_argument("--max-cores", type=int)
            p.add_argument("--max-ram-gb", type=float)
        elif phase in ("robustness", "final"):
            p.add_argument("--config", required=True, type=Path, help="Method manifest written by freeze.")
        if phase == "final":
            p.add_argument("--external", type=Path, help="Index TSV (id,method,network) of registered external predictions.")
            p.add_argument("--external-protocol", type=Path, help="Frozen JSON keyed by method with version, parameters, seeds, selection_rule.")
    args = parser.parse_args()
    if args.phase == "variants":
        for key, options in variants(args.study_phase).items():
            print(key, json.dumps(options.parameters(), sort_keys=True), sep="\t")
    else:
        try:
            run(args)
        except (ValueError, KeyError, OSError) as error:
            parser.exit(2, f"{parser.prog}: {error}\n")


if __name__ == "__main__":
    main()
