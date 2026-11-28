"""Generate and audit shared, signed graphs with the native ANDREA simulator images."""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np


_SOURCES = {
    "ecoli": "ecoli_transcriptional_network_regulonDB_6_7.tsv",
    "yeast": "yeast_transcriptional_network_Balaji2006.tsv",
}


def _dump(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _one(directory: Path, pattern: str) -> Path:
    candidates = sorted(directory.glob(pattern))
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one {pattern} in {directory}, found {len(candidates)}")
    return candidates[0]


def _graph_summary(edges: list[list[str]], n_genes: int, hub_limit: float) -> dict:
    genes = sorted({gene for source, target, _ in edges for gene in (source, target)})
    if len(genes) != n_genes:
        raise ValueError(f"Expected {n_genes} genes, observed {len(genes)}")
    if any(source == target or sign not in {"+", "-"} for source, target, sign in edges):
        raise ValueError("Edges must be signed, with no self interactions")
    if len({(source, target) for source, target, _ in edges}) != len(edges):
        raise ValueError("Duplicate regulator-target pairs")
    indegree = dict.fromkeys(genes, 0)
    outdegree = dict.fromkeys(genes, 0)
    neighbors = {gene: set() for gene in genes}
    children = {gene: [] for gene in genes}
    for source, target, _ in edges:
        indegree[target] += 1
        outdegree[source] += 1
        neighbors[source].add(target)
        neighbors[target].add(source)
        children[source].append(target)
    seen, frontier = set(), {genes[0]}
    while frontier:
        seen.update(frontier)
        frontier = {other for gene in frontier for other in neighbors[gene]} - seen
    if len(seen) != n_genes:
        raise ValueError(f"Graph is not weakly connected: component has {len(seen)} genes")
    n_regulators = sum(degree > 0 for degree in outdegree.values())
    if n_regulators < 5:
        raise ValueError(f"Expected at least 5 regulators, observed {n_regulators}")
    if len(edges) >= n_regulators * (n_genes - 1):
        raise ValueError("No negative candidate edges remain")
    hub_fraction = max(outdegree.values()) / len(edges)
    if hub_fraction > hub_limit:
        raise ValueError(
            f"Largest regulator accounts for {hub_fraction:.6f} of edges; limit {hub_limit:.6f}"
        )
    pending = indegree.copy()
    order = [gene for gene in genes if not pending[gene]]
    for source in order:
        for target in children[source]:
            pending[target] -= 1
            if not pending[target]:
                order.append(target)
    return {
        "n_genes": n_genes,
        "n_edges": len(edges),
        "n_regulators": n_regulators,
        "edge_density": len(edges) / (n_genes * (n_genes - 1)),
        "max_hub_edge_fraction": hub_fraction,
        "largest_regulators": sorted(gene for gene in genes if outdegree[gene] == max(outdegree.values())),
        "weakly_connected": True,
        "has_cycles": len(order) < n_genes,
        "indegree": indegree,
        "outdegree": outdegree,
    }


def _write_induced_network(directory, genes, size):
    """Copy the parent map interactions whose two endpoints belong to this graph."""
    chosen = set(genes)
    if len(chosen) != size or genes != sorted(chosen):
        raise ValueError("The registered gene list must be unique, sorted, and match the graph size")
    edges = []
    source_map = directory / "original-parent.tsv"
    subnetwork = directory / "connected-parent.tsv"
    with source_map.open("rb") as parent, subnetwork.open("wb") as output:
        for line in parent:
            source, target, sign = line.decode().split()
            if source in chosen and target in chosen:
                output.write(line)
                edges.append((source, target, sign))
    return edges


def generate(config, definition, root):
    """Materialize one registered graph with its fixed seed and native input."""
    name, family, n_genes = definition["id"], definition["family"], definition["size"]
    seed, parent = definition["seed"], definition["parent"]
    if family not in {*_SOURCES, "fflatt"} or parent not in {"original", "connected"}:
        raise ValueError(f"Unrecognized topology specification for {name}")
    if parent == "connected" and family not in _SOURCES:
        raise ValueError(f"Missing native parent network for {name}")
    images = {key: image["id"] for key, image in config["images"].items()}
    java = [
        "-Djava.awt.headless=true", "-Duser.home=/tmp", "-Xmx2g", "-jar",
        "/opt/genenetweaver/gnw-3.1.2b.jar",
    ]
    network_root = root / "networks" / name
    directory = network_root / "native"
    directory.mkdir(parents=True)

    def native(simulator: str, entrypoint: str, arguments: list[str], label: str) -> None:
        command = [
            "docker", "run", "--rm", "--pull", "never", "--network", "none",
            "--cpus", "2", "--memory", "4g", "--pids-limit", "256",
            "--user", f"{os.getuid()}:{os.getgid()}",
            "-v", f"{directory}:/work/out", "-w", "/tmp",
            "--entrypoint", entrypoint, images[simulator], *arguments,
        ]
        log = directory / f"{label}.log"
        with log.open("w") as stream:
            result = subprocess.run(command, timeout=600, stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode:
            details = "\n".join(log.read_text(errors="replace").splitlines()[-15:])
            raise RuntimeError(f"{name}: {simulator} {label} failed ({result.returncode})\n{details}")

    source_path = "/opt/genenetweaver/src/ch/epfl/lis/networks/" + _SOURCES[family] if family in _SOURCES else None
    settings_script = (
        "from pathlib import Path\nimport re\n"
        "original = Path('/opt/genenetweaver/sandbox/settings.txt').read_text()\n"
        f"settings, count = re.subn(r'(?m)^randomSeed\\s*=.*$', 'randomSeed = {seed}', original)\n"
        "assert count == 1\n"
        "Path('/work/out/settings.txt').write_text(settings)\n"
    )
    if parent == "connected":
        settings_script += f"Path('/work/out/original-parent.tsv').write_bytes(Path({source_path!r}).read_bytes())\n"
    native("genenetweaver", "python", ["-c", settings_script], "settings")
    common = ["-c", "/work/out/settings.txt", "--output-path", "/work/out"]
    if parent == "connected":
        parent_edges = _write_induced_network(directory, definition["genes"], n_genes)
        native("genenetweaver", "java", java + [
            "--transform", *common, "--input-net", "/work/out/connected-parent.tsv",
            "--input-net-format", "0", "--network-name", name, "--output-net-format", "4",
        ], "transform")
    elif family == "fflatt":
        native("genesnake", "python", ["-c",
            "import csv,random,numpy as np,genesnake as gs\n"
            f"np.random.seed({seed});random.seed({seed})\n"
            f"a=gs.grn.make_FFLATTnetwork(NETWORK_SIZE={n_genes},SPARSITY=2.328,"
            f"RANDOM_SEED={seed},FFL_ENRICHED=1,NO_CYCLES=0,OTHER_MOTIF_DEPLETION=1,"
            "SHUFFLED=0,N_CORES_TO_USE=2,self_loops=False,edge_bias=0.62)\n"
            "a.to_csv('/work/out/native_weighted_matrix.tsv',sep='\\t')\n"
            "with open('/work/out/native_signed_edges.tsv','w',newline='') as f:\n"
            " writer=csv.writer(f,delimiter='\\t')\n"
            " for source in a.index:\n"
            "  for target in a.columns:\n"
            "   value=float(a.loc[source,target])\n"
            "   if source!=target and value:writer.writerow([source,target,'+' if value>0 else '-'])\n",
        ], "native")
        native("genenetweaver", "java", java + [
            "--transform", *common, "--input-net", "/work/out/native_signed_edges.tsv",
            "--input-net-format", "0", "--network-name", name, "--output-net-format", "4",
        ], "transform")
    else:
        native("genenetweaver", "java", java + [
            "--extract", *common, "--input-net", source_path,
            "--input-net-format", "0", "--network-name", name,
            "--subnet-size", str(n_genes), "--num-subnets", "1", "--random-seed",
            "--rat-selection", "20", "--min-num-regulators", str(config["native_min_regulators"]), "--output-net-format", "4",
        ], "extract")
    sbml = _one(directory, "*.xml")
    native("genenetweaver", "java", java + [
        "--transform", *common, "--input-net", f"/work/out/{sbml.name}",
        "--input-net-format", "4", "--output-net-format", "0",
    ], "export")
    with sbml.with_suffix(".tsv").open(newline="") as handle:
        exported = [row for row in csv.reader(handle, delimiter="\t") if row]
    if any(len(row) != 3 for row in exported):
        raise RuntimeError("Unexpected native TSV edge format")
    edges = sorted([source, target, sign] for source, target, sign in exported if source != target)
    if parent == "connected":
        expected = {(s, t) for s, t, _ in parent_edges if s != t}
        known_signs = {(s, t, sign) for s, t, sign in parent_edges if s != t and sign in {"+", "-"}}
        if {(s, t) for s, t, _ in edges} != expected or not known_signs.issubset(map(tuple, edges)):
            raise RuntimeError("GNW export changed an induced interaction or a known parent sign")
    summary = _graph_summary(edges, n_genes, config["max_hub_edge_fraction"])
    genes = sorted(summary["indegree"])
    scale = 0.8 / max(summary["indegree"].values())
    matrix = -np.eye(n_genes)
    gene_index = {gene: index for index, gene in enumerate(genes)}
    for source, target, sign in edges:
        matrix[gene_index[target], gene_index[source]] = scale * (1 if sign == "+" else -1)
    eigenvalues = np.linalg.eigvals(matrix)
    native_dt = 0.1 * float(np.min(np.abs(eigenvalues)))
    slow_rate = -float(np.max(eigenvalues.real))
    if slow_rate <= 0 or native_dt <= 0 or np.max(np.abs(1 + native_dt * eigenvalues)) >= 1:
        raise ValueError(f"Unstable GeneSPIDER2 coefficients for {name}")
    stride = math.ceil(5 / slow_rate / ((config["n_points"] - 1) * native_dt))
    shared = network_root / "shared_signed_network.tsv"
    weighted = network_root / "genespider2_weighted_network.tsv"
    for path, weight in ((shared, 1), (weighted, scale)):
        with path.open("w", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t")
            writer.writerow(["target", "regulator", "effect"])
            writer.writerows((target, source, weight if sign == "+" else -weight) for source, target, sign in edges)
            if path == weighted:
                writer.writerows((gene, gene, -1) for gene in genes)
    topology = {
        "id": name,
        "family": family,
        "seed": seed,
        "genes": genes,
        "edges": edges,
        "sbml": str(sbml.relative_to(network_root)),
        "shared_network": shared.name,
        "weighted_network": weighted.name,
        "stride": stride,
        "genespider2": {
            "offdiagonal_scale": scale, "native_dt": native_dt, "slow_rate": slow_rate,
            "diagonal": -1, "horizon_slow_time_constants": 5,
        },
        "edge_sha256": hashlib.sha256(json.dumps(edges, separators=(",", ":")).encode()).hexdigest(),
        "summary": summary,
        "images": images,
        "sha256": {str(path.relative_to(network_root)): _sha256(path) for path in (sbml, shared, weighted)},
    }
    _dump(network_root / "topology.json", topology)


def degree_pairs(summary):
    """Isomorphic directed graphs have the same multiset, independent of gene names."""
    return sorted((n, summary["outdegree"][g]) for g, n in summary["indegree"].items())


def compare(family, edges, summary, previous):
    checks = []
    candidates = {(s, t) for s, t, _ in edges}
    genes = set(summary["indegree"])
    for ref in previous:
        if ref["family"] != family:
            continue
        if family == "fflatt":
            # G1, G2, ... are local identifiers. Degree pairs allow a label-independent check.
            checks.append({"id": ref["id"], "same_degree_pairs":
                           degree_pairs(summary) == degree_pairs(ref["summary"])})
        else:
            reference = {(s, t) for s, t, _ in ref["edges"]}
            common = len(reference & candidates)
            checks.append({"id": ref["id"], "shared_directed_edges": common,
                           "edge_overlap": common / min(len(reference), len(candidates)),
                           "shared_genes": len(genes.intersection(ref["genes"]))})
    return checks


def read_graph(base, definition, hub_limit=1.0):
    name = definition["id"]
    directory = base / "topologies" / name / "networks" / name
    return _validated_graph(directory, definition, hub_limit)


def _validated_graph(directory, definition, hub_limit):
    name = definition["id"]
    graph = json.loads((directory / "topology.json").read_text())
    if graph["id"] != name or graph["family"] != definition["family"] or graph["seed"] != definition["seed"]:
        raise ValueError(f"Graph identity mismatch: {name}")
    # Fingerprint the three simulator inputs by role, including independently materialized copies.
    paths = [Path(graph[key]) for key in ("sbml", "shared_network", "weighted_network")]
    if any(path.is_absolute() or ".." in path.parts for path in paths):
        raise ValueError(f"Invalid simulator input paths for {name}")
    hashes = {str(path): _sha256(directory / path) for path in paths}
    sbml, shared, weighted = paths
    canonical = {
        f"native/{sbml.name}": hashes[str(sbml)],
        "shared_signed_network.tsv": hashes[str(shared)],
        "genespider2_weighted_network.tsv": hashes[str(weighted)],
    }
    if hashes != graph["sha256"] or canonical != definition["files"]:
        raise ValueError(f"Simulator inputs differ from the registered graph {name}")
    actual = hashlib.sha256(json.dumps(graph["edges"], separators=(",", ":")).encode()).hexdigest()
    if actual != graph["edge_sha256"] or actual != definition["edge_sha256"]:
        raise ValueError(f"Corrupt signed edges for {name}")
    summary = _graph_summary(graph["edges"], definition["size"], hub_limit)
    if graph["summary"] != summary or graph["genes"] != sorted(summary["indegree"]):
        raise ValueError(f"Graph summary differs from its directed edges: {name}")
    if "genes" in definition and graph["genes"] != definition["genes"]:
        raise ValueError(f"Genes differ from the registered graph {name}")
    if graph["stride"] != definition["stride"]:
        raise ValueError(f"Sampling stride differs from the registered graph {name}")
    return graph


def build(config_path, out, development, jobs=1):
    config = json.loads(config_path.read_text())
    if jobs < 1:
        raise ValueError("--jobs must be positive")
    checkout = Path(__file__).resolve().parents[1]
    if out.resolve().is_relative_to(checkout):
        raise ValueError("Choose --out outside the NANIWA checkout for generated benchmark data")
    out.mkdir(parents=True, exist_ok=True)
    references = []
    receipt = {"recipe_sha256": _sha256(config_path)}
    if config["phase"] == "final":
        if development is None:
            raise ValueError("Final topology generation needs --development with the 18 development graphs")
        development_recipe = Path(__file__).parent / "01-construction/topologies.json"
        development_config = json.loads(development_recipe.read_text())
        references = [read_graph(development, g, development_config["max_hub_edge_fraction"])
                      for g in development_config["graphs"]]
        receipt["development_fingerprints"] = {g["id"]: g["edge_sha256"] for g in references}
    images = config["images"]
    for image in images.values():
        image_id = subprocess.check_output(["docker", "image", "inspect", "--format", "{{.Id}}",
                                            image["tag"]], text=True).strip()
        if image_id != image["id"]:
            raise ValueError(f"Image {image['tag']} does not match frozen ID {image['id']}")
    directory = out / "topologies"
    directory.mkdir(exist_ok=True)
    with (directory / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

        def validate(folder, definition, earlier):
            name = definition["id"]
            graph = _validated_graph(folder / "networks" / name, definition, config["max_hub_edge_fraction"])
            if config["phase"] == "final":
                if graph["summary"]["n_regulators"] < 9:
                    raise ValueError(f"Final graph {name} lacks nine regulators")
                for check in compare(graph["family"], graph["edges"], graph["summary"], earlier):
                    if check.get("same_degree_pairs") or check.get("edge_overlap", 0) > config["max_edge_overlap"]:
                        raise ValueError(f"Overlap audit failed for {name}: {check}")
            return graph

        def build_family(family):
            earlier = list(references)
            for definition in (g for g in config["graphs"] if g["family"] == family):
                name = definition["id"]
                dest = directory / name
                if dest.exists():
                    graph = validate(dest, definition, earlier)
                else:
                    with TemporaryDirectory(prefix=f".{name}-", dir=directory) as temp:
                        stage = Path(temp) / name
                        generate(config, definition, stage)
                        graph = validate(stage, definition, earlier)
                        stage.rename(dest)
                earlier.append(graph)
                print(f"{name}  seed={graph['seed']}", flush=True)

        with ThreadPoolExecutor(max_workers=jobs) as pool:
            list(pool.map(build_family, config["families"]))
        _dump(directory / "complete.json", {**receipt, "graphs": [g["id"] for g in config["graphs"]]})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path, help="External benchmark root")
    parser.add_argument("--development", type=Path, help="Development graph root for the final overlap check")
    parser.add_argument("--jobs", type=int, default=1, help="Independent parent families processed concurrently (up to 3)")
    args = parser.parse_args()
    build(args.config.resolve(), args.out.resolve(), args.development, args.jobs)
