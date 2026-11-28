"""Pinned native GreyNet inference on one observed longitudinal bulk course."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
import sys
import time
import warnings
from pathlib import Path


UPSTREAM_COMMIT = "3031bc7383167f27a440d829bce3f2c1897015c8"
UPSTREAM_SOURCES = {
    "Regression.py": "c59a296c1085b5096e4f2a8147cff7c9cf37b0dd21f80c0510ec04b886e9b035",
    "DyGreyAssociation.py": "01310acc6136bfa5f92900102d9a3b54dc5bafa57cb2216e617b9098587be5bc",
}


def patched_source(raw: bytes, filename: str) -> str:
    """Expose native controls and observe completed work without rewriting formulas."""
    if filename not in UPSTREAM_SOURCES or hashlib.sha256(raw).hexdigest() != UPSTREAM_SOURCES[filename]:
        raise ValueError("Unexpected GreyNet source: pinned filename and SHA-256 are required")
    text = raw.decode("utf-8")

    def replace(old, new, count=1):
        nonlocal text
        if text.count(old) != count:
            raise ValueError(f"GreyNet patch context changed: {old!r}")
        text = text.replace(old, new)

    if filename == "Regression.py":
        replace(
            "def __init__(self,data,type='RF'):",
            "def __init__(self,data,type='RF', *, andrea_lag=1, andrea_seed=0, "
            "andrea_threads=1, andrea_fit_callback=None):",
        )
        replace(
            "        self.type = type\n",
            "        self.type = type\n"
            "        self.andrea_lag = andrea_lag\n"
            "        self.andrea_seed = andrea_seed\n"
            "        self.andrea_threads = andrea_threads\n"
            "        self.andrea_fit_callback = andrea_fit_callback\n",
        )
        replace(
            "RandomForestRegressor(n_estimators=1000, max_features='sqrt')",
            "RandomForestRegressor(n_estimators=1000, max_features='sqrt', "
            "random_state=self.andrea_seed, n_jobs=self.andrea_threads)",
        )
        replace(
            "subsample=0.8, bylevel=0.6, bytree=0.6)",
            "subsample=0.8, bylevel=0.6, bytree=0.6, "
            "random_state=self.andrea_seed, n_jobs=self.andrea_threads)",
        )
        replace("self.data_process(index=index)", "self.data_process(index=index, lag=self.andrea_lag)", 4)
        replace(
            "            VIM[:, index] = imp\n",
            "            VIM[:, index] = imp\n"
            "            if self.andrea_fit_callback is not None:\n"
            "                self.andrea_fit_callback(index + 1, index, model, imp)\n",
            4,
        )
    else:
        # Evaluation is absent upstream and none of its symbols are used here.
        replace("from Evaluation import *\n", "")
        replace(
            "def __init__(self,data,columns,iteration=0,numpoint=21,max_win=15,win_len=None):",
            "def __init__(self,data,columns,iteration=0,numpoint=21,max_win=15,win_len=None, "
            "*, andrea_rho=0.5, andrea_target_callback=None):",
        )
        replace(
            "        self.max_win= max_win\n",
            "        self.max_win= max_win\n"
            "        self.andrea_rho = andrea_rho\n"
            "        self.andrea_target_callback = andrea_target_callback\n",
        )
        replace("self.gra_cal(co_seq,re_seq)", "self.gra_cal(co_seq,re_seq,rho=self.andrea_rho)")
        replace(
            "            ind,val = self.GRA(data,win_list,col)\n",
            "            ind,val = self.GRA(data,win_list,col)\n"
            "            if self.andrea_target_callback is not None:\n"
            "                self.andrea_target_callback(i + 1, i, ind, val, win_list)\n",
        )
    compile(text, filename, "exec")
    return text


def prepare_upstream(*, output_dir: Path, source_dir: Path | None = None):
    """Install only the two hash-verified public native inference modules."""
    import urllib.request

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    receipt = {"upstream_commit": UPSTREAM_COMMIT, "sources": {}}
    for filename in UPSTREAM_SOURCES:
        raw = (
            (Path(source_dir) / filename).read_bytes() if source_dir is not None
            else urllib.request.urlopen(
                f"https://raw.githubusercontent.com/zpliulab/GreyNet/{UPSTREAM_COMMIT}/{filename}", timeout=60
            ).read()
        )
        patched = patched_source(raw, filename).encode("utf-8")
        (output_dir / filename.replace(".py", ".original.py")).write_bytes(raw)
        (output_dir / filename).write_bytes(patched)
        receipt["sources"][filename] = {
            "original_sha256": hashlib.sha256(raw).hexdigest(),
            "patched_sha256": hashlib.sha256(patched).hexdigest(),
        }
    (output_dir / "greynet-source-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


if __name__ == "__main__" and sys.argv[1:2] == ["--prepare-upstream"]:
    parser = argparse.ArgumentParser(description="Prepare verified native source; no fits")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path)
    arguments = parser.parse_args(sys.argv[2:])
    print(json.dumps(prepare_upstream(output_dir=arguments.output_dir, source_dir=arguments.source_dir), sort_keys=True))
    raise SystemExit(0)


import numpy as np

from _column_metadata import load_column_descriptors, summarize_temporal_metadata
from _run_tool_common import load_execution_mode, load_params, validate_runtime_inputs, write_progress

DEFAULTS = {"regressor": "RF", "grey_threshold": 0.4, "max_win": 15, "win_len": 0, "lag": 1, "rho": 0.5, "seed": 0}


def parse_params(raw):
    if not isinstance(raw, dict):
        raise ValueError("GreyNet parameters must be a JSON object")
    if set(raw) - set(DEFAULTS):
        raise ValueError(f"Unknown GreyNet parameters: {sorted(set(raw) - set(DEFAULTS))}")
    params = {**DEFAULTS, **raw}
    if params["regressor"] not in ("RF", "Xgboost", "Lasso", "Ridge"):
        raise ValueError("regressor must be RF, Xgboost, Lasso or Ridge")
    for name, minimum, maximum in (("max_win", 3, None), ("win_len", 0, None), ("lag", 1, None), ("seed", 0, 2**32 - 1)):
        value = params[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum or maximum is not None and value > maximum:
            raise ValueError(f"{name} must be an integer from {minimum}" + (f" to {maximum}" if maximum is not None else ""))
    if params["win_len"] in (1, 2):
        raise ValueError("win_len must be zero (native automatic selection) or at least three")
    for name in ("grey_threshold", "rho"):
        value = params[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"{name} must be a finite number from zero to one")
    return params


def read_matrix(path):
    with Path(path).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle, delimiter="\t"))
    if not rows or len(rows[0]) < 2 or len(rows) < 3:
        raise ValueError("GreyNet requires a genes-by-observations TSV with at least two genes")
    header, columns = rows[0], rows[0][1:]
    if any(not item or item != item.strip() for item in header) or len(set(header)) != len(header):
        raise ValueError("Expression headings must be unique exact nonempty identifiers")
    genes, values = [], []
    for number, row in enumerate(rows[1:], 2):
        if not row:
            continue
        if len(row) != len(header):
            raise ValueError(f"Malformed expression row {number}")
        gene = row[0]
        if not gene or gene != gene.strip() or gene in genes:
            raise ValueError("Expression genes must be unique exact nonempty identifiers")
        genes.append(gene)
        values.append([float(value) for value in row[1:]])
    expression = np.asarray(values, dtype=np.float64)
    if len(genes) < 2 or not np.isfinite(expression).all():
        raise ValueError("GreyNet requires at least two genes and finite expression")
    return genes, columns, expression


def prepare_data(expression_path, extra_dir, params):
    """Select stationary roles out and validate one course without merging units."""
    params = parse_params(params)
    genes, columns, expression = read_matrix(expression_path)
    descriptors = load_column_descriptors(Path(extra_dir) / "column_descriptors.tsv", columns)
    temporal = summarize_temporal_metadata(descriptors)
    if temporal["sampling_design"] != "longitudinal":
        raise ValueError("GreyNet requires longitudinal observed bulk time series")
    groups, observations, stationary = {}, set(), []
    for position, column in enumerate(columns):
        row = descriptors[column]
        if row["observation_type"] == "steady_state":
            stationary.append(column)
            continue
        if row["observation_id"] in observations:
            raise ValueError("GreyNet does not implicitly average time-series technical replicates")
        observations.add(row["observation_id"])
        groups.setdefault(row["series_id"], []).append((row["timepoint"], position, row))
    if len(groups) != 1:
        raise ValueError("GreyNet requires exactly one time series; separate courses are not concatenated or averaged")
    series_id, entries = next(iter(groups.items()))
    entries.sort(key=lambda item: item[0])
    times = [item[0] for item in entries]
    conditions = {item[2]["condition"] for item in entries}
    replicates = {item[2]["replicate"] for item in entries}
    if len(conditions) != 1 or len(replicates) != 1:
        raise ValueError("One series must identify one condition and biological replicate")
    if len(times) < 4:
        raise ValueError("GreyNet requires at least four observed timepoints")
    intervals = np.diff(times)
    interval = float(intervals[0])
    if not math.isfinite(interval) or interval <= 0 or any(
        not math.isfinite(float(value)) or value <= 0
        or not math.isclose(float(value), interval, rel_tol=1e-9, abs_tol=1e-12)
        for value in intervals
    ):
        raise ValueError("GreyNet requires strictly increasing regular observed times")
    effective_win_len = params["win_len"] or int(0.66 * len(times))
    if effective_win_len < 3:
        raise ValueError("GreyNet requires an initial window of at least three observations; choose win_len or provide five timepoints")
    if len(times) - params["lag"] < 2:
        raise ValueError("GreyNet requires at least two rows after applying lag")
    positions = [item[1] for item in entries]
    matrix = expression[:, positions].T.copy()
    predictors, targets = matrix[:-params["lag"]], matrix[params["lag"]:]
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        standardized = [(value - value.mean(axis=0)) / value.std(axis=0) for value in (predictors, targets)]
    if any(not np.isfinite(value).all() for value in standardized):
        raise ValueError("GreyNet requires finite nonconstant lagged predictors and targets for native z-score standardization")
    if params["regressor"] in ("RF", "Xgboost") and any(not np.isfinite(value.astype(np.float32)).all() for value in standardized):
        raise ValueError("GreyNet standardized expression must remain finite at native tree float32 precision")
    native_genes = [f"ANDREA_GENE_{index:08d}" for index in range(len(genes))]
    series = [{"series_id": series_id, "condition": next(iter(conditions)), "replicate": next(iter(replicates)),
               "columns": [columns[position] for position in positions], "timepoints": times}]
    return {
        "genes": genes, "native_genes": native_genes, "columns": columns,
        "expression": expression, "time_series": matrix, "descriptors": descriptors,
        "temporal": temporal, "series": series, "unused_steady_state_columns": stationary,
        "timepoints_per_series": len(times), "sampling_interval": interval,
        "effective_win_len": effective_win_len, "effective_training_rows": len(predictors),
        "lagged_predictors": standardized[0], "lagged_targets": standardized[1],
    }


def _json_value(value):
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path, payload):
    Path(path).write_text(json.dumps(_json_value(payload), indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def _write_matrix(path, values, rows, columns):
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["row", *columns])
        writer.writerows([name, *(repr(float(value)) for value in row)] for name, row in zip(rows, values))


def export_network(path, matrix, genes):
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (len(genes), len(genes)) or not np.isfinite(matrix).all() or np.any(matrix < 0):
        raise ValueError("GreyNet confidence must be a finite nonnegative regulator-by-target matrix")
    edges = [(source, target, float(matrix[i, j])) for i, source in enumerate(genes) for j, target in enumerate(genes) if source != target and matrix[i, j] > 0]
    edges.sort(key=lambda row: (-row[2], row[0], row[1]))
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["source", "target", "score", "sign", "evidence", "context"])
        writer.writerows((source, target, repr(score), "?", "association", "global") for source, target, score in edges)
    return len(edges)


def _resource_snapshot(model):
    from threadpoolctl import threadpool_info

    result = {"pid": os.getpid(), "model_params": model.get_params(), "native_pools": threadpool_info(),
              "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None, "cgroups": {}}
    if hasattr(model, "get_booster"):
        result["native_config"] = json.loads(model.get_booster().save_config())
        result["observed_tree_count"] = model.get_booster().num_boosted_rounds()
    elif hasattr(model, "estimators_"):
        result["observed_tree_count"] = len(model.estimators_)
    if hasattr(model, "n_iter_"):
        result["observed_iterations"] = model.n_iter_
    for name in ("cpu.max", "memory.max", "cpu.stat", "memory.current", "memory.peak"):
        path = Path("/sys/fs/cgroup") / name
        if path.is_file():
            result["cgroups"][name] = path.read_text().strip()
    return result


def _session(association, regression):
    session = {"upstream_commit": UPSTREAM_COMMIT, "packages": {}, "sources": {}}
    for package in ("numpy", "scipy", "pandas", "scikit-learn", "xgboost-cpu", "joblib", "threadpoolctl"):
        try:
            session["packages"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            session["packages"][package] = "unavailable in recording test environment"
    for implementation in (association, regression):
        source_path = getattr(implementation, "__file__", None)
        if source_path:
            session["sources"][Path(source_path).name] = hashlib.sha256(Path(source_path).read_bytes()).hexdigest()
    receipt = Path("/app/greynet-source-receipt.json")
    if receipt.is_file():
        session["source_receipt"] = json.loads(receipt.read_text())
    return session


def run(*, input_path, params_path, extra_dir, output_dir, threads, upstream=None):
    input_path, params_path, extra_dir, output_dir = map(Path, (input_path, params_path, extra_dir, output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = output_dir / "provenance"
    raw.mkdir(exist_ok=True)
    completed, total, last_percent = 0, None, 0

    def progress(*, phase, message, percent, status="running", error=None):
        nonlocal last_percent
        last_percent = max(last_percent, percent)
        write_progress(output_dir / "progress.json", status=status, phase=phase, message=message, percent=last_percent, completed=completed, total=total, error=error)
        event = json.loads((output_dir / "progress.json").read_text())
        event["observed_monotonic_ns"] = time.monotonic_ns()
        with (raw / "progress-events.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")

    progress(phase="validate", message="Validating GreyNet temporal inputs", percent=0)
    try:
        validate_runtime_inputs(input_path=input_path, params_path=params_path, extra_dir=extra_dir, threads=threads)
        load_execution_mode(params_path, supported_modes={"global"})
        params = parse_params(load_params(params_path))
        data = prepare_data(input_path, extra_dir, params)
        total = 2 * len(data["genes"])
        resolved = {
            "params": params, "genes": data["genes"], "native_genes": data["native_genes"],
            "temporal": data["temporal"], "series": data["series"], "column_descriptors": data["descriptors"],
            "unused_steady_state_columns": data["unused_steady_state_columns"],
            "timepoints_per_series": data["timepoints_per_series"], "independent_series_count": 1,
            "sampling_interval": data["sampling_interval"], "lag_duration": params["lag"] * data["sampling_interval"],
            "effective_training_rows": data["effective_training_rows"], "effective_win_len": data["effective_win_len"], "threads": threads,
            "model_settings": "Unchanged native constructors; only RF/XGBoost seed and thread count are controlled. XGBoost lr/ntrees/bylevel/bytree remain unused upstream keyword arguments; effective configuration is recorded for every fitted target.",
            "normalization": "Native lagged z-scores followed by the selected native regressor. Lasso/Ridge retain normalize=True with scikit-learn 1.1.3. Native grey threshold is a binary multiplicative gate.",
            "input_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in (input_path, params_path, extra_dir / "column_descriptors.tsv")},
        }
        _write_json(raw / "resolved.json", resolved)
        _write_matrix(raw / "native-timeseries.tsv", data["time_series"], data["series"][0]["columns"], data["genes"])
        association = upstream["association"] if upstream is not None else importlib.import_module("DyGreyAssociation")
        regression = upstream["regression"] if upstream is not None else importlib.import_module("Regression")
        _write_json(raw / "session.json", _session(association, regression))
        windows = []

        def associated(count, target_index, indices, values, lengths):
            nonlocal completed
            if count != completed + 1 or target_index != completed or count > len(data["genes"]):
                raise ValueError("GreyNet returned inconsistent completed grey-association observations")
            if not np.isfinite(values).all() or len(values) != len(data["genes"]) or set(indices) != set(data["native_genes"]):
                raise ValueError("Native GreyNet grey association is non-finite or inconsistent; the input lies outside its numerical domain")
            completed = count
            windows.append({"target_gene": data["genes"][target_index], "window_lengths": lengths})
            progress(phase="grey_association", message=f"Completed grey association for {data['genes'][target_index]}", percent=5 + 85 * completed // total)

        def fitted(count, target_index, model, importance):
            nonlocal completed
            if count != completed - len(data["genes"]) + 1 or target_index != count - 1 or count > len(data["genes"]):
                raise ValueError("GreyNet returned inconsistent completed regression observations")
            if len(importance) != len(data["genes"]) or not np.isfinite(importance).all() or np.any(np.asarray(importance) < 0):
                raise ValueError("Native GreyNet regression produced invalid importance")
            snapshot = _resource_snapshot(model)
            snapshot["target_gene"] = data["genes"][target_index]
            _write_json(raw / f"model-{target_index:08d}.json", snapshot)
            completed += 1
            progress(phase="regression", message=f"Fitted {params['regressor']} for {data['genes'][target_index]}", percent=5 + 85 * completed // total)

        import pandas as pd
        from threadpoolctl import threadpool_limits

        frame = pd.DataFrame(data["time_series"], columns=data["native_genes"])
        progress(phase="grey_association", message="Computing native adaptive grey association", percent=5)
        caught = []
        try:
            # Targets are sequential; the selected backend owns its CPU budget.
            with warnings.catch_warnings(record=True) as caught, threadpool_limits(
                limits=threads if params["regressor"] in ("Lasso", "Ridge") else 1, user_api="blas"
            ), threadpool_limits(limits=threads if params["regressor"] == "Xgboost" else 1, user_api="openmp"):
                warnings.simplefilter("always")
                grey = association.ASW_GRA(
                    data=frame.values, columns=frame.columns.values, iteration=0,
                    numpoint=len(frame), max_win=params["max_win"], win_len=params["win_len"] or None,
                    andrea_rho=params["rho"], andrea_target_callback=associated,
                )
                indices, values = grey.ADAPATIVE_GRA()
                grey_series = association.GRA_Series(indices, values, frame.columns.values)
                grey_values = np.asarray(grey_series, dtype=np.float64).reshape(len(frame.columns), -1)
                if not np.isfinite(grey_values).all():
                    raise ValueError("Native GreyNet grey association must be finite before thresholding")
                _write_matrix(raw / "grey-association.tsv", grey_values, data["genes"], data["genes"])
                _write_json(raw / "adaptive-windows.json", windows)
                gate = association.normalize(grey_series.copy(), params["grey_threshold"])
                _write_matrix(raw / "grey-gate.tsv", np.asarray(gate).reshape(len(frame.columns), -1), data["genes"], data["genes"])
                progress(phase="regression", message=f"Fitting native GreyNet {params['regressor']} target models", percent=5 + 85 * completed // total)
                model = regression.Regression(
                    frame, type=params["regressor"], andrea_lag=params["lag"],
                    andrea_seed=params["seed"], andrea_threads=threads, andrea_fit_callback=fitted,
                )
                probabilities = model.fit()
                if not np.isfinite(np.asarray(probabilities, dtype=np.float64)).all():
                    raise ValueError("Native GreyNet regression scores must be finite before gating")
                _write_matrix(raw / "regression-importance.tsv", np.asarray(probabilities).reshape(len(frame.columns), -1), data["genes"], data["genes"])
                progress(phase="gating", message="Applying the native binary grey-association gate", percent=92)
                matrix = np.asarray(probabilities * gate, dtype=np.float64).reshape(len(frame.columns), -1)
        finally:
            _write_json(raw / "native-warnings.json", [{"category": item.category.__name__, "message": str(item.message)} for item in caught])
        if completed != total:
            raise ValueError("GreyNet returned before every native target calculation was observed")
        progress(phase="export", message="Exporting native GreyNet confidence", percent=95)
        _write_matrix(raw / "native-score-matrix.tsv", matrix, data["genes"], data["genes"])
        resolved["network_edge_count"] = export_network(output_dir / "network.csv", matrix, data["genes"])
        _write_json(raw / "resolved.json", resolved)
        progress(status="completed", phase="done", message="GreyNet network exported", percent=100)
        return resolved
    except Exception as error:
        progress(status="failed", phase="failed", message="GreyNet failed", percent=last_percent, error=str(error))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--params", required=True, type=Path)
    parser.add_argument("--extra", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--threads", required=True, type=int)
    arguments = parser.parse_args()
    run(input_path=arguments.input, params_path=arguments.params, extra_dir=arguments.extra, output_dir=arguments.output_dir, threads=arguments.threads)


if __name__ == "__main__":
    main()
