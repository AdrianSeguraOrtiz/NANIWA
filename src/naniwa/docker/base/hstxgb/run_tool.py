"""Pinned HSTXGB expression-only inference on observed longitudinal bulk courses."""

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
from pathlib import Path

UPSTREAM_COMMIT = "689705a91a7f40ea39f15e6f8bdcd706c5c32fa0"
UPSTREAM_ZIP_SHA256 = "187b66b7c9424c5e8c1162f8b46d1e131c26ab627f65ae25a95bf3c613648b78"
UPSTREAM_SOURCE = {
    "path": "Ecoli/PXBNet_Ecoli.py",
    "module": "PXBNet_Ecoli",
    "sha256": "d044cab9f72b1b16d9be0a5ceb0a18be5839c53f8b625b70b2db33f7bda2c7de",
}


def patched_source(raw: bytes) -> str:
    """Adapt course geometry and observe native fits without replacing the method."""
    if hashlib.sha256(raw).hexdigest() != UPSTREAM_SOURCE["sha256"]:
        raise ValueError("Unexpected HSTXGB source: pinned SHA-256 is required")
    text = raw.decode("utf-8")

    def replace(old, new, count=1):
        nonlocal text
        if text.count(old) != count:
            raise ValueError(f"HSTXGB patch context changed: {old!r}")
        text = text.replace(old, new)

    controls = (
        ", *, andrea_seed=42, andrea_threads=1, andrea_fit_callback=None, "
        "andrea_search_callback=None, andrea_phase_callback=None"
    )
    replace(
        "def main(data, num_gene, time_points, k, alpha, feature_num, iter_num):",
        "def main(data, num_gene, time_points, k, alpha, feature_num, iter_num" + controls + "):",
    )
    replace(
        "    data1 = np.hstack((data[0,:,:], data[1,:,:], data[2,:,:]))\n"
        "    data2 = np.zeros((3, num_gene, int((np.shape(data1)[1])/3)))\n\n"
        "    for i in range(3):\n",
        "    data1 = np.hstack(tuple(data[i,:,:] for i in range(data.shape[0])))\n"
        "    data2 = np.zeros((data.shape[0], num_gene, time_points))\n\n"
        "    for i in range(data.shape[0]):\n",
    )
    replace(
        "np.transpose(np.hstack((m_accu[0,:,:], m_accu[1,:,:], m_accu[2,:,:])))",
        "np.transpose(np.hstack(tuple(m_accu[i,:,:] for i in range(data_3d))))",
    )
    replace(
        "mgr_xg = xgboost_weight(data_tm, feature_num, iter_num)",
        "mgr_xg = xgboost_weight(data_tm, feature_num, iter_num, "
        "andrea_seed=andrea_seed, andrea_threads=andrea_threads, "
        "andrea_fit_callback=andrea_fit_callback, "
        "andrea_search_callback=andrea_search_callback, andrea_phase_callback=andrea_phase_callback)",
    )
    replace(
        "def xgboost_weight(data, feature_num, iter_num):",
        "def xgboost_weight(data, feature_num, iter_num" + controls + "):",
    )
    replace(
        "    # Use a statistical technology to refine the inferred GRN.\n",
        "    if andrea_phase_callback is not None:\n"
        "        andrea_phase_callback('aggregation')\n\n"
        "    # Use a statistical technology to refine the inferred GRN.\n",
    )
    replace("            random_state=42,", "            random_state=andrea_seed,", count=2)
    replace("            verbosity=0\n", "            verbosity=0,\n            n_jobs=1\n")
    replace(
        "            n_jobs=-1,                  # n_jobs=-1 indicates using all processors",
        "            n_jobs=andrea_threads,\n            pre_dispatch=andrea_threads,",
    )
    replace(
        "        random_search.fit(x, y)\n",
        "        if andrea_phase_callback is not None:\n"
        "            andrea_phase_callback('search', i)\n"
        "        random_search.fit(x, y)\n"
        "        if andrea_search_callback is not None:\n"
        "            andrea_search_callback(i, random_search)\n",
    )
    replace("        dtrain = xgb.DMatrix(x, y)\n", "        dtrain = xgb.DMatrix(x, y, nthread=andrea_threads)\n")
    replace(
        "        plst = list(random_search.best_params_.items())\n"
        "        model = xgb.train(plst, dtrain, iter_num)\n",
        "        final_params = dict(random_search.best_params_)\n"
        "        final_params['nthread'] = andrea_threads\n"
        "        if andrea_phase_callback is not None:\n"
        "            andrea_phase_callback('final_fit', i)\n"
        "        plst = list(final_params.items())\n"
        "        model = xgb.train(plst, dtrain, iter_num)\n"
        "        if andrea_fit_callback is not None:\n"
        "            andrea_fit_callback(i + 1, i, model, final_params)\n",
    )
    replace(
        "            num = re.findall(r'\\d+', importance[j][0])\n"
        "            num = np.array(num)\n"
        "            num = np.core.defchararray.strip(num, '()')\n"
        "            num = int(num)",
        "            num = int(re.findall(r'\\d+', importance[j][0])[0])",
    )
    compile(text, UPSTREAM_SOURCE["path"], "exec")
    return text


def prepare_upstream(*, output_dir: Path, archive_path: Path | None = None):
    """Download the immutable public archive; install only the active native module."""
    import io
    import urllib.request
    import zipfile

    output_dir.mkdir(parents=True, exist_ok=True)
    archive_raw = (
        archive_path.read_bytes() if archive_path is not None else
        urllib.request.urlopen(
            f"https://raw.githubusercontent.com/MingqingHuang-SHU/HSTXGB/{UPSTREAM_COMMIT}/HSTXGB.zip",
            timeout=60,
        ).read()
    )
    if hashlib.sha256(archive_raw).hexdigest() != UPSTREAM_ZIP_SHA256:
        raise ValueError("Unexpected HSTXGB source archive: pinned SHA-256 is required")
    with zipfile.ZipFile(io.BytesIO(archive_raw)) as archive:
        raw = archive.read(UPSTREAM_SOURCE["path"])
    patched = patched_source(raw).encode("utf-8")
    (output_dir / f"{UPSTREAM_SOURCE['module']}.original.py").write_bytes(raw)
    (output_dir / f"{UPSTREAM_SOURCE['module']}.py").write_bytes(patched)
    receipt = {
        "upstream_commit": UPSTREAM_COMMIT,
        "archive_sha256": UPSTREAM_ZIP_SHA256,
        "workflow": "expression_only",
        "upstream_path": UPSTREAM_SOURCE["path"],
        "original_source_sha256": hashlib.sha256(raw).hexdigest(),
        "patched_source_sha256": hashlib.sha256(patched).hexdigest(),
    }
    (output_dir / "hstxgb-source-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return receipt


if __name__ == "__main__" and sys.argv[1:2] == ["--prepare-upstream"]:
    parser = argparse.ArgumentParser(description="Prepare verified native source; no fits")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--archive-path", type=Path)
    arguments = parser.parse_args(sys.argv[2:])
    print(json.dumps(prepare_upstream(output_dir=arguments.output_dir, archive_path=arguments.archive_path), sort_keys=True))
    raise SystemExit(0)


import numpy as np

from _column_metadata import load_column_descriptors, summarize_temporal_metadata
from _run_tool_common import (
    load_execution_mode,
    load_params,
    validate_runtime_inputs,
    write_progress,
)

DEFAULTS = {"k": 2, "alpha": 0.49, "iter_num": 500, "seed": 42}
SEARCH_ITERATIONS = 20
CV_FOLDS = 3
SEARCH_SPACE = {
    "max_depth": {"distribution": "randint", "low": 5, "high_exclusive": 8},
    "min_child_weight": {"distribution": "randint", "low": 3, "high_exclusive": 4},
    "learning_rate": {"distribution": "uniform", "loc": 0.001, "scale": 0.001},
    "alpha": {"distribution": "uniform", "loc": 0.0, "scale": 0.2},
    "lambda": {"distribution": "uniform", "loc": 0.0, "scale": 0.0},
    "gamma": {"distribution": "uniform", "loc": 0.2, "scale": 0.1},
    "subsample": {"distribution": "uniform", "loc": 0.7, "scale": 0.0},
    "colsample_bytree": {"distribution": "uniform", "loc": 0.89, "scale": 0.02},
}


def parse_params(raw):
    if not isinstance(raw, dict):
        raise ValueError("HSTXGB parameters must be a JSON object")
    unknown = set(raw) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown HSTXGB parameters: {sorted(unknown)}")
    params = {**DEFAULTS, **raw}
    for name, minimum, maximum in (
        ("k", 0, None),
        ("iter_num", 1, None),
        ("seed", 0, 2**32 - 1),
    ):
        value = params[name]
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < minimum
            or maximum is not None
            and value > maximum
        ):
            raise ValueError(
                f"{name} must be an integer from {minimum}"
                + (f" to {maximum}" if maximum is not None else "")
            )
    value = params["alpha"]
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0 <= value <= 1
    ):
        raise ValueError("alpha must be a finite number from 0 to 1")
    return params


def read_matrix(path):
    with Path(path).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle, delimiter="\t"))
    if not rows or len(rows[0]) < 2 or len(rows) < 3:
        raise ValueError(
            "HSTXGB requires a genes-by-observations TSV with at least two genes"
        )
    header, columns = rows[0], rows[0][1:]
    if any(not item or item != item.strip() for item in header) or len(
        set(header)
    ) != len(header):
        raise ValueError(
            "Expression headings must be unique exact nonempty identifiers"
        )
    genes, values = [], []
    for number, row in enumerate(rows[1:], 2):
        if not row:
            continue
        if len(row) != len(header):
            raise ValueError(f"Malformed expression row {number}")
        gene = row[0]
        if not gene or gene != gene.strip() or gene in genes:
            raise ValueError(
                "Expression genes must be unique exact nonempty identifiers"
            )
        genes.append(gene)
        values.append([float(value) for value in row[1:]])
    expression = np.asarray(values, dtype=np.float64)
    if len(genes) < 2 or not np.isfinite(expression).all():
        raise ValueError("HSTXGB requires at least two genes and finite expression")
    return genes, columns, expression


def _read_regulators(path, genes):
    if not path.is_file():
        return list(genes)
    regulators = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if (
        not regulators
        or len(set(regulators)) != len(regulators)
        or not set(regulators) <= set(genes)
    ):
        raise ValueError("tf_list must contain unique exact expression gene IDs")
    if len(regulators) < 2:
        raise ValueError(
            "HSTXGB needs at least two candidate regulators because each target excludes itself"
        )
    return regulators


def prepare_data(expression_path, extra_dir, params):
    """Resolve exact column roles and independent regular series without averaging."""
    params = parse_params(params)
    extra_dir = Path(extra_dir)
    genes, columns, expression = read_matrix(expression_path)
    descriptors = load_column_descriptors(extra_dir / "column_descriptors.tsv", columns)
    temporal = summarize_temporal_metadata(descriptors)
    if temporal["sampling_design"] != "longitudinal":
        raise ValueError("HSTXGB requires longitudinal observed bulk time series")
    groups, observations, stationary = {}, set(), []
    for position, column in enumerate(columns):
        row = descriptors[column]
        if row["observation_type"] == "steady_state":
            stationary.append(column)
            continue
        if row["observation_id"] in observations:
            raise ValueError(
                "HSTXGB does not implicitly average time-series technical replicates"
            )
        observations.add(row["observation_id"])
        groups.setdefault(row["series_id"], []).append(
            (row["timepoint"], position, row)
        )
    blocks, series, expected_length, common_interval = [], [], None, None
    for series_id, entries in sorted(groups.items()):
        entries.sort(key=lambda value: value[0])
        times = [value[0] for value in entries]
        conditions = {value[2]["condition"] for value in entries}
        units = {value[2]["replicate"] for value in entries}
        if len(conditions) != 1 or len(units) != 1:
            raise ValueError(
                "One series must identify one condition and biological replicate"
            )
        if len(times) < 2 or len(times) <= params["k"]:
            raise ValueError("Each series needs at least two observed timepoints and more than k")
        intervals = np.diff(times)
        interval = float(intervals[0])
        if (
            not math.isfinite(interval)
            or interval <= 0
            or any(
                not math.isfinite(float(value))
                or value <= 0
                or not math.isclose(float(value), interval, rel_tol=1e-9, abs_tol=1e-12)
                for value in intervals
            )
        ):
            raise ValueError(
                "HSTXGB requires strictly increasing regular observed times"
            )
        if expected_length is not None and len(times) != expected_length:
            raise ValueError(
                "HSTXGB requires the same number of timepoints in every series"
            )
        if common_interval is not None and not math.isclose(
            interval, common_interval, rel_tol=1e-9, abs_tol=1e-12
        ):
            raise ValueError("HSTXGB requires a common sampling interval across series")
        expected_length, common_interval = len(times), interval
        positions = [value[1] for value in entries]
        blocks.append(expression[:, positions].T.copy())
        series.append(
            {
                "series_id": series_id,
                "condition": next(iter(conditions)),
                "replicate": next(iter(units)),
                "columns": [columns[index] for index in positions],
                "timepoints": times,
            }
        )
    if not blocks:
        raise ValueError("HSTXGB requires at least one time series")
    tf_path = extra_dir / "tf_list.txt"
    regulators = _read_regulators(tf_path, genes)
    native_genes = regulators + [gene for gene in genes if gene not in regulators]
    order = [genes.index(gene) for gene in native_genes]
    blocks = [part[:, order].copy() for part in blocks]

    # Validate the same arithmetic domains the unchanged native transforms use.
    # These matrices are validation receipts, not replacements for native code.
    k, alpha = params["k"], params["alpha"]
    transformed, labels = [], []
    with np.errstate(over="ignore", invalid="ignore"):
        for block in blocks:
            predictor = 0
            for lag in range(k + 1):
                predictor = (
                    predictor
                    + pow(alpha, k - lag) * block[lag : expected_length - k + lag]
                )
            transformed.append(predictor)
            labels.append(predictor)
        predictors = np.concatenate(transformed)
        targets = np.concatenate(labels)
        finite_native = all(
            np.isfinite(value).all() and np.isfinite(value.astype(np.float32)).all()
            for value in (predictors, targets)
        )
    if not finite_native:
        raise ValueError(
            "Transformed expression must remain finite at native XGBoost float32 precision"
        )
    if len(predictors) < CV_FOLDS:
        raise ValueError("HSTXGB requires at least three accumulated rows for native cross-validation")
    return {
        "genes": genes,
        "native_genes": native_genes,
        "columns": columns,
        "expression": expression,
        "time_series": np.concatenate(blocks),
        "expression_only_data": np.stack([part.T for part in blocks]),
        "regulators": regulators,
        "descriptors": descriptors,
        "temporal": temporal,
        "series": series,
        "unused_steady_state_columns": stationary,
        "timepoints_per_series": expected_length,
        "sampling_interval": common_interval,
        "accumulation_offsets": list(range(k + 1)),
        "accumulation_durations": [value * common_interval for value in range(k + 1)],
        "effective_training_rows": len(predictors),
        "transformed_predictors": predictors,
        "transformed_targets": targets,
    }


def _write_json(path, payload):
    Path(path).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_matrix(path, values, rows, columns):
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["row", *columns])
        writer.writerows(
            [label, *(repr(float(value)) for value in row)]
            for label, row in zip(rows, values)
        )


def export_network(path, matrix, source_genes, target_genes):
    values = np.asarray(matrix, dtype=np.float64)
    if values.shape != (len(source_genes), len(target_genes)):
        raise ValueError(
            "Native HSTXGB matrix dimensions do not match its gene identifiers"
        )
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("Native HSTXGB confidence must be finite and nonnegative")
    rows = [
        (source, target, float(values[i, j]))
        for i, source in enumerate(source_genes)
        for j, target in enumerate(target_genes)
        if source != target and values[i, j] > 0
    ]
    rows.sort(key=lambda row: (-row[2], row[0], row[1]))
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["source", "target", "score", "sign", "evidence", "context"])
        writer.writerows(
            (source, target, repr(score), "?", "association", "global")
            for source, target, score in rows
        )
    return len(rows)


def _resource_snapshot(model, model_params):
    from threadpoolctl import threadpool_info

    result = {
        "pid": os.getpid(),
        "model_params": model_params,
        "native_config": json.loads(model.save_config()),
        "observed_tree_count": model.num_boosted_rounds(),
        "native_pools": threadpool_info(),
        "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "cgroups": {},
    }
    for name in ("cpu.max", "memory.max", "cpu.stat", "memory.current", "memory.peak"):
        path = Path("/sys/fs/cgroup") / name
        if path.is_file():
            result["cgroups"][name] = path.read_text().strip()
    return result


def _session(implementation):
    session = {
        "upstream_commit": UPSTREAM_COMMIT,
        "workflow": "expression_only",
        "packages": {},
        "patches": "Generalized independent-course concatenation; feature-index scalar API; resource/seed controls and completed-search/fit observation.",
    }
    for package in ("numpy", "scipy", "scikit-learn", "xgboost-cpu", "joblib", "threadpoolctl"):
        try:
            session["packages"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            session["packages"][package] = "unavailable in recording test environment"
    source_path = getattr(implementation, "__file__", None)
    if source_path:
        session["patched_source_sha256"] = hashlib.sha256(Path(source_path).read_bytes()).hexdigest()
    path = Path("/app/hstxgb-source-receipt.json")
    if path.is_file():
        session["source_receipt"] = json.loads(path.read_text())
    return session


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


def run(*, input_path, params_path, extra_dir, output_dir, threads, upstream=None):
    input_path, params_path, extra_dir, output_dir = map(
        Path, (input_path, params_path, extra_dir, output_dir)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = output_dir / "provenance"
    raw.mkdir(exist_ok=True)
    completed, total, last_percent = 0, None, 0

    def progress(*, phase, message, percent, status="running", error=None):
        nonlocal last_percent
        last_percent = max(last_percent, percent)
        write_progress(
            output_dir / "progress.json",
            status=status,
            phase=phase,
            message=message,
            percent=last_percent,
            completed=completed,
            total=total,
            error=error,
        )
        event = json.loads((output_dir / "progress.json").read_text())
        event["observed_monotonic_ns"] = time.monotonic_ns()
        with (raw / "progress-events.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")

    progress(phase="validate", message="Validating HSTXGB temporal inputs", percent=0)
    try:
        validate_runtime_inputs(
            input_path=input_path,
            params_path=params_path,
            extra_dir=extra_dir,
            threads=threads,
        )
        load_execution_mode(params_path, supported_modes={"global"})
        params = parse_params(load_params(params_path))
        data = prepare_data(input_path, extra_dir, params)
        total = len(data["native_genes"])
        observed_columns = [
            column for row in data["series"] for column in row["columns"]
        ]
        _write_matrix(
            raw / "native-timeseries.tsv",
            data["time_series"],
            observed_columns,
            data["native_genes"],
        )
        effective_ids = [
            f"{row['series_id']}:{index}"
            for row in data["series"]
            for index in range(params["k"], data["timepoints_per_series"])
        ]
        for key, filename in (
            ("transformed_predictors", "transformed-predictors.tsv"),
            ("transformed_targets", "transformed-targets.tsv"),
        ):
            _write_matrix(
                raw / filename, data[key], effective_ids, data["native_genes"]
            )
        input_paths = [input_path, params_path, extra_dir / "column_descriptors.tsv"]
        input_paths.extend(
            path for path in (extra_dir / "tf_list.txt",) if path.is_file()
        )
        resolved = {
            "params": params,
            "search": {
                "iterations": SEARCH_ITERATIONS, "cv_folds": CV_FOLDS,
                "scoring": "neg_mean_squared_error", "shuffle": False,
                "distributions": SEARCH_SPACE, "search_and_cv_seed": params["seed"],
                "final_model_seed": 0, "search_n_jobs": threads,
                "cv_model_n_jobs": 1, "pre_dispatch": threads,
                "refit_best_estimator": True, "separate_final_train": True,
            },
            "genes": data["genes"],
            "native_genes": data["native_genes"],
            "regulators": data["regulators"],
            "temporal": data["temporal"],
            "series": data["series"],
            "column_descriptors": data["descriptors"],
            "unused_steady_state_columns": data["unused_steady_state_columns"],
            "samples_argument": data["timepoints_per_series"],
            "independent_series_count": len(data["series"]),
            "timepoints_per_series": data["timepoints_per_series"],
            "sampling_interval": data["sampling_interval"],
            "accumulation_offsets": data["accumulation_offsets"],
            "accumulation_durations": data["accumulation_durations"],
            "accumulation_reference": "Backward sample offsets from each accumulated window's endpoint, including the contemporaneous observation. These describe preprocessing, not inferred edge-specific delays.",
            "effective_training_rows": data["effective_training_rows"],
            "candidate_regulator_count": len(data["regulators"]),
            "effective_predictor_columns": {
                gene: len(data["regulators"]) - int(gene in data["regulators"])
                for gene in data["native_genes"]
            },
            "threads": threads,
            "native_nthread_argument": threads,
            "input_sha256": {
                path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in input_paths
            },
            "normalization": "Native simultaneous accumulated-expression regression and population-1 row variance multiplier; no prior or L2 normalization.",
        }
        _write_json(raw / "resolved.json", resolved)
        # Loky children inherit this before loading native libraries. Each CV
        # worker owns one CPU; final training receives its thread count directly.
        os.environ["OMP_NUM_THREADS"] = "1"
        implementation = (
            upstream
            if upstream is not None
            else importlib.import_module(UPSTREAM_SOURCE["module"])
        )
        _write_json(raw / "session.json", _session(implementation))
        progress(
            phase="inference", message="Fitting pinned HSTXGB target models", percent=5
        )

        searched = set()

        def searched_target(target_index, search):
            if target_index != completed or target_index in searched:
                raise ValueError("Native HSTXGB returned inconsistent completed-search observations")
            results = _json_value(search.cv_results_)
            if len(results.get("params", [])) != SEARCH_ITERATIONS:
                raise ValueError("Native HSTXGB search candidate count differs from the fixed panel")
            scores = results.get("mean_test_score", [])
            if len(scores) != SEARCH_ITERATIONS or any(value is None or not math.isfinite(value) for value in scores):
                raise ValueError("Native HSTXGB cross-validation produced non-finite candidate scores")
            searched.add(target_index)
            _write_json(raw / f"search-{target_index:08d}.json", {
                "target_gene": data["native_genes"][target_index],
                "best_index": int(search.best_index_),
                "best_params": _json_value(search.best_params_),
                "best_score": float(search.best_score_),
                "cv_results": results,
                "cv_folds": CV_FOLDS,
                "completed_cv_fits": SEARCH_ITERATIONS * CV_FOLDS,
                "best_estimator_refitted": True,
            })

        def fitted(count, target_index, model, model_params):
            nonlocal completed
            if count != completed + 1 or target_index != completed or count > total:
                raise ValueError(
                    "Native HSTXGB returned inconsistent completed-fit observations"
                )
            if target_index not in searched:
                raise ValueError("Native HSTXGB final fit preceded its completed search")
            completed = count
            snapshot = _resource_snapshot(model, model_params)
            snapshot.update(
                {
                    "target_gene": data["native_genes"][target_index],
                    "completed_fits": completed,
                    "effective_predictor_columns": resolved[
                        "effective_predictor_columns"
                    ][data["native_genes"][target_index]],
                    "predictor_count_basis": "Candidate regulator prefix with the current target excluded when present; derived from resolved input geometry.",
                }
            )
            _write_json(raw / f"model-{target_index:08d}.json", snapshot)
            if snapshot["observed_tree_count"] != params["iter_num"]:
                raise ValueError("Native HSTXGB tree count differs from iter_num")
            progress(
                phase="inference",
                message=f"Fitted target model for {data['native_genes'][target_index]}",
                percent=5 + 80 * completed // total,
            )

        def phase_changed(phase, target_index=None):
            if phase == "aggregation":
                if completed != total:
                    raise ValueError("Native HSTXGB aggregation began before all target fits completed")
                progress(phase="aggregation", message="Combining native HSTXGB confidences", percent=90)
            elif phase in {"search", "final_fit"}:
                if target_index != completed or (phase == "final_fit" and target_index not in searched):
                    raise ValueError("Native HSTXGB returned an inconsistent target phase")
                progress(
                    phase="hyperparameter_search" if phase == "search" else "final_fit",
                    message=("Searching hyperparameters for " if phase == "search" else "Fitting selected parameters for ") + data["native_genes"][target_index],
                    percent=5 + 80 * completed // total,
                )
            else:
                raise ValueError("Native HSTXGB returned an unknown phase")

        from threadpoolctl import threadpool_limits

        controls = {
            "andrea_seed": params["seed"],
            "andrea_threads": threads,
            "andrea_fit_callback": fitted,
            "andrea_search_callback": searched_target,
            "andrea_phase_callback": phase_changed,
        }
        # Cross-validation workers fit serial native models. Final training uses
        # the assigned thread count; unrelated BLAS pools stay serial.
        with (
            threadpool_limits(limits=1, user_api="blas"),
            threadpool_limits(limits=threads, user_api="openmp"),
            np.errstate(over="raise", invalid="raise", divide="raise"),
        ):
            matrix = implementation.main(
                data["expression_only_data"],
                len(data["native_genes"]),
                data["timepoints_per_series"],
                params["k"],
                params["alpha"],
                len(data["regulators"]),
                params["iter_num"],
                **controls,
            )
        if completed != total:
            raise ValueError(
                "HSTXGB returned before every native target fit was observed"
            )
        progress(
            phase="export", message="Exporting native HSTXGB confidence", percent=95
        )
        matrix = np.asarray(matrix, dtype=np.float64)
        if matrix.shape != (len(data["regulators"]), len(data["native_genes"])):
            raise ValueError(
                "Native HSTXGB returned an unexpected confidence matrix shape"
            )
        _write_matrix(
            raw / "native-score-matrix.tsv",
            matrix,
            data["regulators"],
            data["native_genes"],
        )
        resolved["network_edge_count"] = export_network(
            output_dir / "network.csv", matrix, data["regulators"], data["native_genes"]
        )
        _write_json(raw / "resolved.json", resolved)
        progress(
            status="completed",
            phase="done",
            message="HSTXGB network exported",
            percent=100,
        )
        return resolved
    except Exception as error:
        progress(
            status="failed",
            phase="failed",
            message="HSTXGB failed",
            percent=last_percent,
            error=str(error),
        )
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--params", required=True, type=Path)
    parser.add_argument("--extra", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--threads", required=True, type=int)
    arguments = parser.parse_args()
    run(
        input_path=arguments.input,
        params_path=arguments.params,
        extra_dir=arguments.extra,
        output_dir=arguments.output_dir,
        threads=arguments.threads,
    )


if __name__ == "__main__":
    main()
