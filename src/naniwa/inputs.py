"""Read ANDREA datasets and align raw source networks on the candidate edges."""
from __future__ import annotations

import csv
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Dataset:
    genes: tuple[str, ...]
    regulators: tuple[str, ...]
    X: np.ndarray  # series × time × gene
    times: np.ndarray  # series × time
    mask: np.ndarray  # target gene × regulator


def _check_ids(ids, label):
    if not ids or any(not name.strip() for name in ids):
        raise ValueError(f"Missing or blank {label}")
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate {label}")


def _records(path, required, delimiter):
    """Validate column names and row widths before interpreting any fields."""
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, delimiter=delimiter, strict=True)
        fields = reader.fieldnames or []
        _check_ids(fields, f"column names in {path}")
        missing = set(required).difference(fields)
        if missing:
            raise ValueError(f"{path}: missing columns {', '.join(sorted(missing))}")
        for row in reader:
            if None in row or None in row.values():
                raise ValueError(f"{path}, line {reader.line_num}: incorrect number of fields")
            yield reader.line_num, row


def _expression(path, *, load_values=True, label=None):
    """Validate each row, retaining the numeric matrix only when scoring needs it."""
    if label is None:
        label = path
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        rows = csv.reader(stream, delimiter="\t", strict=True)
        header = next(rows, [])
        columns = header[1:]
        _check_ids(columns, f"sample IDs in {label}")
        genes = []
        values = [] if load_values else None
        for row in rows:
            if len(row) != len(header):
                raise ValueError(f"{label}, line {rows.line_num}: incorrect number of expression values")
            genes.append(row[0])
            try:
                measurements = np.asarray(row[1:], dtype=float)
            except ValueError as error:
                raise ValueError(f"{label}, line {rows.line_num}: invalid expression value") from error
            if not np.isfinite(measurements).all():
                raise ValueError(f"{label}, line {rows.line_num}: expression values must be finite")
            if values is not None:
                values.append(measurements)
    _check_ids(genes, f"gene IDs in {label}")
    return tuple(genes), columns, np.asarray(values) if load_values else None


def _sample_layout(path, columns):
    by_column = {}
    series_labels = {}
    observations = set()
    time_unit = None
    for line, row in _records(path, ("column", "series_id", "timepoint"), "\t"):
        column, series = row["column"], row["series_id"]
        if row.get("observation_type", "time_series") != "time_series":
            raise ValueError(f"{path}, line {line}: consensus requires time_series observations")
        if row.get("sampling_design", "longitudinal") != "longitudinal":
            raise ValueError(f"{path}, line {line}: consensus requires longitudinal sampling_design")
        if not column.strip() or not series.strip() or column in by_column:
            raise ValueError(f"{path}, line {line}: missing series or blank/duplicate sample ID")
        if "time_unit" in row:
            unit = row["time_unit"]
            if not unit.strip() or (time_unit is not None and unit != time_unit):
                raise ValueError(f"{path}, line {line}: use one nonempty time_unit across all samples")
            time_unit = unit
        labels = tuple(row[key] for key in ("condition", "replicate") if key in row)
        if any(not value.strip() for value in labels) or series_labels.setdefault(series, labels) != labels:
            raise ValueError(f"{path}, line {line}: each series_id must identify one condition and biological replicate")
        if "observation_id" in row:
            observation = row["observation_id"]
            if not observation.strip() or observation in observations:
                raise ValueError(f"{path}, line {line}: consensus requires one measurement per observation_id")
            observations.add(observation)
        try:
            time = float(row["timepoint"])
        except ValueError as error:
            raise ValueError(f"{path}, line {line}: invalid timepoint") from error
        if not math.isfinite(time):
            raise ValueError(f"{path}, line {line}: timepoint must be finite")
        by_column[column] = (series, time)
    if set(by_column) != set(columns):
        raise ValueError(f"{path}: sample IDs must match the expression columns exactly")
    groups = {}
    for i, column in enumerate(columns):
        series, time = by_column[column]
        groups.setdefault(series, []).append((time, i))
    times, positions = [], []
    for series, members in sorted(groups.items()):
        ordered = sorted(members)
        grid = [t for t, _ in ordered]
        if len(grid) < 2 or len(grid) != len(set(grid)):
            raise ValueError(f"{path}: series {series!r} needs at least two observations with unique timepoints")
        if times and len(grid) != len(times[0]):
            raise ValueError(f"{path}: all series must have the same number of timepoints")
        times.append(grid)
        positions.append([i for _, i in ordered])
    return np.asarray(times), positions


def dataset_source(manifest=None, expression=None, samples=None, regulators=None):
    """Select one manifest path or one complete set of three input files."""
    files = (expression, samples, regulators)
    if manifest is not None:
        if any(item is not None for item in files):
            raise ValueError("Use either a dataset manifest or the three individual dataset files")
        return Path(manifest)
    if any(item is None for item in files):
        raise ValueError("Provide a dataset manifest or --expression, --samples and --regulators together")
    return tuple(Path(item) for item in files)


def _regulators(path, genes):
    ids = tuple(Path(path).read_text(encoding="utf-8-sig").splitlines())
    _check_ids(ids, f"regulator IDs in {path}")
    unknown = set(ids).difference(genes)
    if unknown:
        raise ValueError(f"{path}: regulators missing from expression: {', '.join(sorted(unknown))}")
    return ids


def load_dataset(manifest=None, expression=None, samples=None, regulators=None) -> Dataset:
    """Read a dataset manifest, or the expression TSV, sample TSV and regulator list."""
    source = dataset_source(manifest, expression, samples, regulators)
    if isinstance(source, Path):
        with source.open(encoding="utf-8-sig") as stream:
            data = json.load(stream)
        try:
            files = (data["dataset"]["expression_matrix"], data["extras"]["column_descriptors"],
                     data["extras"]["tf_list"])
            if any(not isinstance(value, str) or not value.strip() for value in files):
                raise ValueError
        except (TypeError, KeyError, ValueError) as error:
            raise ValueError(f"{source}: expected dataset.expression_matrix, extras.column_descriptors "
                             "and extras.tf_list paths") from error
        source = tuple(source.parent / value for value in files)
    expression, samples, regulators = source
    genes, columns, matrix = _expression(expression)
    times, positions = _sample_layout(samples, columns)
    regulator_ids = _regulators(regulators, genes)
    mask = np.asarray(genes)[:, None] != np.asarray(regulator_ids)[None, :]
    if not mask.any():
        raise ValueError("The dataset has no eligible non-self interactions")
    X = matrix.T[positions]
    return Dataset(genes, regulator_ids, X, times, mask)


def network_rows(path, expected_tool=None):
    """Yield validated global edge scores from a joint or single-tool network CSV."""
    required = ("source", "target", "score")
    if expected_tool is None:
        required += ("tool_id",)
    for line, row in _records(path, required, ","):
        tool = row.get("tool_id", expected_tool)
        if not tool.strip() or (expected_tool is not None and tool != expected_tool):
            raise ValueError(f"{path}, line {line}: blank or inconsistent tool_id {tool!r}")
        if row.get("context", "global") != "global":
            raise ValueError(f"{path}, line {line}: expected a global network context")
        source, target = row["source"], row["target"]
        if not source.strip() or not target.strip():
            raise ValueError(f"{path}, line {line}: blank source or target")
        try:
            score = float(row["score"])
        except ValueError as error:
            raise ValueError(f"{path}, line {line}: invalid score") from error
        if not math.isfinite(score) or score < 0:
            raise ValueError(f"{path}, line {line}: scores must be finite and nonnegative")
        yield tool, source, target, score


def load_networks(dataset: Dataset, merged=None, individual: Mapping | None = None, *, require_variation=True):
    """Align raw networks, requiring a nonzero score range for min-max scaling."""
    if (merged is None) == (individual is None):
        raise ValueError("Provide either a joint CSV or individual network files")
    if individual is not None:
        _check_ids(tuple(individual), "tool IDs")
    targets = {gene: i for i, gene in enumerate(dataset.genes)}
    regulators = {gene: i for i, gene in enumerate(dataset.regulators)}
    arrays = {}
    observed = set()
    sources = [(None, merged)] if merged is not None else individual.items()
    for expected_tool, path in sources:
        if expected_tool is not None:
            arrays[expected_tool] = np.zeros(dataset.mask.shape)
        for tool, source, target, score in network_rows(path, expected_tool):
            if tool not in arrays:
                arrays[tool] = np.zeros(dataset.mask.shape)
            i, j = targets.get(target), regulators.get(source)
            if i is not None and j is not None and dataset.mask[i, j]:
                observed.add(tool)
                arrays[tool][i, j] = max(arrays[tool][i, j], score)
    if not arrays:
        raise ValueError("No source networks in the supplied files")
    tools = tuple(sorted(arrays))
    for tool in tools:
        if tool not in observed:
            raise ValueError(f"Tool {tool!r} has no eligible interaction in the supplied network")
        if require_variation and np.ptp(arrays[tool][dataset.mask]) == 0:
            raise ValueError(f"Tool {tool!r} has no score variation across eligible edges")
    scores = np.stack([arrays[tool] for tool in tools])
    scores[:, ~dataset.mask] = np.nan
    return tools, scores


def write_merged_networks(path, dataset: Dataset, tool_names: Sequence[str], raw):
    """Export eligible scores on the common input schema in a sparse, lossless form."""
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("source", "target", "score", "tool_id"))
        for tool, matrix in zip(tool_names, raw, strict=True):
            edges = np.argwhere(dataset.mask & (matrix > 0))
            if len(edges) == 0:
                # One explicit zero preserves the tool ID in the joint CSV.
                edges = [np.unravel_index(dataset.mask.argmax(), dataset.mask.shape)]
            for target, source in edges:
                writer.writerow((dataset.regulators[source], dataset.genes[target], matrix[target, source], tool))
