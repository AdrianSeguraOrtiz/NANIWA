"""Adapt standalone bulk time series to ANDREA's dataset manifest."""
import csv
import json
import re
from contextlib import contextmanager
from pathlib import Path
from shutil import copyfile
from tempfile import TemporaryDirectory

from .docker.shared._column_metadata import (
    TEMPORAL_COLUMNS, load_column_descriptors, summarize_temporal_metadata,
)
from .inputs import _expression, _records, _regulators, dataset_source


_COMPACT_COLUMNS = {"column", "series_id", "timepoint", "time_unit", "sampling_design"}


@contextmanager
def inference_manifest(dataset, expression=None, samples=None, regulators=None):
    """Use the supplied manifest or keep a standalone snapshot alive for the operation."""
    source = dataset_source(dataset, expression, samples, regulators)
    if isinstance(source, Path):
        if not source.is_file():
            raise ValueError(f"Dataset manifest not found: {source}")
        yield source.resolve()
    else:
        with TemporaryDirectory(prefix="naniwa-dataset-") as directory:
            yield write_manifest(Path(directory), *source)


def _prepare_samples(path, destination, columns):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        headings = set(next(csv.reader(stream, delimiter="\t", strict=True), ()))
    compact = headings == _COMPACT_COLUMNS
    if compact:
        with destination.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, TEMPORAL_COLUMNS, delimiter="\t", lineterminator="\n")
            writer.writeheader()
            for _, row in _records(path, _COMPACT_COLUMNS, "\t"):
                writer.writerow(dict(row, observation_type="time_series", condition="default",
                                     replicate=row["series_id"], observation_id=row["column"],
                                     technical_replicate="1"))
    elif set(TEMPORAL_COLUMNS).issubset(headings):
        copyfile(path, destination)
    else:
        raise ValueError(f"{path}: for inference, --samples needs exactly column, series_id, timepoint, "
                         "time_unit and sampling_design, or the full ANDREA temporal descriptors")

    try:
        metadata = load_column_descriptors(destination, columns)
        temporal = summarize_temporal_metadata(metadata)
    except ValueError as error:
        raise ValueError(f"{path}: {error}") from error
    if temporal["sampling_design"] != "longitudinal":
        raise ValueError(f"{path}: NANIWA's inference tools require longitudinal sampling_design")
    if compact and len({(row["series_id"], row["timepoint"]) for row in metadata.values()}) != len(metadata):
        raise ValueError(f"{path}: compact samples require a unique timepoint within each series")


def write_manifest(directory, expression, samples, regulators) -> Path:
    """Snapshot three files in a private directory and return the ANDREA manifest."""
    directory = Path(directory)
    matrix = directory / "expression.tsv"
    descriptors = directory / "column_descriptors.tsv"
    tf_list = directory / "tf_list.txt"
    copyfile(expression, matrix)
    genes, columns, _ = _expression(matrix, load_values=False, label=expression)
    tf_list.write_text("\n".join(_regulators(regulators, genes)) + "\n", encoding="utf-8")
    _prepare_samples(Path(samples), descriptors, columns)

    name = Path(expression).stem or "dataset"
    dataset_id = re.sub(r"[^a-z0-9_-]+", "-", name.lower()).lstrip("-_") or "dataset"
    spec = {
        "schema_version": "1.0", "id": dataset_id, "name": name,
        "expression": {"genes": len(genes), "columns": len(columns),
                       "column_kind": "timepoints", "expression_profile": "bulk"},
        "organism": {"taxonomic_group": "unknown", "ncbi_taxon_id": None},
    }
    manifest = {
        "schema_version": "1.0", "id": dataset_id,
        "dataset": {"spec": spec, "expression_matrix": matrix.name},
        "extras": {"column_descriptors": descriptors.name, "tf_list": tf_list.name},
    }
    path = directory / "dataset.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path
