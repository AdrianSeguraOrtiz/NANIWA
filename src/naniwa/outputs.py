"""Result directories and consensus CSV output."""
import csv
import json
import os
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from .inputs import write_merged_networks


def output_directory(path):
    """Accept a new or empty directory, preserving existing results."""
    path = Path(path)
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise ValueError(f"Choose a new or empty output directory: {path}")
    return path


def write_consensus(destination, dataset, scores, tools, raw=None, *, options):
    """Publish all result files together once the complete ranking is written."""
    targets, regulators = np.nonzero(dataset.mask)
    values = scores[targets, regulators]
    sources = np.asarray(dataset.regulators)[regulators]
    destinations = np.asarray(dataset.genes)[targets]
    order = np.lexsort((destinations, sources, -values))

    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=destination.parent) as tmp:
        result = Path(tmp) / "result"
        result.mkdir()
        with (result / "consensus.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(("source", "target", "score"))
            writer.writerows((sources[i], destinations[i], float(values[i])) for i in order)
        settings = {"naniwa_version": version("naniwa"), "options": options.parameters()}
        (result / "consensus_options.json").write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
        if raw is not None:
            write_merged_networks(result / "networks.csv", dataset, tools, raw)
        os.replace(result, destination)
    return destination / "consensus.csv"
