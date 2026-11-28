"""Run the native GreyNet image once per trajectory and emit one global network."""

import argparse
import csv
import math
import subprocess
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory


FIELDS = ("source", "target", "score", "sign", "evidence", "context")


def partition_inputs(expression, descriptors, work):
    # Reuse the exact metadata checks shipped in the GreyNet base image.
    from _column_metadata import load_column_descriptors, summarize_temporal_metadata

    with expression.open(encoding="utf-8", newline="") as stream, ExitStack() as files:
        reader = csv.reader(stream, delimiter="\t")
        header = next(reader, [])
        metadata = load_column_descriptors(descriptors, header[1:])
        summarize_temporal_metadata(metadata)
        series = {}
        for index, column in enumerate(header[1:], start=1):
            row = metadata[column]
            if row["observation_type"] == "time_series":
                series.setdefault(row["series_id"], []).append(index)

        # Keep descriptor strings, expression values and column order unchanged.
        with descriptors.open(encoding="utf-8", newline="") as file:
            table = csv.DictReader(file, delimiter="\t")
            raw = {row["column"]: row for row in table}
            fields = table.fieldnames
        outputs = []
        children = []
        for number, label in enumerate(sorted(series), start=1):
            child = work / str(number)
            children.append(child)
            (child / "extra").mkdir(parents=True)
            columns = series[label]
            with (child / "extra/column_descriptors.tsv").open(
                "w", encoding="utf-8", newline=""
            ) as file:
                writer = csv.DictWriter(file, fieldnames=fields, delimiter="\t", lineterminator="\n")
                writer.writeheader()
                writer.writerows(raw[header[index]] for index in columns)
            file = files.enter_context((child / "expression.tsv").open("w", encoding="utf-8", newline=""))
            writer = csv.writer(file, delimiter="\t", lineterminator="\n")
            indices = [0, *columns]
            writer.writerow([header[index] for index in indices])
            outputs.append((writer, indices))

        for number, row in enumerate(reader, start=2):
            if not row:
                continue
            if len(row) != len(header):
                raise ValueError(f"Malformed expression row {number}")
            for writer, indices in outputs:
                writer.writerow([row[index] for index in indices])
    return children


def aggregate(networks, destination):
    """Average the union of unsigned edges, giving each trajectory equal weight."""
    totals = {}
    for path in networks:
        with path.open(encoding="utf-8", newline="") as file:
            reader = csv.DictReader(file)
            if not set(FIELDS).issubset(reader.fieldnames or ()):
                raise ValueError(f"Missing GreyNet network columns in {path}")
            for row in reader:
                score = float(row["score"])
                if row["sign"] != "?" or row["context"] != "global":
                    raise ValueError("GreyNet must emit unsigned edges in the global context")
                if not math.isfinite(score) or score <= 0:
                    raise ValueError("GreyNet must emit finite positive edge scores")
                pair = row["source"], row["target"]
                totals[pair] = totals.get(pair, 0.0) + score

    with destination.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(FIELDS)
        for (source, target), value in totals.items():
            score = value / len(networks)
            if score > 0:
                magnitude = repr(score).removesuffix(".0")
                writer.writerow((source, target, magnitude, "?", "andrea_series_agg_mean_effect", "global"))


def run(expression, params, extra, output, threads):
    output.mkdir(parents=True, exist_ok=True)
    network = output / "network.csv"
    network.unlink(missing_ok=True)
    with TemporaryDirectory(dir=output, prefix=".greynet-") as temporary:
        work = Path(temporary)
        children = partition_inputs(expression, extra / "column_descriptors.tsv", work)
        networks = []
        for child in children:
            subprocess.run([
                "/app/run_tool.sh", "--input", str(child / "expression.tsv"),
                "--params", str(params), "--extra", str(child / "extra"),
                "--output-dir", str(child / "out"), "--threads", str(threads),
            ], check=True)
            networks.append(child / "out/network.csv")
        staged = work / "network.csv"
        aggregate(networks, staged)
        staged.replace(network)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--params", required=True, type=Path)
    parser.add_argument("--extra", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--threads", required=True, type=int)
    args = parser.parse_args()
    run(args.input, args.params, args.extra, args.output_dir, args.threads)
