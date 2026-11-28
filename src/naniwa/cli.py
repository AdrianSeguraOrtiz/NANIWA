"""Command-line entry point for NANIWA."""
import argparse
import csv
from importlib.metadata import version
from pathlib import Path

from .consensus import STAGES, ConsensusOptions
from .consensus.options import COMPONENT_CHOICES
from .inference import catalog, run_inference
from .workflow import consensus, run


def _individual_files(entries):
    if entries is None:
        return None
    files = {}
    for entry in entries:
        tool, separator, path = entry.partition("=")
        if not separator or not tool.strip() or not path.strip():
            raise ValueError("--network needs TOOL=FILE, for example --network swing=swing.csv")
        if tool in files:
            raise ValueError(f"Duplicate tool ID in --network: {tool!r}")
        files[tool] = Path(path)
    return files


def _scoring_parser():
    parent = argparse.ArgumentParser(add_help=False)
    group = parent.add_argument_group("consensus method")
    group.add_argument("--stage", choices=STAGES,
                       help="Start from a preset and optionally override its components. Default: naniwa.")
    group.add_argument("--vote", choices=COMPONENT_CHOICES["vote"],
                       help="Give equal votes to all inputs, distinct networks or predictive families. "
                            "A vote supplied alone returns its base score.")
    group.add_argument("--evidence", choices=COMPONENT_CHOICES["evidence"],
                       help="Use the base vote, marginal temporal support, or joint reassignment among leaders. "
                            "Enabling evidence on a base vote selects global intensity by default.")
    group.add_argument("--intensity", choices=COMPONENT_CHOICES["intensity"],
                       help="Disable the temporal update, estimate one intensity across targets, or use intensity one. "
                            "Enabling intensity on a base vote selects joint evidence by default.")
    return parent


def main(argv=None):
    parser = argparse.ArgumentParser(prog="naniwa", description="Combine inferred gene regulatory networks.")
    parser.add_argument("--version", action="version", version=f"%(prog)s {version('naniwa')}")
    commands = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    data = common.add_argument_group("time-series dataset")
    source = data.add_mutually_exclusive_group(required=True)
    source.add_argument("--dataset", type=Path, help="ANDREA dataset manifest (JSON).")
    source.add_argument("--expression", type=Path, help="Gene-by-sample expression TSV. Use with --samples and --regulators.")
    data.add_argument("--samples", type=Path, help="Sample TSV: column, series_id, timepoint. Inference also needs time_unit and sampling_design.")
    data.add_argument("--regulators", type=Path, help="Regulator gene IDs, one per line.")
    common.add_argument("--out", type=Path, required=True, help="New or empty result directory.")
    inference = argparse.ArgumentParser(add_help=False)
    inference.add_argument("--tools", nargs="+", choices=tuple(catalog()), help="Tool IDs. Default: all seven.")
    inference.add_argument("--andrea", default="andrea", help="ANDREA executable (default: andrea on PATH).")
    inference.add_argument("--max-cores", type=int, help="CPU budget for ANDREA's planner.")
    inference.add_argument("--max-ram-gb", type=float, help="Memory budget in GB for ANDREA's planner.")
    scoring = _scoring_parser()
    commands.add_parser("infer", parents=[common, inference], help="Run Docker inference tools through ANDREA.")
    commands.add_parser("run", parents=[common, inference, scoring],
                        help="Infer networks with ANDREA and score their consensus.")
    command = commands.add_parser("consensus", parents=[common, scoring],
                                  help="Score interactions using source networks and time series.")
    networks = command.add_mutually_exclusive_group(required=True)
    networks.add_argument("--networks", type=Path, help="Joint raw, unscaled scores with a tool_id column (ANDREA: merged_network_raw.csv).")
    networks.add_argument("--network", action="append", metavar="TOOL=FILE", help="Individual CSV, repeat per tool.")
    arguments = vars(parser.parse_args(argv))
    action = arguments.pop("command")
    try:
        if action in ("run", "consensus"):
            selection = {key: arguments.pop(key) for key in ("stage", *COMPONENT_CHOICES)}
            arguments["options"] = ConsensusOptions.resolve(**selection)
        if action == "consensus":
            arguments["individual"] = _individual_files(arguments.pop("network"))
            consensus(**arguments)
        elif action == "run":
            run(**arguments)
        else:
            print(run_inference(**arguments))
    except (ValueError, OSError, csv.Error) as error:
        parser.exit(2, f"naniwa: {error}\n")
