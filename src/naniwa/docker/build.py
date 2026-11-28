"""Build the selected inference images from the recipes included with NANIWA."""
import argparse
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory

from naniwa.inference import catalog


ROOT = Path(__file__).resolve().parent
SHARED_FILES = ("_run_tool_common.py", "_column_metadata.py", "run_tool.sh")


def prepare_base_context(tool, context):
    """Assemble the exact five files used in the ANDREA base image build."""
    for name in ("Dockerfile", "run_tool.py"):
        shutil.copyfile(ROOT / "base" / tool / name, context / name)
    for name in SHARED_FILES:
        shutil.copyfile(ROOT / "shared" / name, context / name)
    (context / "run_tool.sh").chmod(0o755)


def build_image(tag, context):
    print(f"Building {tag}", flush=True)
    subprocess.run(["docker", "build", "--platform", "linux/amd64", "--tag", tag, str(context)], check=True)


def main(argv=None):
    panel = catalog()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tools", nargs="*", metavar="TOOL",
                        help="Build selected tools (default all): " + ", ".join(panel))
    args = parser.parse_args(argv)
    selected = tuple(dict.fromkeys(args.tools or panel))
    unknown = set(selected).difference(panel)
    if unknown:
        parser.error("Unknown tool: " + ", ".join(sorted(unknown)))

    try:
        for tool in selected:
            tag = panel[tool]["docker_image"]
            with TemporaryDirectory() as tmp:
                context = Path(tmp)
                prepare_base_context(tool, context)
                if tool == "greynet":
                    build_image("adriansegura99/inference-tools_greynet:1.0.0", context)
                else:
                    build_image(tag, context)
            if tool == "greynet":
                build_image(tag, ROOT / "greynet")
    except (OSError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"Image build failed: {error}\n")


if __name__ == "__main__":
    main()
