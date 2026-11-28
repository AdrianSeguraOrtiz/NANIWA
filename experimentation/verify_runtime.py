"""Verify the external runtime bundle or the installed software for one benchmark stage."""
import argparse
import hashlib
import json
import platform
import subprocess
from importlib import metadata, util
from pathlib import Path


HERE = Path(__file__).resolve().parent


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256")


def package_fingerprint(package):
    """Hash Python sources and analysis catalogs, excluding bytecode and build metadata."""
    spec = util.find_spec(package)
    if spec is None or spec.origin is None:
        raise ValueError(f"Run with the Python installation containing {package}")
    root = Path(spec.origin).parent
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root)
        if path.suffix not in {".py", ".json"}:
            continue
        if package == "naniwa" and rel.parts[0] == "docker":
            continue
        if package == "andrea" and path.suffix == ".json" and not rel.parts[0].startswith("catalog_"):
            continue
        digest.update(rel.as_posix().encode() + b"\0" + sha256(path).digest())
    return digest.hexdigest()


def check(name, value, expected):
    if value != expected:
        raise ValueError(f"{name}: expected {expected}, found {value}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("archives", "osaka", "naniwa"))
    parser.add_argument("--bundle", type=Path, help="Directory with the three external archives.")
    parser.add_argument("--print-constraints", action="store_true", help="Print the selected Python environment's pinned requirements.")
    args = parser.parse_args()
    lock = json.loads((HERE / "runtime.lock.json").read_text())
    try:
        if args.print_constraints:
            if args.stage == "archives":
                parser.error("Choose osaka or naniwa to print Python package constraints.")
            for name, expected in lock[args.stage]["packages"].items():
                print(f"{name}=={expected}")
            return
        if args.stage == "archives":
            if args.bundle is None:
                parser.error("archives requires --bundle")
            for name, expected in lock["archives_sha256"].items():
                check(name, sha256(args.bundle / name).hexdigest(), expected)
                print(f"Verified {name}")
            return

        entry = lock[args.stage]
        check("Python", platform.python_version(), lock["python"])
        engine, engine_platform = subprocess.check_output(
            ["docker", "version", "--format", "{{.Server.Version}} {{.Server.Os}}/{{.Server.Arch}}"],
            text=True,
        ).split()
        check("Docker server", engine, lock["docker_server"])
        check("Docker platform", engine_platform, lock["platform"])
        check(entry["package"], package_fingerprint(entry["package"]), entry["code_sha256"])
        for name, expected in entry["packages"].items():
            check(name, metadata.version(name), expected)
        if args.stage == "osaka":
            recipes = json.loads((HERE / "simulators.json").read_text())["simulators"]
            images = {rule["image"]: rule["image_id"] for rule in recipes.values()}
        else:
            check("Public ANDREA sources", package_fingerprint("andrea"), entry["inference_andrea_code_sha256"])
            images = entry["images"]
        for tag, expected in images.items():
            actual = subprocess.check_output(["docker", "image", "inspect", "--format", "{{.Id}}", tag], text=True).strip()
            check(tag, actual, expected)
        print(f"Verified {args.stage}: Python, package sources, dependencies, Docker server/platform and {len(images)} images")
    except (OSError, ValueError, metadata.PackageNotFoundError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
