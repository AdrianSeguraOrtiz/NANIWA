"""Content identities of the expression, metadata and references in a registered benchmark."""
import hashlib
import json


def inventory_hashes(out, rows):
    """Bind each canonical file digest to its view ID before aggregating across the bank."""
    digests = {}
    for row in sorted(rows, key=lambda r: r["id"]):
        manifest = out / row["dataset_manifest"]
        truth_manifest = out / row["ground_truth_manifest"]
        data = json.loads(manifest.read_text(encoding="utf-8"))
        truth = json.loads(truth_manifest.read_text(encoding="utf-8"))
        files = {
            "expression": manifest.parent / data["dataset"]["expression_matrix"],
            "descriptors": manifest.parent / data["extras"]["column_descriptors"],
            "regulators": manifest.parent / data["extras"]["tf_list"],
            "genes": truth_manifest.parent / truth["candidate_space"]["targets"],
            "truth_network": truth_manifest.parent / truth["outputs"]["networks"],
        }
        for key, path in files.items():
            if key not in digests:
                digests[key] = hashlib.sha256()
            with path.open("rb") as stream:
                value = hashlib.file_digest(stream, "sha256").hexdigest()
            digests[key].update(f"{row['id']}\t{value}\n".encode("utf-8"))
    return {"views": len(rows), "sha256": {key: value.hexdigest() for key, value in digests.items()}}
