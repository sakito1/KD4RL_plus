#!/usr/bin/env python3
import hashlib
import json
import re
from pathlib import Path

root = Path(__file__).resolve().parents[1]
manifest = json.loads((root / "MANIFEST.json").read_text(encoding="utf-8"))
failures = []
manifest_paths = {entry["path"] for entry in manifest["files"]}


def ignored(path: Path) -> bool:
    relative = path.relative_to(root)
    return (
        "__pycache__" in relative.parts
        or path.suffix == ".pyc"
        or (relative.parts and relative.parts[0] == "outputs")
        or relative.parts[:2] == ("appendix", "outputs")
    )


for entry in manifest["files"]:
    path = root / entry["path"]
    if not path.is_file() or path.is_symlink():
        failures.append(f"missing-or-symlink: {entry['path']}")
        continue
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != entry["sha256"]:
        failures.append(f"hash-mismatch: {entry['path']}")

for path in root.rglob("*"):
    if path.is_symlink() and not ignored(path):
        failures.append(f"unlisted-symlink: {path.relative_to(root).as_posix()}")
    elif path.is_file() and path.name != "MANIFEST.json" and not ignored(path):
        relative = path.relative_to(root).as_posix()
        if relative not in manifest_paths:
            failures.append(f"unlisted-file: {relative}")

seed_path_pattern = re.compile(r"(?:seed[_-]?\d+|nas49|sh90)", re.IGNORECASE)
for path in root.rglob("*"):
    if not ignored(path) and seed_path_pattern.search(path.name):
        failures.append(f"seed-exposed-in-public-path: {path.relative_to(root)}")

expected_models = {
    "checkpoints/nasdaq100/checkpoints/best_model.pth",
    "checkpoints/csi300/checkpoints/best_model.pth",
}
actual_models = {
    path.relative_to(root).as_posix()
    for path in root.rglob("*.pth")
    if not ignored(path)
}
if actual_models != expected_models:
    failures.append(
        "model-inventory: expected only "
        + ", ".join(sorted(expected_models))
        + "; found "
        + ", ".join(sorted(actual_models))
    )

model_manifest = json.loads(
    (root / "checkpoints/MODEL_MANIFEST.json").read_text(encoding="utf-8")
)
for entry in model_manifest["files"]:
    path = root / entry["path"]
    if not path.is_file() or path.is_symlink():
        failures.append(f"model-manifest-missing: {entry['path']}")
        continue
    if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
        failures.append(f"model-manifest-hash: {entry['path']}")

model_version = json.loads(
    (root / "appendix/MODEL_VERSION.json").read_text(encoding="utf-8")
)
if model_version["markets"]["nas"]["dataset"] != "NASDAQ-100":
    failures.append("appendix-model-identity: nas-dataset")
if model_version["markets"]["nas"]["seed"] != 49:
    failures.append("appendix-model-identity: nas-seed")
if model_version["markets"]["sh"]["dataset"] != "CSI-300":
    failures.append("appendix-model-identity: sh-dataset")
if model_version["markets"]["sh"]["seed"] != 90:
    failures.append("appendix-model-identity: sh-seed")
if model_version["training"]["transaction_cost_rate"] != 0.00005:
    failures.append("appendix-training-cost")
if model_version["evaluation"]["paper_cost_rate"] != 0.0001:
    failures.append("appendix-evaluation-cost")
if list((root / "appendix").rglob("*.pth")):
    failures.append("appendix-must-not-contain-model-files")

if failures:
    raise SystemExit("\n".join(failures))
print(f"verified {len(manifest['files'])} packaged files")
