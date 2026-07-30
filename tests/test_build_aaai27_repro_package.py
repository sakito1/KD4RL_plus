import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from tools.build_aaai27_repro_package import (
    SOURCE_FILES,
    TRACE_FILES,
    build_package,
)


def write_fixture_file(path: Path, content: bytes = b"fixture\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def fixture_repository(tmp_path: Path) -> tuple[Path, dict[str, Path]]:
    source = tmp_path / "source"
    source.mkdir()
    for relative in SOURCE_FILES:
        write_fixture_file(source / relative)
    for source_path in TRACE_FILES:
        write_fixture_file(source / source_path)
    write_fixture_file(source / "utils/NAS100_pool.txt", b"A\nB\n")
    write_fixture_file(source / "utils/SH_pool.txt", b"X\nY\n")

    nas_run = tmp_path / "nas_run/nas/ppo/seed_49"
    nas_frozen = tmp_path / "nas_frozen/nas/ppo/seed_49"
    sh_run = tmp_path / "sh_run/sh/ppo/seed_90"
    for name in ("best_model.pth", "controller_best.pth"):
        write_fixture_file(nas_run / "checkpoints" / name, name.encode())
    write_fixture_file(
        nas_frozen / "checkpoints/hrl_fixed_best.pth",
        b"nas-frozen-hrl",
    )
    write_fixture_file(
        nas_run.parents[1] / "seed_49_command.json",
        b'{"seed": 49}\n',
    )
    write_fixture_file(
        nas_frozen.parents[1] / "seed_49_command.json",
        b'{"seed": 49}\n',
    )

    for name in ("best_model.pth", "controller_best.pth", "hrl_fixed_best.pth"):
        write_fixture_file(sh_run / "checkpoints" / name, name.encode())
    write_fixture_file(
        sh_run.parents[1] / "seed_90_command.json",
        b'{"seed": 90}\n',
    )
    return source, {
        "nas_run": nas_run,
        "nas_frozen_run": nas_frozen,
        "sh_run": sh_run,
    }


def test_build_package_creates_paper_first_tree_without_symlinks(
    tmp_path: Path,
) -> None:
    source, artifacts = fixture_repository(tmp_path)
    destination = tmp_path / "package"

    build_package(source, destination, artifact_roots=artifacts)

    assert (destination / "README.md").is_file()
    assert (destination / "EXPECTED_RESULTS.md").is_file()
    assert (destination / "PACKAGE_STATUS.md").is_file()
    assert (destination / "MODEL_PROVENANCE.md").is_file()
    assert (destination / "src/run_hrl_training.py").is_file()
    assert (
        destination / "src/paper_experiments/render_aaai27_figure3.py"
    ).is_file()
    assert (destination / "configs/aaai27_figure3_cases.json").is_file()
    assert (destination / "expected/table1.csv").is_file()
    assert (destination / "expected/table2.csv").is_file()
    assert (
        destination / "checkpoints/nas_seed49/checkpoints/best_model.pth"
    ).read_bytes() == b"best_model.pth"
    assert (
        destination
        / "checkpoints/nas_seed49/checkpoints/original_hrl_fixed_best.pth"
    ).read_bytes() == b"nas-frozen-hrl"
    assert (
        destination / "checkpoints/sh_seed90/checkpoints/best_model.pth"
    ).read_bytes() == b"best_model.pth"
    model_manifest = json.loads(
        (destination / "checkpoints/MODEL_MANIFEST.json").read_text(
            encoding="utf-8"
        )
    )
    nas_final = next(
        entry
        for entry in model_manifest["files"]
        if entry["path"].endswith("nas_seed49/checkpoints/best_model.pth")
    )
    assert nas_final["market"] == "Nasdaq-100"
    assert nas_final["seed"] == 49
    assert nas_final["role"] == "paper_final_checkpoint"
    assert not any(path.is_symlink() for path in destination.rglob("*"))


def test_manifest_hashes_every_packaged_file_with_relative_paths(
    tmp_path: Path,
) -> None:
    source, artifacts = fixture_repository(tmp_path)
    destination = tmp_path / "package"

    build_package(source, destination, artifact_roots=artifacts)

    manifest = json.loads(
        (destination / "MANIFEST.json").read_text(encoding="utf-8")
    )
    manifest_paths = {entry["path"] for entry in manifest["files"]}
    regular_files = {
        path.relative_to(destination).as_posix()
        for path in destination.rglob("*")
        if path.is_file() and path.name != "MANIFEST.json"
    }
    assert manifest_paths == regular_files
    assert all(not Path(path).is_absolute() for path in manifest_paths)
    for entry in manifest["files"]:
        payload = (destination / entry["path"]).read_bytes()
        assert entry["sha256"] == hashlib.sha256(payload).hexdigest()


def test_expected_tables_match_pdf_values(tmp_path: Path) -> None:
    source, artifacts = fixture_repository(tmp_path)
    destination = tmp_path / "package"

    build_package(source, destination, artifact_roots=artifacts)

    table1 = pd.read_csv(destination / "expected/table1.csv")
    csi = table1.loc[
        (table1["market"] == "CSI-300") & (table1["method"] == "CMTFlow")
    ].iloc[0]
    nas = table1.loc[
        (table1["market"] == "Nasdaq-100") & (table1["method"] == "CMTFlow")
    ].iloc[0]
    assert csi["total_return_pct"] == pytest.approx(237.01)
    assert nas["total_return_pct"] == pytest.approx(262.49)


def test_builder_refuses_nonempty_destination(tmp_path: Path) -> None:
    source, artifacts = fixture_repository(tmp_path)
    destination = tmp_path / "package"
    destination.mkdir()
    (destination / "keep.txt").write_text("user data", encoding="utf-8")

    with pytest.raises(FileExistsError, match="not empty"):
        build_package(source, destination, artifact_roots=artifacts)


def test_builder_removes_author_home_paths_from_packaged_text(
    tmp_path: Path,
) -> None:
    source, artifacts = fixture_repository(tmp_path)
    (source / "run_hrl_training.py").write_text(
        'python="/home/tongwenxuan/conda/envs/xuangu/bin/python"\n'
        'legacy="/home/tongwenxuan/KD4RL"\n',
        encoding="utf-8",
    )
    destination = tmp_path / "package"

    build_package(source, destination, artifact_roots=artifacts)

    packaged = (destination / "src/run_hrl_training.py").read_text(
        encoding="utf-8"
    )
    assert "/home/tongwenxuan" not in packaged
