import json
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "train_sh" / "test_full_cmtflow_controller_maxhold60.sh"


def _write_run(source_root: Path, seed: int, *, completed: bool) -> Path:
    market_root = source_root / f"nas_full_42135_seed{seed}" / "nas"
    market_root.mkdir(parents=True)
    command = [
        "/fake/python",
        "-u",
        str(ROOT / "run_hrl_training.py"),
        "--child",
        "--market",
        "nas",
        "--seed",
        str(seed),
        "--run_root",
        str(market_root.parent),
        "--max_hold",
        "30",
        "--controller_eval_max_hold",
        "30",
    ]
    (market_root / f"seed_{seed}_command.json").write_text(
        json.dumps({"command": command}),
        encoding="utf-8",
    )
    checkpoint = market_root / "ppo" / f"seed_{seed}" / "checkpoints" / "best_model.pth"
    if completed:
        checkpoint.parent.mkdir(parents=True)
        checkpoint.touch()
    return checkpoint


def test_dry_run_replays_only_completed_runs_with_controller_maxhold60(tmp_path):
    assert SCRIPT.is_file(), f"missing script: {SCRIPT}"

    source_root = tmp_path / "source"
    output_root = tmp_path / "output"
    completed_checkpoint = _write_run(source_root, 44, completed=True)
    _write_run(source_root, 45, completed=False)

    env = os.environ.copy()
    env.update(
        {
            "SOURCE_ROOT": str(source_root),
            "OUTPUT_ROOT": str(output_root),
            "PYTHON_BIN": "/runtime/python",
            "GPU_ID": "2",
            "DRY_RUN": "1",
        }
    )
    result = subprocess.run(
        ["bash", str(SCRIPT)],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--max_hold 30" in result.stdout
    assert "--controller_eval_max_hold 60" in result.stdout
    assert f"--test_only_checkpoint {completed_checkpoint}" in result.stdout
    assert "/runtime/python" in result.stdout
    assert "seed=44" in result.stdout
    assert "SKIP seed=45" in result.stdout
    assert "completed=1 skipped=1 failed=0" in result.stdout


def test_dry_run_distributes_completed_runs_over_four_single_job_gpu_queues(tmp_path):
    source_root = tmp_path / "source"
    output_root = tmp_path / "output"
    for seed in range(44, 49):
        _write_run(source_root, seed, completed=True)

    env = os.environ.copy()
    env.update(
        {
            "SOURCE_ROOT": str(source_root),
            "OUTPUT_ROOT": str(output_root),
            "PYTHON_BIN": "/runtime/python",
            "GPU_IDS": "0 1 2 3",
            "DRY_RUN": "1",
        }
    )
    result = subprocess.run(
        ["bash", str(SCRIPT)],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    for gpu in range(4):
        assert f"GPU {gpu} queue:" in result.stdout
        assert f"CUDA_VISIBLE_DEVICES={gpu}" in result.stdout
    assert "completed=5 skipped=0 failed=0" in result.stdout
