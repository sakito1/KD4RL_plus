import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FINAL_TRAINING_SCRIPT = (
    ROOT / "train_sh" / "run_end_to_end_hrl_controller_joint_nas49_sh90.sh"
)
SWEEP_SCRIPT = ROOT / "scripts" / "run_multi_market_multi_k_seed_sweep.sh"


def test_final_training_script_forwards_k_and_filters_market(tmp_path):
    env = os.environ.copy()
    env.pop("REPRODUCE_BEST_MODE", None)
    env.update(
        {
            "PYTHON_BIN": "/bin/echo",
            "OUTPUT_ROOT": str(tmp_path / "outputs"),
            "RUN_NAME": "compat",
            "MARKETS": "nas",
            "NAS_SEEDS": "42 43",
            "SH_SEEDS": "90",
            "TRADE_NUM": "5",
            "CUDA_VISIBLE_DEVICES": "0",
        }
    )

    result = subprocess.run(
        ["bash", str(FINAL_TRAINING_SCRIPT)],
        cwd=tmp_path,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=30,
    )

    assert result.returncode == 0, result.stdout
    assert "--markets nas --seeds 42 43" in result.stdout
    assert "--trade_num 5" in result.stdout
    assert "--markets sh" not in result.stdout


def test_launcher_dry_run_round_robins_configs_across_available_gpus(tmp_path):
    env = os.environ.copy()
    env.update(
        {
            "DRY_RUN": "1",
            "GPU_IDS": "2 5",
            "MARKETS": "nas sh",
            "K_VALUES": "5 15",
            "NAS_SEEDS": "42 43",
            "SH_SEEDS": "90 91",
            "OUTPUT_ROOT": str(tmp_path / "outputs"),
        }
    )

    result = subprocess.run(
        ["bash", str(SWEEP_SCRIPT)],
        cwd=tmp_path,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=30,
    )

    assert result.returncode == 0, result.stdout
    assert "job=0 gpu=2 market=nas k=5 seeds=42 43" in result.stdout
    assert "job=1 gpu=5 market=nas k=15 seeds=42 43" in result.stdout
    assert "job=2 gpu=2 market=sh k=5 seeds=90 91" in result.stdout
    assert "job=3 gpu=5 market=sh k=15 seeds=90 91" in result.stdout
    assert "gpu_workers=2 configurations=4 total_seed_runs=8" in result.stdout


def test_launcher_default_matrix_contains_sixty_seed_runs(tmp_path):
    env = os.environ.copy()
    for key in ("GPU_IDS", "MARKETS", "K_VALUES", "NAS_SEEDS", "SH_SEEDS"):
        env.pop(key, None)
    env.update(
        {
            "DRY_RUN": "1",
            "OUTPUT_ROOT": str(tmp_path / "outputs"),
        }
    )

    result = subprocess.run(
        ["bash", str(SWEEP_SCRIPT)],
        cwd=tmp_path,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=30,
    )

    assert result.returncode == 0, result.stdout
    assert "gpu_workers=1 configurations=4 total_seed_runs=60" in result.stdout
    assert (
        "market=nas k=5 seeds=42 43 44 45 46 47 48 49 50 51 52 53 54 55 56"
        in result.stdout
    )
    assert (
        "market=sh k=15 seeds=83 84 85 86 87 88 89 90 91 92 93 94 95 96 97"
        in result.stdout
    )
