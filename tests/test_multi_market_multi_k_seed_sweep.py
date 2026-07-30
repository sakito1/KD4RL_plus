import os
import subprocess
import time
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
FINAL_TRAINING_SCRIPT = (
    ROOT / "train_sh" / "run_end_to_end_hrl_controller_joint_nas49_sh90.sh"
)
SWEEP_SCRIPT = ROOT / "scripts" / "run_multi_market_multi_k_seed_sweep.sh"
WORKER_SCRIPT = ROOT / "scripts" / "run_multi_market_multi_k_seed_worker.sh"


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
    assert f"Repository root: {ROOT}" in result.stdout
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


@pytest.mark.parametrize(
    ("variable", "value", "message"),
    [
        ("GPU_IDS", "0 0", "GPU_IDS contains a duplicate value: 0"),
        ("MARKETS", "nas nas", "MARKETS contains a duplicate value: nas"),
        ("K_VALUES", "5 5", "K_VALUES contains a duplicate value: 5"),
    ],
)
def test_launcher_rejects_duplicate_scheduling_values(
    tmp_path, variable, value, message
):
    env = os.environ.copy()
    env.update(
        {
            "DRY_RUN": "1",
            "GPU_IDS": "0",
            "MARKETS": "nas sh",
            "K_VALUES": "5 15",
            "NAS_SEEDS": "42",
            "SH_SEEDS": "90",
            "OUTPUT_ROOT": str(tmp_path / "outputs"),
            variable: value,
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

    assert result.returncode == 2, result.stdout
    assert message in result.stdout


def test_launcher_forces_sweep_mode_during_real_worker_execution(tmp_path):
    env = os.environ.copy()
    env.update(
        {
            "DRY_RUN": "0",
            "REPRODUCE_BEST_MODE": "1",
            "PYTHON_BIN": "/bin/echo",
            "GPU_IDS": "0",
            "MARKETS": "nas",
            "K_VALUES": "5",
            "NAS_SEEDS": "42",
            "SH_SEEDS": "90",
            "OUTPUT_ROOT": str(tmp_path / "outputs"),
            "RUN_PREFIX": "mode_guard",
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
    assert f"Repository root: {ROOT}" in result.stdout
    assert "--markets nas --seeds 42" in result.stdout
    assert "--trade_num 5" in result.stdout
    assert "Full reproduction run" not in result.stdout
    assert (tmp_path / "outputs" / "launcher_logs" / "gpu_0.log").is_file()


def test_launcher_propagates_training_failure(tmp_path):
    fake_python = tmp_path / "fake_python"
    fake_python.write_text(
        "#!/usr/bin/env bash\n"
        'if [[ " $* " == *" --trade_num 15 "* ]]; then exit 17; fi\n'
        "exit 0\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)

    env = os.environ.copy()
    env.update(
        {
            "DRY_RUN": "0",
            "REPRODUCE_BEST_MODE": "0",
            "PYTHON_BIN": str(fake_python),
            "GPU_IDS": "0",
            "MARKETS": "nas",
            "K_VALUES": "5 15",
            "NAS_SEEDS": "42",
            "SH_SEEDS": "90",
            "OUTPUT_ROOT": str(tmp_path / "outputs"),
            "RUN_PREFIX": "failure_guard",
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

    assert result.returncode == 1, result.stdout
    assert "Failed GPU workers: gpu_0" in result.stdout


def test_sweep_chain_contains_no_unverified_legacy_repository_path():
    for script in (SWEEP_SCRIPT, WORKER_SCRIPT, FINAL_TRAINING_SCRIPT):
        assert script.is_file()
        assert "/home/tongwenxuan/KD4RL_plus" not in script.read_text(
            encoding="utf-8"
        )


def test_launcher_termination_stops_descendant_training_process(tmp_path):
    pid_file = tmp_path / "fake_python.pid"
    fake_python = tmp_path / "long_running_python"
    fake_python.write_text(
        "#!/usr/bin/env bash\n"
        'echo "$$" > "$SWEEP_TEST_PID_FILE"\n'
        "trap 'exit 0' INT TERM\n"
        "while true; do sleep 1; done\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)

    env = os.environ.copy()
    env.update(
        {
            "DRY_RUN": "0",
            "PYTHON_BIN": str(fake_python),
            "SWEEP_TEST_PID_FILE": str(pid_file),
            "GPU_IDS": "0",
            "MARKETS": "nas",
            "K_VALUES": "5",
            "NAS_SEEDS": "42",
            "SH_SEEDS": "90",
            "OUTPUT_ROOT": str(tmp_path / "outputs"),
            "RUN_PREFIX": "termination_guard",
        }
    )

    process = subprocess.Popen(
        ["bash", str(SWEEP_SCRIPT)],
        cwd=tmp_path,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    deadline = time.monotonic() + 10
    while not pid_file.exists() and time.monotonic() < deadline:
        time.sleep(0.02)

    try:
        assert pid_file.is_file()
        training_pid = int(pid_file.read_text(encoding="utf-8").strip())
        process.terminate()
        stdout, _ = process.communicate(timeout=10)
        assert process.returncode == 143, stdout

        deadline = time.monotonic() + 5
        while Path(f"/proc/{training_pid}").exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert not Path(f"/proc/{training_pid}").exists()
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
