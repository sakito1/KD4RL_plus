import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SINGLE_SCRIPT = ROOT / "train_sh" / "train_full_cmtflow_seed.sh"
SCHEDULER_SCRIPT = ROOT / "train_sh" / "run_full_cmtflow_seed_sweep_4gpu.sh"


def run_bash(script: Path, extra_env=None):
    env = os.environ.copy()
    env.update(
        {
            "DRY_RUN": "1",
            "PYTHON_BIN": "/bin/echo",
        }
    )
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        ["bash", str(script)],
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=30,
    )


def test_single_job_runs_full_42135_schedule_and_test_from_scratch():
    result = run_bash(
        SINGLE_SCRIPT,
        {
            "MARKET": "sh",
            "SEED": "54",
            "GPU_ID": "3",
            "OUTPUT_ROOT": "results/test_full_cmtflow",
        },
    )

    assert result.returncode == 0, result.stdout
    output = result.stdout
    assert "--markets sh" in output
    assert "--seeds 54" in output
    assert "--warmup_outer_epochs 4" in output
    assert "--warmup_inner_epochs 2" in output
    assert "--joint_epochs 1" in output
    assert "--joint_single_full_episode" in output
    assert "--controller_sup_pretrain_epochs 3" in output
    assert "--controller_aux_replay_epochs 30" in output
    assert "--controller_epochs 5" in output
    assert "--controller_guidance_pretrain_coef 1.0" in output
    assert "--controller_sup_coef 0.01" in output
    assert "--controller_aux_mdd_coef 0.01" in output
    assert "--controller_aux_switch_adv_coef 0.01" in output
    assert "--controller_switch_rate_penalty_coef 0.01" in output
    assert "--frozen_hrl_checkpoint" not in output
    assert "--controller_only_finetune" not in output
    assert "--end_to_end_controller_joint" not in output
    assert "--controller_joint_epochs" not in output
    assert "--no_train_controller" not in output
    assert "--skip_test" not in output


def test_scheduler_assigns_default_seeds_to_one_lane_per_gpu():
    result = run_bash(
        SCHEDULER_SCRIPT,
        {
            "OUTPUT_ROOT": "results/test_full_cmtflow",
            "GPU0": "0",
            "GPU1": "1",
            "GPU2": "2",
            "GPU3": "3",
        },
    )

    assert result.returncode == 0, result.stdout
    output = result.stdout
    expected = {
        "nas": ["44", "45", "47", "50", "56", "57", "58"],
        "sh": ["44", "46", "49", "54"],
    }
    for market, seeds in expected.items():
        for seed in seeds:
            assert output.count(f"starting market={market} seed={seed};") == 1

    assert output.count("lane 0 queue:") == 4
    assert "lane 1 queue:" not in output
    assert "Concurrent jobs per GPU: 1" in output
    assert "Schedule: Outer 4 -> Inner 2 -> Outer+Inner joint 1" in output
    assert "--skip_test" not in output


def test_scheduler_uses_python_from_active_path_when_not_overridden(tmp_path):
    fake_python = tmp_path / "python"
    fake_python.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    fake_python.chmod(0o755)

    env = os.environ.copy()
    env.pop("PYTHON_BIN", None)
    env.update(
        {
            "DRY_RUN": "1",
            "PATH": f"{tmp_path}:/usr/bin:/bin",
            "NAS_SEEDS": "44",
            "SH_SEEDS": "",
            "OUTPUT_ROOT": "results/test_full_cmtflow",
        }
    )
    result = subprocess.run(
        ["/bin/bash", str(SCHEDULER_SCRIPT)],
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=30,
    )

    assert result.returncode == 0, result.stdout
    assert str(fake_python) in result.stdout
    assert "/home/tongwenxuan/conda/envs/xuangu/bin/python" not in result.stdout
