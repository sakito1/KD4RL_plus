import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SINGLE_SCRIPT = (
    ROOT / "train_sh" / "train_sh_full_cmtflow_risk10_adv0_seed.sh"
)
SCHEDULER_SCRIPT = (
    ROOT / "train_sh" / "run_sh_full_cmtflow_risk10_adv0_4gpu.sh"
)


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
        ["/bin/bash", str(script)],
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=30,
    )


def test_single_seed_command_trains_all_modules_with_requested_guidance_rule():
    result = run_bash(
        SINGLE_SCRIPT,
        {
            "SEED": "54",
            "GPU_ID": "3",
            "OUTPUT_ROOT": "results/test_sh_risk10_adv0",
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
    assert "--controller_guidance_risk_threshold 0.10" in output
    assert (
        "--controller_guidance_risk_min_advantage_threshold 0.00"
        in output
    )
    assert "--controller_guidance_advantage_threshold 0.10" in output
    assert "--frozen_hrl_checkpoint" not in output
    assert "--controller_only_finetune" not in output
    assert "--no_train_controller" not in output
    assert "--skip_test" not in output


def test_scheduler_runs_four_requested_sh_seeds_on_four_gpus():
    result = run_bash(
        SCHEDULER_SCRIPT,
        {
            "OUTPUT_ROOT": "results/test_sh_risk10_adv0",
            "GPU0": "4",
            "GPU1": "5",
            "GPU2": "6",
            "GPU3": "7",
        },
    )

    assert result.returncode == 0, result.stdout
    output = result.stdout
    for seed, gpu in zip(("44", "46", "49", "54"), ("4", "5", "6", "7")):
        assert output.count(f"GPU {gpu} starting SH seed={seed};") == 1
        assert (
            f"--run_name sh_full_42135_risk10_adv0_or_adv10_seed{seed}"
            in output
        )
    assert output.count("lane 0 queue:") == 4
    assert "lane 1 queue:" not in output
    assert "Concurrent jobs per GPU: 1" in output
    assert "SH seeds: 44 46 49 54" in output
    assert "--markets nas" not in output
