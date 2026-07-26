import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SINGLE_SCRIPT = ROOT / "train_sh" / "train_controller_from_outer_inner.sh"
SCHEDULER_SCRIPT = ROOT / "train_sh" / "run_controller_seed_sweep_3gpu.sh"


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


def test_single_job_uses_same_market_seed_checkpoint_and_runs_test():
    result = run_bash(
        SINGLE_SCRIPT,
        {
            "MARKET": "sh",
            "CONTROLLER_SEED": "54",
            "GPU_ID": "2",
            "SOURCE_ROOT": "results/source_models",
            "OUTPUT_ROOT": "results/test_controller_sweep",
        },
    )

    assert result.returncode == 0, result.stdout
    output = result.stdout
    assert "--markets sh" in output
    assert "--seeds 54" in output
    assert (
        "results/source_models/sh/ppo/seed_54/checkpoints/hrl_fixed_best.pth"
        in output
    )
    assert "--controller_only_finetune" in output
    assert "--controller_guidance_pretrain_coef 1.0" in output
    assert "--controller_sup_pretrain_epochs 4" in output
    assert "--controller_aux_replay_epochs 30" in output
    assert "--controller_epochs 5" in output
    assert "--controller_sup_coef 0.01" in output
    assert "--controller_aux_mdd_coef 0.01" in output
    assert "--controller_aux_switch_adv_coef 0.01" in output
    assert "--controller_switch_rate_penalty_coef 0.01" in output
    assert "--skip_test" not in output


def test_scheduler_assigns_default_seed_jobs_to_two_lanes_per_gpu():
    result = run_bash(
        SCHEDULER_SCRIPT,
        {
            "SOURCE_ROOT": "results/source_models",
            "OUTPUT_ROOT": "results/test_controller_sweep",
            "GPU0": "0",
            "GPU1": "1",
            "GPU2": "2",
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
            assert f"market={market} seed={seed}" in output
            assert (
                f"results/source_models/{market}/ppo/seed_{seed}/checkpoints/"
                "hrl_fixed_best.pth"
            ) in output

    assert output.count("lane 0 queue:") == 3
    assert output.count("lane 1 queue:") == 3
    assert "Concurrent jobs per GPU: 2" in output
    assert "--skip_test" not in output
