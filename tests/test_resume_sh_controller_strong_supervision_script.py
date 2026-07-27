import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "train_sh" / "resume_sh_controller_44_46_strong_sup.sh"


def run_dry_run(extra_env=None):
    env = os.environ.copy()
    env.update(
        {
            "DRY_RUN": "1",
            "PYTHON_BIN": "/bin/echo",
            "SOURCE_CHECKPOINT_44": "results/source/sh44/hrl_fixed_best.pth",
            "SOURCE_CHECKPOINT_46": "results/source/sh46/hrl_fixed_best.pth",
        }
    )
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        ["bash", str(SCRIPT)],
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=30,
    )


def test_dry_run_builds_two_frozen_controller_only_jobs():
    result = run_dry_run()

    assert result.returncode == 0, result.stdout
    output = result.stdout
    assert output.count("--markets sh") == 2
    assert output.count("--controller_only_finetune") == 2
    assert output.count("--frozen_hrl_checkpoint") == 2
    assert "results/source/sh44/hrl_fixed_best.pth" in output
    assert "results/source/sh46/hrl_fixed_best.pth" in output
    assert "--seeds 44" in output
    assert "--seeds 46" in output
    assert output.count("--warmup_outer_epochs 0") == 2
    assert output.count("--warmup_inner_epochs 0") == 2
    assert output.count("--joint_epochs 0") == 2
    assert output.count("--max_hold 30") == 2
    assert output.count("--controller_train_max_hold 30") == 2
    assert output.count("--controller_eval_max_hold 60") == 2
    assert output.count("--controller_sup_coef 0.10") == 2
    assert output.count("--controller_guidance_pretrain_coef 1.0") == 2
    assert output.count("--controller_guidance_risk_threshold 0.10") == 2
    assert (
        output.count(
            "--controller_guidance_risk_min_advantage_threshold 0.02"
        )
        == 2
    )
    assert output.count("--controller_guidance_advantage_threshold 0.10") == 2
    assert output.count("--controller_aux_mdd_coef 0.01") == 2
    assert output.count("--controller_aux_switch_adv_coef 0.01") == 2
    assert "--skip_test" not in output
    assert "seed=44 GPU=0" in output
    assert "seed=46 GPU=1" in output


def test_training_knobs_can_be_overridden_without_changing_pretraining_scale():
    result = run_dry_run(
        {
            "CONTROLLER_EPOCHS": "7",
            "CONTROLLER_SUP_COEF": "0.20",
            "GPU_44": "2",
            "GPU_46": "3",
        }
    )

    assert result.returncode == 0, result.stdout
    output = result.stdout
    assert output.count("--controller_epochs 7") == 2
    assert output.count("--controller_sup_coef 0.20") == 2
    assert output.count("--controller_guidance_pretrain_coef 1.0") == 2
    assert "seed=44 GPU=2" in output
    assert "seed=46 GPU=3" in output
