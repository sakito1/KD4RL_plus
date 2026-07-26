import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "train_sh" / "explore_controller_from_nas45_outer_inner.sh"


def run_script() -> str:
    env = os.environ.copy()
    env.update(
        {
            "DRY_RUN": "1",
            "PYTHON_BIN": "/bin/echo",
            "OUTPUT_ROOT": "results/test_nas45_controller_exploration",
        }
    )
    result = subprocess.run(
        ["bash", str(SCRIPT)],
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout
    return result.stdout


def test_script_emits_one_controller_sampled_sup_pg_configuration():
    output = run_script()

    assert "--markets nas" in output
    assert "--seeds 45" in output
    assert "--frozen_hrl_checkpoint" in output
    assert "outer_inner_4_3_2_k5/nas/ppo/seed_45/checkpoints/hrl_fixed_best.pth" in output
    assert "--controller_only_finetune" in output
    assert "--trade_num 5" in output
    assert "--warmup_outer_epochs 0" in output
    assert "--warmup_inner_epochs 0" in output
    assert "--joint_epochs 0" in output
    assert "--controller_rollout_len 300" in output
    assert "--controller_sup_pretrain_rollout_len 300" in output
    assert "--controller_windows_per_epoch 12" in output
    assert "--controller_pg_batch_windows 12" in output
    assert "--controller_train_max_hold 30" in output
    assert "--controller_eval_max_hold 30" in output
    assert "--controller_sup_pretrain_epochs 1" in output
    assert "--controller_aux_replay_epochs 30" in output
    assert "--controller_epochs 3" in output
    assert "--controller_use_switch_supervision" in output
    assert "--controller_sup_coef 0.1" in output
    assert "--controller_aux_mdd_coef 0.1" in output
    assert "--controller_aux_switch_adv_coef 0.1" in output
    assert "--controller_aux_mdd_target_scale 20.0" in output
    assert "--controller_aux_switch_adv_target_scale 20.0" in output
    assert "--controller_aux_switch_adv_loss_type mse" in output
    assert "--controller_switch_rate_min 0.05" in output
    assert "--controller_switch_rate_max 0.15" in output
    assert "--controller_aux_pretrain_offpolicy" not in output
    assert "--controller_pretrain_only" not in output
    assert "--controller_guidance_probe_only" not in output


def test_script_has_no_obsolete_training_modes():
    source = SCRIPT.read_text(encoding="utf-8")

    assert "MODE=" not in source
    assert 'case "$MODE"' not in source
    assert "probe" not in source
    assert "pg_only" not in source
    assert "sup_only" not in source
