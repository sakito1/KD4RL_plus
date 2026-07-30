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

