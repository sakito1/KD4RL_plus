import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "train_sh" / "run_outer_inner_seed_sweep_3gpu.sh"


def test_three_gpu_outer_inner_seed_sweep_dry_run():
    env = os.environ.copy()
    env.update({
        "DRY_RUN": "1",
        "PYTHON_BIN": "/bin/echo",
        "NAS_SEEDS": "11 12 13 14",
        "SH_SEEDS": "21 22",
    })

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
    assert "GPU 0 queue: nas:11 nas:14" in result.stdout
    assert "GPU 1 queue: nas:12 sh:21" in result.stdout
    assert "GPU 2 queue: nas:13 sh:22" in result.stdout
    assert "--markets nas --seeds 11" in result.stdout
    assert "--markets sh --seeds 21" in result.stdout
    assert "--warmup_outer_epochs 4" in result.stdout
    assert "--warmup_inner_epochs 3" in result.stdout
    assert "--joint_epochs 2" in result.stdout
    assert "--trade_num 5" in result.stdout
    assert "--no_train_controller" in result.stdout
