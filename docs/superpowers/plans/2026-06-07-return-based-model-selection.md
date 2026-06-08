# Return-Based Model Selection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use $superpower-subagents (recommended) or $superpower-executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking via update_plan.

**Goal:** Add an opt-in training variant that selects HRL checkpoints by highest validation return and uses the discussed stronger inner/controller training schedule, without changing the current default training behavior or the existing count-band training script.

**Architecture:** Keep current training defaults unchanged. Add a small selection-metric abstraction in `Train/PPO_train.py`, expose it through `run_hrl_training.py`, and create a separate shell script for the return-selection experiment. The new script opts into return-based checkpoint selection and the discussed schedule changes: `warmup_inner_epochs=5`, `controller_epochs=10`, `controller_val_interval_epochs=1`, `controller_mdd_coef=5.0`, and `controller_return_coef=0.2`; existing defaults and the current script remain untouched.

**Tech Stack:** Python, PyTorch, argparse, unittest, Bash.

---

## File Structure

- Modify: `Train/PPO_train.py`
  - Add helper methods to score validation metrics by `sharpe`, `return`, or `controller_return`.
  - Use the helper for warmup outer, warmup inner, joint, and controller checkpoint selection.
  - Default behavior remains current: fixed HRL phases use Sharpe; controller uses the existing risk-return score.
- Modify: `run_hrl_training.py`
  - Add CLI flags for selection metrics.
  - Pass flags to child processes and runtime config.
  - Defaults keep existing behavior.
- Create: `tests/test_return_selection_metric.py`
  - Verify default metrics are unchanged.
  - Verify return-selection mode chooses `total_ret`.
  - Verify controller return-selection mode chooses validation `total_ret`.
- Create: `scripts/run_hrl_min10_max40_pg_countband_return_select.sh`
  - A separate script that runs the same NAS/SH seed split as the current script, but opt-in selects checkpoints by highest validation return.
  - It also applies the discussed training schedule: more inner warmup, more controller PG epochs, every-epoch controller validation, and risk-prioritized controller reward coefficients.

---

### Task 1: Add Tests For Selection Metric Behavior

**Files:**
- Create: `tests/test_return_selection_metric.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_return_selection_metric.py`:

```python
import os
import sys
import unittest
from types import SimpleNamespace


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from Train.PPO_train import HRL_Trainer


class ReturnSelectionMetricTests(unittest.TestCase):
    def test_fixed_hrl_default_selection_uses_sharpe(self):
        cfg = SimpleNamespace(model_selection_metric="sharpe")
        metrics = {"sharpe": 0.7, "total_ret": 0.2, "max_dd": 0.4}

        score = HRL_Trainer._validation_score(metrics, cfg, phase="joint")

        self.assertAlmostEqual(score, 0.7)

    def test_fixed_hrl_return_selection_uses_total_return(self):
        cfg = SimpleNamespace(model_selection_metric="return")
        metrics = {"sharpe": 0.7, "total_ret": 0.2, "max_dd": 0.4}

        score = HRL_Trainer._validation_score(metrics, cfg, phase="joint")

        self.assertAlmostEqual(score, 0.2)

    def test_controller_default_selection_keeps_existing_risk_return_score(self):
        cfg = SimpleNamespace(
            controller_selection_metric="risk_return",
            controller_mdd_coef=2.0,
            controller_return_coef=0.5,
        )
        metrics = {"sharpe": 0.7, "total_ret": 0.3, "max_dd": 0.2}

        score = HRL_Trainer._validation_score(metrics, cfg, phase="controller")

        self.assertAlmostEqual(score, -2.0 * 0.2 + 0.5 * 0.3)

    def test_controller_return_selection_uses_total_return(self):
        cfg = SimpleNamespace(controller_selection_metric="return")
        metrics = {"sharpe": 0.7, "total_ret": 0.3, "max_dd": 0.2}

        score = HRL_Trainer._validation_score(metrics, cfg, phase="controller")

        self.assertAlmostEqual(score, 0.3)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
cd /home/tongwenxuan/KD4RL_plus
/home/tongwenxuan/conda/envs/xuangu/bin/python tests/test_return_selection_metric.py
```

Expected: fail with `AttributeError: type object 'HRL_Trainer' has no attribute '_validation_score'`.

---

### Task 2: Implement Selection Score Helper

**Files:**
- Modify: `Train/PPO_train.py`

- [ ] **Step 1: Add `_validation_score` helper to `HRL_Trainer`**

Add this method inside `class HRL_Trainer`, near `_compute_metrics` or other evaluation helpers:

```python
    @staticmethod
    def _validation_score(metrics, cfg, phase: str = "joint") -> float:
        if phase == "controller":
            metric_name = str(getattr(cfg, "controller_selection_metric", "risk_return"))
        else:
            metric_name = str(getattr(cfg, "model_selection_metric", "sharpe"))

        if metric_name in ("return", "total_ret", "ret"):
            return float(metrics["total_ret"])
        if metric_name in ("mdd", "min_mdd"):
            return -float(metrics["max_dd"])
        if metric_name in ("risk_return", "controller_default"):
            mdd_coef = float(getattr(cfg, "controller_mdd_coef", 2.0))
            return_coef = float(getattr(cfg, "controller_return_coef", 0.5))
            return -mdd_coef * float(metrics["max_dd"]) + return_coef * float(metrics["total_ret"])
        if metric_name == "sharpe":
            return float(metrics["sharpe"])
        raise ValueError(f"Unknown validation selection metric: {metric_name}")
```

- [ ] **Step 2: Run the focused test**

Run:

```bash
cd /home/tongwenxuan/KD4RL_plus
/home/tongwenxuan/conda/envs/xuangu/bin/python tests/test_return_selection_metric.py
```

Expected: tests pass.

---

### Task 3: Use Selection Helper In Training Phases

**Files:**
- Modify: `Train/PPO_train.py`

- [ ] **Step 1: Update fixed HRL validation comparisons**

In `train_warmup_then_joint_with_monitor`, replace comparisons like:

```python
if float(m["sharpe"]) > phase_best_sharpe:
    phase_best_sharpe = float(m["sharpe"])
```

with:

```python
score = trainer._validation_score(m, trainer.cfg, phase="joint")
if score > phase_best_sharpe:
    phase_best_sharpe = score
```

Apply this to:
- Warmup outer best selection.
- Warmup inner best selection.
- Outer+inner joint best selection.

Keep log messages readable by changing:

```python
trainer.logger.info(f"       (New Warmup-Inner Best: {phase_best_sharpe:.4f})")
```

to:

```python
trainer.logger.info(
    "       (New Warmup-Inner Best by %s: %.4f)",
    getattr(trainer.cfg, "model_selection_metric", "sharpe"),
    phase_best_sharpe,
)
```

- [ ] **Step 2: Update controller validation score**

Inside `train_controller_counterfactual_pg`, replace:

```python
score = (
    -float(getattr(self.cfg, "controller_mdd_coef", 2.0)) * float(metrics["max_dd"])
    + float(getattr(self.cfg, "controller_return_coef", 0.5)) * float(metrics["total_ret"])
)
```

with:

```python
score = self._validation_score(metrics, self.cfg, phase="controller")
```

Change the validation log to include the metric:

```python
self.logger.info(
    "   >>> [VAL controller_pg ep=%s] select=%s score=%.4f Sharpe=%.4f Ret=%.2f%% MDD=%.2f%% switches=%s",
    epoch_idx,
    getattr(self.cfg, "controller_selection_metric", "risk_return"),
    score,
    metrics["sharpe"],
    metrics["total_ret"] * 100.0,
    metrics["max_dd"] * 100.0,
    ret_stats.get("switch_count", 0),
)
```

- [ ] **Step 3: Run existing tests**

Run:

```bash
cd /home/tongwenxuan/KD4RL_plus
/home/tongwenxuan/conda/envs/xuangu/bin/python tests/test_return_selection_metric.py
/home/tongwenxuan/conda/envs/xuangu/bin/python tests/test_controller_counterfactual_pg.py
```

Expected: all tests pass.

---

### Task 4: Add CLI Flags Without Changing Defaults

**Files:**
- Modify: `run_hrl_training.py`

- [ ] **Step 1: Add parser arguments**

In `parse_args()`, add:

```python
    parser.add_argument(
        "--model_selection_metric",
        choices=["sharpe", "return", "mdd"],
        default="sharpe",
        help="Checkpoint selection metric for outer/inner/joint phases. Default keeps current behavior: sharpe.",
    )
    parser.add_argument(
        "--controller_selection_metric",
        choices=["risk_return", "return", "mdd", "sharpe"],
        default="risk_return",
        help="Checkpoint selection metric for controller PG. Default keeps current behavior: risk_return.",
    )
```

- [ ] **Step 2: Pass arguments to child command**

In `build_child_command`, add:

```python
        "--model_selection_metric",
        str(args.model_selection_metric),
        "--controller_selection_metric",
        str(args.controller_selection_metric),
```

- [ ] **Step 3: Store in runtime config and metadata**

In `set_runtime_training_args`, add:

```python
    runtime_config.model_selection_metric = str(args.model_selection_metric)
    runtime_config.controller_selection_metric = str(args.controller_selection_metric)
```

In `write_child_metadata`, add under `"training"`:

```python
            "model_selection_metric": getattr(runtime_config, "model_selection_metric", None),
            "controller_selection_metric": getattr(runtime_config, "controller_selection_metric", None),
```

In the child logger section, add:

```python
    logger.info(
        "Checkpoint selection: fixed_hrl=%s controller=%s",
        getattr(runtime_config, "model_selection_metric", "sharpe"),
        getattr(runtime_config, "controller_selection_metric", "risk_return"),
    )
```

- [ ] **Step 4: Verify default behavior stays unchanged**

Run:

```bash
cd /home/tongwenxuan/KD4RL_plus
/home/tongwenxuan/conda/envs/xuangu/bin/python -c "import run_hrl_training as r, sys; sys.argv=['run_hrl_training.py']; a=r.parse_args(); r.normalize_training_schedule(a); print(a.model_selection_metric, a.controller_selection_metric)"
```

Expected:

```text
sharpe risk_return
```

---

### Task 5: Add Separate Return-Selection Run Script

**Files:**
- Create: `scripts/run_hrl_min10_max40_pg_countband_return_select.sh`

- [ ] **Step 1: Create independent script**

Create:

```bash
#!/usr/bin/env bash
set -euo pipefail

cd /home/tongwenxuan/KD4RL_plus

PYTHON_BIN="${PYTHON_BIN:-/home/tongwenxuan/conda/envs/xuangu/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/hrl_controller_pg}"
RUN_NAME="${RUN_NAME:-min10_max40_pg_countband_return_select}"
GPU_ID="${GPU_ID:-0}"
HEARTBEAT_SECONDS="${HEARTBEAT_SECONDS:-300}"
JOINT_LR_MULT="${JOINT_LR_MULT:-0.01}"

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl-kd4rl}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPU_ID}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python not found or not executable: $PYTHON_BIN" >&2
  exit 1
fi

mkdir -p "$OUTPUT_ROOT/logs"

echo "Run name: $RUN_NAME"
echo "Output root: $OUTPUT_ROOT"
echo "Checkpoint selection: fixed_hrl=return controller=return"
echo "Joint LR multiplier: $JOINT_LR_MULT (base lr 1e-3 -> joint lr 1e-5)"

"$PYTHON_BIN" -u run_hrl_training.py \
  --markets nas \
  --seeds 42 43 44 45 46 \
  --output_root "$OUTPUT_ROOT" \
  --run_name "$RUN_NAME" \
  --device cuda \
  --warmup_inner_epochs 5 \
  --controller_epochs 10 \
  --controller_val_interval_epochs 1 \
  --controller_mdd_coef 5.0 \
  --controller_return_coef 0.2 \
  --joint_lr_mult "$JOINT_LR_MULT" \
  --model_selection_metric return \
  --controller_selection_metric return \
  --heartbeat_seconds "$HEARTBEAT_SECONDS" \
  --continue_on_error \
  2>&1 | tee "$OUTPUT_ROOT/logs/${RUN_NAME}_nas.log"

"$PYTHON_BIN" -u run_hrl_training.py \
  --markets sh \
  --seeds 75 76 77 78 79 \
  --output_root "$OUTPUT_ROOT" \
  --run_name "$RUN_NAME" \
  --device cuda \
  --warmup_inner_epochs 5 \
  --controller_epochs 10 \
  --controller_val_interval_epochs 1 \
  --controller_mdd_coef 5.0 \
  --controller_return_coef 0.2 \
  --joint_lr_mult "$JOINT_LR_MULT" \
  --model_selection_metric return \
  --controller_selection_metric return \
  --heartbeat_seconds "$HEARTBEAT_SECONDS" \
  --continue_on_error \
  2>&1 | tee "$OUTPUT_ROOT/logs/${RUN_NAME}_sh.log"
```

- [ ] **Step 2: Make executable and syntax-check**

Run:

```bash
cd /home/tongwenxuan/KD4RL_plus
chmod +x scripts/run_hrl_min10_max40_pg_countband_return_select.sh
bash -n scripts/run_hrl_min10_max40_pg_countband_return_select.sh
```

Expected: exit code 0.

---

### Task 6: Smoke Test The Opt-In Path

**Files:**
- No code changes.

- [ ] **Step 1: Run smoke with return selection**

Run:

```bash
cd /home/tongwenxuan/KD4RL_plus
MPLCONFIGDIR=/tmp/mpl-kd4rl /home/tongwenxuan/conda/envs/xuangu/bin/python -u run_hrl_training.py \
  --smoke \
  --markets nas \
  --seeds 42 \
  --output_root results/smoke_return_select \
  --run_name smoke \
  --device cpu \
  --warmup_inner_epochs 5 \
  --controller_epochs 10 \
  --controller_val_interval_epochs 1 \
  --controller_mdd_coef 5.0 \
  --controller_return_coef 0.2 \
  --joint_lr_mult 0.01 \
  --model_selection_metric return \
  --controller_selection_metric return \
  --heartbeat_seconds 30 \
  --continue_on_error
```

Expected:
- Command exits 0.
- Log includes `Checkpoint selection: fixed_hrl=return controller=return`.
- Smoke result files are written under `results/smoke_return_select/smoke/nas/`.

- [ ] **Step 2: Compile check**

Run:

```bash
cd /home/tongwenxuan/KD4RL_plus
/home/tongwenxuan/conda/envs/xuangu/bin/python -m py_compile \
  Train/PPO_train.py \
  run_hrl_training.py \
  Train/controller_pg.py
```

Expected: exit code 0.

---

## Verification

Run these commands after implementation:

```bash
cd /home/tongwenxuan/KD4RL_plus
/home/tongwenxuan/conda/envs/xuangu/bin/python tests/test_return_selection_metric.py
/home/tongwenxuan/conda/envs/xuangu/bin/python tests/test_controller_counterfactual_pg.py
bash -n scripts/run_hrl_min10_max40_pg_countband_return_select.sh
/home/tongwenxuan/conda/envs/xuangu/bin/python -m py_compile Train/PPO_train.py run_hrl_training.py Train/controller_pg.py
```

Optional smoke:

```bash
cd /home/tongwenxuan/KD4RL_plus
MPLCONFIGDIR=/tmp/mpl-kd4rl /home/tongwenxuan/conda/envs/xuangu/bin/python -u run_hrl_training.py \
  --smoke \
  --markets nas \
  --seeds 42 \
  --output_root results/smoke_return_select \
  --run_name smoke \
  --device cpu \
  --joint_lr_mult 0.01 \
  --model_selection_metric return \
  --controller_selection_metric return \
  --heartbeat_seconds 30 \
  --continue_on_error
```

---

## Expected Training Command After Implementation

Use the separate return-selection script:

```bash
cd /home/tongwenxuan/KD4RL_plus
bash scripts/run_hrl_min10_max40_pg_countband_return_select.sh
```

This leaves the current script unchanged:

```bash
bash scripts/run_hrl_min10_max40_pg_countband.sh
```

The return-selection script intentionally uses these opt-in training parameters:

```bash
--warmup_inner_epochs 5
--controller_epochs 10
--controller_val_interval_epochs 1
--controller_mdd_coef 5.0
--controller_return_coef 0.2
--model_selection_metric return
--controller_selection_metric return
--joint_lr_mult 0.01
```

---

## Self-Review

- Spec coverage: The plan adds return-based checkpoint selection while keeping current defaults and current script behavior unchanged.
- Placeholder scan: No TODO/TBD placeholders are present.
- Type consistency: The new config names are `model_selection_metric` and `controller_selection_metric` throughout parser, runtime config, metadata, and training code.

## Next Skill

Use `$superpower-executing-plans` for inline implementation, or `$superpower-subagents` for task-by-task implementation with review checkpoints.
