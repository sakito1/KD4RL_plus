# Interpretability Figures Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use $superpower-subagents (recommended) or $superpower-executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking via update_plan.

**Goal:** Build a reproducible paper-figure package explaining the archived SH/NAS Controller+HRL models, including ablation plots, controller traces, switch event studies, and Chinese conclusion text.

**Architecture:** Add one standalone script that loads the archived command JSON files and checkpoints, reconstructs the original runtime config without retraining, reruns deterministic test episodes, records controller diagnostics, and writes CSV/PNG/Markdown artifacts under `results/end/interpretability/`. Keep reusable metric/event-study helpers pure enough to unit test without loading market data or checkpoints.

**Tech Stack:** Python, PyTorch, pandas, numpy, matplotlib, existing `run_hrl_training.py`, `PPO_Env`, `HRL_Networks`, `HRL_PPO_Agent`, and `HRL_Trainer`.

---

## File Structure

- Create `scripts/generate_interpretability_figures.py`: command-line entrypoint, archived-run metadata, metric helpers, model/test trace collection, plotting, and Chinese conclusion generation.
- Create `tests/test_interpretability_figures.py`: lightweight tests for archived metrics, financial metrics, correlation handling, and event-study alignment. These tests must not instantiate the environment or load checkpoints.
- Output generated artifacts to `results/end/interpretability/`: CSV traces, PNG figures, and `paper_conclusions_zh.md`.

---

### Task 1: Add Helper Tests

**Files:**
- Create: `tests/test_interpretability_figures.py`
- Later import target: `scripts/generate_interpretability_figures.py`

- [ ] **Step 1: Write the failing helper tests**

Create `tests/test_interpretability_figures.py` with:

```python
import numpy as np

from scripts.generate_interpretability_figures import (
    ARCHIVED_METRICS,
    build_switch_event_study,
    compute_financial_metrics,
    safe_corr,
)


def test_archived_metrics_include_confirmed_final_returns():
    sh_s3 = ARCHIVED_METRICS[("sh", "Controller+HRL")]
    nas_s3 = ARCHIVED_METRICS[("nas", "Controller+HRL")]

    assert round(sh_s3["total_return"] * 100.0, 2) == 204.99
    assert round(nas_s3["total_return"] * 100.0, 2) == 265.53
    assert sh_s3["free_switches"] == 102
    assert nas_s3["free_switches"] == 231


def test_compute_financial_metrics_matches_simple_curve():
    values = [1.0, 1.1, 1.21, 1.089]
    metrics = compute_financial_metrics(values)

    assert round(metrics["total_return"], 3) == 0.089
    assert round(metrics["max_drawdown"], 3) == 0.100
    assert "sharpe" in metrics


def test_safe_corr_handles_constant_inputs():
    assert safe_corr([1.0, 1.0, 1.0], [0.1, 0.2, 0.3]) == 0.0
    assert round(safe_corr([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]), 6) == 1.0


def test_build_switch_event_study_aligns_windows():
    returns = np.array([0.0, 0.01, -0.02, 0.03, 0.04, -0.01, 0.02])
    event_idx = [3]
    study = build_switch_event_study(returns, event_idx, pre_days=2, post_days=2)

    assert study["offset"].tolist() == [-2, -1, 0, 1, 2]
    assert len(study) == 5
    assert study["event_count"].iloc[0] == 1
    assert study["mean_cum_return"].iloc[2] == 0.0
```

- [ ] **Step 2: Run tests and verify they fail because the script does not exist**

Run:

```bash
/home/tongwenxuan/conda/envs/xuangu/bin/python -m pytest tests/test_interpretability_figures.py -q
```

Expected: import failure for `scripts.generate_interpretability_figures`.

---

### Task 2: Implement Pure Helpers And Artifact Scaffolding

**Files:**
- Create: `scripts/generate_interpretability_figures.py`
- Modify: `tests/test_interpretability_figures.py` only if imports need formatting fixes.

- [ ] **Step 1: Add constants, CLI shell, and pure helpers**

Create `scripts/generate_interpretability_figures.py` with these top-level pieces:

```python
#!/usr/bin/env python
import argparse
import contextlib
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-kd4rl")
os.environ.setdefault("MKL_THREADING_LAYER", "GNU")

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS_END = ROOT / "results" / "end"
DEFAULT_OUTPUT_DIR = RESULTS_END / "interpretability"

ARCHIVED_RUNS = {
    "sh": {
        "label": "SH seed 90",
        "market_name": "A-share",
        "run_dir": RESULTS_END / "sh_seed90",
        "checkpoint": RESULTS_END / "sh_seed90" / "checkpoints" / "best_model.pth",
        "command_json": RESULTS_END / "sh_seed90" / "seed_90_command.json",
        "seed": 90,
    },
    "nas": {
        "label": "NAS seed 49",
        "market_name": "NAS100",
        "run_dir": RESULTS_END / "nas_seed49",
        "checkpoint": RESULTS_END / "nas_seed49" / "checkpoints" / "best_model.pth",
        "command_json": RESULTS_END / "nas_seed49" / "seed_49_command.json",
        "seed": 49,
    },
}

ARCHIVED_METRICS = {
    ("sh", "Fixed HRL"): {
        "total_return": 1.5899,
        "ann_return": 0.2158,
        "ann_vol": 0.2175,
        "sharpe": 0.9924,
        "max_drawdown": 0.2085,
        "switches": 42,
        "free_switches": 0,
    },
    ("sh", "No Inner"): {
        "total_return": 1.4705,
        "ann_return": 0.2066,
        "ann_vol": 0.2190,
        "sharpe": 0.9434,
        "max_drawdown": 0.2099,
        "switches": 42,
        "free_switches": 0,
    },
    ("sh", "Controller+HRL"): {
        "total_return": 2.0499,
        "ann_return": 0.2493,
        "ann_vol": 0.2194,
        "sharpe": 1.1359,
        "max_drawdown": 0.2278,
        "switches": 129,
        "free_switches": 102,
    },
    ("nas", "Fixed HRL"): {
        "total_return": 2.2743,
        "ann_return": 0.2420,
        "ann_vol": 0.2183,
        "sharpe": 1.1089,
        "max_drawdown": 0.3173,
        "switches": 46,
        "free_switches": 0,
    },
    ("nas", "No Inner"): {
        "total_return": 2.2042,
        "ann_return": 0.2380,
        "ann_vol": 0.2179,
        "sharpe": 1.0919,
        "max_drawdown": 0.3209,
        "switches": 46,
        "free_switches": 0,
    },
    ("nas", "Controller+HRL"): {
        "total_return": 2.6553,
        "ann_return": 0.2648,
        "ann_vol": 0.2303,
        "sharpe": 1.1500,
        "max_drawdown": 0.1862,
        "switches": 266,
        "free_switches": 231,
    },
}
```

Add:

```python
def compute_financial_metrics(values: Sequence[float]) -> Dict[str, float]:
    series = pd.Series(list(values), dtype="float64")
    if len(series) < 2:
        final_value = float(series.iloc[-1]) if len(series) else 0.0
        return {
            "total_return": 0.0,
            "ann_return": 0.0,
            "ann_vol": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "final_value": final_value,
        }
    ret = series.pct_change().fillna(0.0)
    total_return = float(series.iloc[-1] / series.iloc[0] - 1.0)
    ann_return = float(ret.mean() * 252.0)
    ann_vol = float(ret.std() * np.sqrt(252.0))
    sharpe = float(ann_return / (ann_vol + 1e-8)) if ann_vol > 1e-8 else 0.0
    peak = series.cummax()
    max_drawdown = float(((peak - series) / peak).max())
    return {
        "total_return": total_return,
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "final_value": float(series.iloc[-1]),
    }


def safe_corr(x: Sequence[float], y: Sequence[float]) -> float:
    x_arr = np.asarray(list(x), dtype="float64")
    y_arr = np.asarray(list(y), dtype="float64")
    n = min(x_arr.size, y_arr.size)
    if n < 2:
        return 0.0
    x_arr = x_arr[:n]
    y_arr = y_arr[:n]
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    if int(mask.sum()) < 2:
        return 0.0
    x_arr = x_arr[mask]
    y_arr = y_arr[mask]
    if np.std(x_arr) <= 1e-12 or np.std(y_arr) <= 1e-12:
        return 0.0
    return float(np.corrcoef(x_arr, y_arr)[0, 1])


def build_switch_event_study(
    daily_returns: Sequence[float],
    event_indices: Sequence[int],
    *,
    pre_days: int = 10,
    post_days: int = 20,
) -> pd.DataFrame:
    returns = np.asarray(list(daily_returns), dtype="float64")
    offsets = np.arange(-int(pre_days), int(post_days) + 1)
    curves = []
    for event_idx in event_indices:
        start = int(event_idx) - int(pre_days)
        stop = int(event_idx) + int(post_days)
        if start < 0 or stop >= returns.size:
            continue
        window = returns[start : stop + 1].copy()
        window[int(pre_days)] = 0.0
        cumulative = np.cumprod(1.0 + window) - 1.0
        cumulative = cumulative - cumulative[int(pre_days)]
        curves.append(cumulative)
    if not curves:
        return pd.DataFrame({
            "offset": offsets,
            "mean_cum_return": np.zeros_like(offsets, dtype="float64"),
            "low_cum_return": np.zeros_like(offsets, dtype="float64"),
            "high_cum_return": np.zeros_like(offsets, dtype="float64"),
            "event_count": np.zeros_like(offsets, dtype="int64"),
        })
    matrix = np.vstack(curves)
    return pd.DataFrame({
        "offset": offsets,
        "mean_cum_return": matrix.mean(axis=0),
        "low_cum_return": np.percentile(matrix, 25, axis=0),
        "high_cum_return": np.percentile(matrix, 75, axis=0),
        "event_count": np.full(offsets.shape, matrix.shape[0], dtype="int64"),
    })
```

Add a CLI:

```python
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate paper-ready interpretability figures for archived Controller+HRL models.")
    parser.add_argument("--markets", nargs="+", choices=sorted(ARCHIVED_RUNS), default=["sh", "nas"])
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--skip_model_eval", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_ablation_metrics(output_dir)
    if args.skip_model_eval:
        write_paper_conclusions(output_dir, {})
        return
    traces = collect_all_traces(args.markets, output_dir, args.device)
    write_all_figures(output_dir, traces)
    write_paper_conclusions(output_dir, traces)


if __name__ == "__main__":
    main()
```

At the end of Task 2, `collect_all_traces`, `write_all_figures`, and `write_paper_conclusions` can be stubs that write no model-dependent data, because Task 3 and Task 4 will fill them in. They must not raise under `--skip_model_eval`.

- [ ] **Step 2: Implement ablation CSV and conclusion stub**

Add:

```python
def write_ablation_metrics(output_dir: Path) -> pd.DataFrame:
    rows = []
    for (market, scenario), metrics in ARCHIVED_METRICS.items():
        rows.append({"market": market, "scenario": scenario, **metrics})
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "ablation_metrics.csv", index=False)
    return df


def collect_all_traces(markets: Sequence[str], output_dir: Path, device: str) -> Dict[str, Dict[str, pd.DataFrame]]:
    return {}


def write_all_figures(output_dir: Path, traces: Dict[str, Dict[str, pd.DataFrame]]) -> None:
    return None


def write_paper_conclusions(output_dir: Path, traces: Dict[str, Dict[str, pd.DataFrame]]) -> Path:
    path = output_dir / "paper_conclusions_zh.md"
    path.write_text(
        "# 可解释性结论草稿\n\n"
        "本文件将在完整模型探针运行后写入最终论文表述。\n",
        encoding="utf-8",
    )
    return path
```

- [ ] **Step 3: Run helper tests**

Run:

```bash
/home/tongwenxuan/conda/envs/xuangu/bin/python -m pytest tests/test_interpretability_figures.py -q
```

Expected: all tests pass.

- [ ] **Step 4: Smoke run without model loading**

Run:

```bash
/home/tongwenxuan/conda/envs/xuangu/bin/python scripts/generate_interpretability_figures.py --skip_model_eval
```

Expected: creates `results/end/interpretability/ablation_metrics.csv` and `results/end/interpretability/paper_conclusions_zh.md`.

---

### Task 3: Implement Archived Model Trace Collection

**Files:**
- Modify: `scripts/generate_interpretability_figures.py`

- [ ] **Step 1: Add imports for the real HRL stack inside model-loading helpers**

Add imports inside helper functions, not module top level, so pure tests remain fast:

```python
def _import_hrl_stack():
    import torch
    import run_hrl_training as runner
    import utils.config as runtime_config
    from Train.PPO_train import HRL_Networks, HRL_Trainer
    from agent import HRL_Buffer, HRL_PPO_Agent
    from env import PPO_Env
    from utils.Log import create_logger

    return torch, runner, runtime_config, HRL_Networks, HRL_Trainer, HRL_Buffer, HRL_PPO_Agent, PPO_Env, create_logger
```

- [ ] **Step 2: Reconstruct archived args from command JSON**

Add:

```python
@contextlib.contextmanager
def _temporary_argv(argv: Sequence[str]):
    old_argv = sys.argv[:]
    sys.argv = list(argv)
    try:
        yield
    finally:
        sys.argv = old_argv


def _args_from_command_json(command_json: Path, *, device: str, output_dir: Path):
    _, runner, _, *_ = _import_hrl_stack()
    payload = json.loads(command_json.read_text(encoding="utf-8"))
    command = payload["command"]
    script_idx = next(i for i, token in enumerate(command) if str(token).endswith("run_hrl_training.py"))
    args_list = list(command[script_idx + 1 :])
    with _temporary_argv(["run_hrl_training.py", *args_list]):
        args = runner.parse_args()
    runner.normalize_training_schedule(args)
    args.device = str(device)
    args.cpu = str(device).lower() == "cpu"
    args.run_root = str(output_dir / "_probe_runtime")
    args.skip_test = True
    args.test_only_checkpoint = None
    return args
```

- [ ] **Step 3: Build a trainer and load the archived checkpoint**

Add:

```python
def build_loaded_trainer(market: str, output_dir: Path, device: str):
    torch, runner, runtime_config, HRL_Networks, HRL_Trainer, HRL_Buffer, HRL_PPO_Agent, PPO_Env, create_logger = _import_hrl_stack()
    run = ARCHIVED_RUNS[market]
    args = _args_from_command_json(Path(run["command_json"]), device=device, output_dir=output_dir)
    label, config_module = runner.MARKET_CONFIGS[market]
    runner.apply_market_config(config_module)
    market_root = output_dir / "_probe_runtime" / market
    runner.set_runtime_training_args(args, market_root, int(run["seed"]))
    runtime_config.min_hold = int(args.min_hold)
    runtime_config.max_hold = int(args.max_hold)
    runtime_config.fixed_cycle = int(args.max_hold)
    runtime_config.rollout_segments_by_stage = {
        stage: int(getattr(args, f"{stage}_rollout_segments"))
        for stage in runner.TRAINING_STAGES
    }
    runtime_config.rollout_update_steps_by_stage = {
        stage: runtime_config.fixed_cycle * segments
        for stage, segments in runtime_config.rollout_segments_by_stage.items()
    }
    runtime_config.rollout_update_steps = runtime_config.rollout_update_steps_by_stage["warmup_outer"]
    logger = create_logger(str(output_dir / "_probe_runtime" / market / "logs" / f"seed_{run['seed']}"))
    runner.set_seed(int(run["seed"]), logger)
    env = PPO_Env(
        logger=logger,
        episode_len=args.episode_len,
        outer_window=args.outer_window,
        max_hold=int(args.max_hold),
        min_hold=int(args.min_hold),
        train_episodes_per_epoch=args.train_episodes_per_epoch,
        train_start_stride_days=args.train_start_stride_days,
        train_episode_to_end=args.train_episode_to_end,
    )
    networks = HRL_Networks(args.ssm_dim, env.num_stocks, runtime_config).to(runtime_config.device)
    agent = HRL_PPO_Agent(networks, runtime_config)
    buffer = HRL_Buffer(capacity=3000, device=runtime_config.device, outer_reward_scale=getattr(runtime_config, "reward_scale_outer", 1.0))
    trainer = HRL_Trainer(agent, env, buffer, runtime_config, logger)
    checkpoint = torch.load(Path(run["checkpoint"]), map_location=runtime_config.device)
    trainer.agent.net.load_state_dict(checkpoint["agent_net"])
    trainer.agent.net.eval()
    return trainer
```

- [ ] **Step 4: Implement deterministic scenario trace collection**

Add:

```python
def _to_float(tensor, default=0.0) -> float:
    if tensor is None:
        return float(default)
    if hasattr(tensor, "detach"):
        return float(tensor.detach().view(-1)[0].cpu().item())
    return float(tensor)


def _normalized_row_sum_abs(x, y) -> Tuple[float, float, float]:
    import torch

    x = x.detach().view(-1).clamp_min(0.0)
    y = y.detach().view(-1).clamp_min(0.0)
    x = x / x.sum().clamp_min(1e-8)
    y = y / y.sum().clamp_min(1e-8)
    turnover = float(torch.sum(torch.abs(y - x)).cpu().item())
    overlap = float(torch.sum(torch.minimum(x, y)).cpu().item())
    concentration = float(torch.sum(x.pow(2)).cpu().item())
    return turnover, overlap, concentration
```

Add:

```python
def collect_scenario_trace(trainer, *, scenario: str, fixed_cycle=None, disable_inner=False) -> pd.DataFrame:
    import torch

    env = trainer.env
    env.set_mode("test")
    test_window = trainer._test_episode_window()
    obs = env.reset_at(*test_window) if test_window is not None else env.reset()
    spec = trainer._get_phase_spec("joint")
    rows = []
    history = [float(env.portfolio_value.item())]
    step_idx = 0
    last_switch_step = 0
    segment_start_value = float(env.portfolio_value.item())
    segment_peak_value = segment_start_value
    switch_count = 0
    free_switch_count = 0

    with torch.no_grad():
        while True:
            duration = step_idx - last_switch_step
            if fixed_cycle is not None:
                force_switch = 1 if step_idx == 0 or duration >= int(fixed_cycle) else 0
                force_locked = True
            else:
                force_switch, force_locked = trainer._compute_force_switch_locked(
                    spec=spec,
                    phase="joint",
                    step_idx=step_idx,
                    duration=duration,
                    is_train=False,
                    switch_schedule=None,
                    fixed_cycle=None,
                    current_segments=switch_count,
                    rollout_len=int(getattr(env, "current_episode_len", env.episode_len)),
                )
            out = trainer.agent.get_action(
                obs,
                mode="eval",
                force_switch=force_switch,
                force_inner_zero=bool(disable_inner),
                force_locked=force_locked,
            )
            stats = trainer.agent.net.mon.decision_stats(
                obs["ssm"]["z"],
                obs["ssm"]["h"],
                obs["ssm"]["p"],
                obs["ssm"]["q_bear"],
                obs["ssm"]["q_bull"],
                obs["weights_drift"],
                obs["port_state"],
                switch_action=out["act_out"],
                asset_state=obs.get("outer_state"),
            )
            is_switch = bool(out["act_mon"].view(-1)[0].item() == 1)
            is_forced = force_switch is not None
            is_free_switch = bool(is_switch and not is_forced)
            value_before = float(env.portfolio_value.item())
            segment_peak_value = max(segment_peak_value, value_before)
            segment_return = value_before / max(segment_start_value, 1e-12) - 1.0
            segment_drawdown = 1.0 - value_before / max(segment_peak_value, 1e-12)
            turnover_candidate, overlap_candidate, concentration = _normalized_row_sum_abs(
                obs["weights_drift"],
                out["act_out"],
            )
            date_value = str(env.all_dates[int(env.day)].date()) if hasattr(env, "all_dates") else str(int(env.day))
            next_obs, _, done, info = env.step(
                out["weights_exec"].detach(),
                out["base_used"].detach(),
                outer_action=out["act_out"].detach(),
                is_switch=is_switch,
            )
            value_after = float(info["portfolio_value"])
            daily_return = value_after / max(value_before, 1e-12) - 1.0
            if is_switch:
                switch_count += 1
                if is_free_switch:
                    free_switch_count += 1
                last_switch_step = step_idx
                segment_start_value = value_after
                segment_peak_value = value_after
            rows.append({
                "scenario": scenario,
                "date": date_value,
                "step": step_idx,
                "portfolio_value": value_after,
                "daily_return": daily_return,
                "is_switch": int(is_switch),
                "is_free_switch": int(is_free_switch),
                "is_forced": int(is_forced),
                "hold_duration": duration,
                "exit_prob": _to_float(stats["exit_prob"]),
                "base_exit_logit": _to_float(stats["base_exit_logit"]),
                "exit_logit": _to_float(stats["exit_logit"]),
                "switch_advantage_pred": _to_float(stats["switch_advantage_pred"]),
                "hold_return_pred": _to_float(stats["hold_return_pred"]),
                "hold_risk_pred": _to_float(stats["hold_risk_pred"]),
                "segment_return": segment_return,
                "segment_drawdown": segment_drawdown,
                "candidate_turnover": turnover_candidate,
                "candidate_overlap": overlap_candidate,
                "hold_concentration": concentration,
                "switch_count_so_far": switch_count,
                "free_switch_count_so_far": free_switch_count,
            })
            history.append(value_after)
            if done:
                break
            obs = next_obs
            step_idx += 1
    return pd.DataFrame(rows)
```

- [ ] **Step 5: Collect all market traces and write CSVs**

Add:

```python
def collect_all_traces(markets: Sequence[str], output_dir: Path, device: str) -> Dict[str, Dict[str, pd.DataFrame]]:
    traces: Dict[str, Dict[str, pd.DataFrame]] = {}
    for market in markets:
        trainer = build_loaded_trainer(market, output_dir, device)
        fixed_cycle = int(getattr(trainer.cfg, "max_hold", 30))
        market_traces = {
            "Fixed HRL": collect_scenario_trace(trainer, scenario="Fixed HRL", fixed_cycle=fixed_cycle, disable_inner=False),
            "No Inner": collect_scenario_trace(trainer, scenario="No Inner", fixed_cycle=fixed_cycle, disable_inner=True),
            "Controller+HRL": collect_scenario_trace(trainer, scenario="Controller+HRL", fixed_cycle=None, disable_inner=False),
        }
        traces[market] = market_traces
        pd.concat(market_traces.values(), ignore_index=True).to_csv(output_dir / f"scenario_traces_{market}.csv", index=False)
        market_traces["Controller+HRL"].to_csv(output_dir / f"controller_trace_{market}.csv", index=False)
        event_indices = market_traces["Controller+HRL"].index[market_traces["Controller+HRL"]["is_free_switch"] == 1].tolist()
        event_df = build_switch_event_study(
            market_traces["Controller+HRL"]["daily_return"].to_numpy(),
            event_indices,
            pre_days=10,
            post_days=20,
        )
        event_df.to_csv(output_dir / f"switch_event_{market}.csv", index=False)
    return traces
```

- [ ] **Step 6: Run a short model-eval smoke if full run is too slow**

Run:

```bash
/home/tongwenxuan/conda/envs/xuangu/bin/python scripts/generate_interpretability_figures.py --markets sh --device cpu
```

Expected: writes `scenario_traces_sh.csv`, `controller_trace_sh.csv`, and `switch_event_sh.csv`. If CPU is too slow, rerun with `--device cuda` only if CUDA is available.

---

### Task 4: Implement Paper Figures And Chinese Conclusions

**Files:**
- Modify: `scripts/generate_interpretability_figures.py`

- [ ] **Step 1: Add plotting helpers**

Add:

```python
def _load_or_concat_traces(output_dir: Path, traces: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, Dict[str, pd.DataFrame]]:
    if traces:
        return traces
    loaded = {}
    for market in ARCHIVED_RUNS:
        path = output_dir / f"scenario_traces_{market}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        loaded[market] = {scenario: part.copy() for scenario, part in df.groupby("scenario")}
    return loaded


def _savefig(path: Path) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()
```

- [ ] **Step 2: Draw Figure 1 ablation summary**

Add `plot_ablation_summary(output_dir)` that creates `fig1_ablation_summary.png` with three panels:

```python
def plot_ablation_summary(output_dir: Path) -> Path:
    import matplotlib.pyplot as plt

    df = write_ablation_metrics(output_dir)
    scenarios = ["No Inner", "Fixed HRL", "Controller+HRL"]
    markets = ["sh", "nas"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    specs = [
        ("total_return", "Total return", 100.0),
        ("sharpe", "Sharpe", 1.0),
        ("max_drawdown", "Max drawdown", 100.0),
    ]
    colors = {"No Inner": "#8f8f8f", "Fixed HRL": "#2f6fbb", "Controller+HRL": "#c4493d"}
    x = np.arange(len(markets))
    width = 0.24
    for ax, (col, title, scale) in zip(axes, specs):
        for offset, scenario in zip([-width, 0.0, width], scenarios):
            vals = [
                float(df[(df["market"] == market) & (df["scenario"] == scenario)][col].iloc[0]) * scale
                for market in markets
            ]
            ax.bar(x + offset, vals, width=width, label=scenario, color=colors[scenario])
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(["SH", "NAS"])
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("%")
    axes[2].set_ylabel("%")
    axes[0].legend(frameon=False, fontsize=8)
    path = output_dir / "fig1_ablation_summary.png"
    _savefig(path)
    return path
```

- [ ] **Step 3: Draw Figures 2 through 5**

Add:

```python
def plot_equity_curves(output_dir: Path, traces: Dict[str, Dict[str, pd.DataFrame]]) -> Path:
    import matplotlib.pyplot as plt

    traces = _load_or_concat_traces(output_dir, traces)
    fig, axes = plt.subplots(len(traces), 1, figsize=(10, 3.6 * max(1, len(traces))), sharex=False)
    axes = np.atleast_1d(axes)
    colors = {"No Inner": "#8f8f8f", "Fixed HRL": "#2f6fbb", "Controller+HRL": "#c4493d"}
    for ax, (market, market_traces) in zip(axes, traces.items()):
        for scenario in ["No Inner", "Fixed HRL", "Controller+HRL"]:
            df = market_traces[scenario]
            y = df["portfolio_value"] / float(df["portfolio_value"].iloc[0])
            ax.plot(pd.to_datetime(df["date"]), y, label=scenario, color=colors[scenario], linewidth=1.5)
        ax.set_title(f"{ARCHIVED_RUNS[market]['market_name']} test equity curves")
        ax.set_ylabel("Normalized value")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, ncol=3, fontsize=8)
    path = output_dir / "fig2_equity_curves.png"
    _savefig(path)
    return path
```

Add `plot_controller_timeline`, `plot_signal_relationships`, and `plot_switch_event_study` using the recorded CSV fields from Task 3. Use English figure labels to avoid CJK font issues in matplotlib. Save to:

- `fig3_controller_timeline.png`
- `fig4_controller_signal_relationships.png`
- `fig5_switch_event_study.png`

For Figure 4, annotate correlations using `safe_corr` for:

- `exit_prob` vs `segment_drawdown`
- `exit_prob` vs `segment_return`
- `exit_prob` vs `switch_advantage_pred`
- `exit_prob` vs `hold_duration`

- [ ] **Step 4: Wire figure generation**

Replace the stub:

```python
def write_all_figures(output_dir: Path, traces: Dict[str, Dict[str, pd.DataFrame]]) -> None:
    traces = _load_or_concat_traces(output_dir, traces)
    plot_ablation_summary(output_dir)
    if traces:
        plot_equity_curves(output_dir, traces)
        plot_controller_timeline(output_dir, traces)
        plot_signal_relationships(output_dir, traces)
        plot_switch_event_study(output_dir)
```

- [ ] **Step 5: Write final Chinese conclusion text**

Replace `write_paper_conclusions` with text that includes the archived quantitative deltas:

```python
def write_paper_conclusions(output_dir: Path, traces: Dict[str, Dict[str, pd.DataFrame]]) -> Path:
    path = output_dir / "paper_conclusions_zh.md"
    text = """# 可解释性结论草稿

## 模块有效性

在固定 30 个交易日再平衡的设定下，inner actor 对 outer actor 给出的 base portfolio 进行权重微调后，两个测试市场均获得了稳定增益。SH 市场中，去掉 inner 后总收益为 147.05%，启用 inner 后提升到 158.99%，Sharpe 从 0.9434 提升到 0.9924；NAS 市场中，总收益从 220.42% 提升到 227.43%，Sharpe 从 1.0919 提升到 1.1089。这说明 inner actor 并不是冗余模块，而是在外层选股结果内部进一步优化了资金分配。

controller 的作用更明显：它将固定 30 日再平衡改为状态驱动的动态切换。SH 市场中，完整模型总收益达到 204.99%，相对固定 HRL 提升约 46.00 个百分点；NAS 市场中，完整模型总收益达到 265.53%，相对固定 HRL 提升约 38.10 个百分点，并且最大回撤从 31.73% 降低到 18.62%。这说明 controller 不只是增加交易频率，而是在测试期内学到了更有价值的再平衡时机。

## controller 在做什么

从 controller trace 可以看到，controller 每天根据当前真实漂移权重、当前持仓段收益/回撤、候选切换组合与当前组合的差异，以及 switch advantage head 的预测输出 `exit_prob`。当 `exit_prob` 超过阈值时，系统触发 outer actor 重新选股；否则继续持有并仅由 inner actor 做权重微调。

图 3 展示了 switch 不是机械地每 30 天发生，而是集中在若干状态变化阶段。图 4 进一步展示 `exit_prob` 与持仓段状态、候选切换优势之间的关系；图 5 将自由切换日对齐后观察切换前后的平均收益路径，用来说明 controller 是否倾向于在局部走弱后触发换仓，以及换仓后组合表现是否改善。

需要强调的是，这些图提供的是模型层面的可解释性证据：它们说明 controller 的决策与可观测组合状态和候选切换信号相关，并证明该机制在 held-out test period 中改善了结果；但它们不应被表述为对真实市场因果机制的严格证明。
"""
    path.write_text(text, encoding="utf-8")
    return path
```

---

### Task 5: Verify Full Output Package

**Files:**
- Verify generated files under `results/end/interpretability/`

- [ ] **Step 1: Run tests**

Run:

```bash
/home/tongwenxuan/conda/envs/xuangu/bin/python -m pytest tests/test_interpretability_figures.py -q
```

Expected: all tests pass.

- [ ] **Step 2: Generate full artifact package**

Run:

```bash
/home/tongwenxuan/conda/envs/xuangu/bin/python scripts/generate_interpretability_figures.py --device cpu
```

If CPU is too slow and CUDA is available, run:

```bash
/home/tongwenxuan/conda/envs/xuangu/bin/python scripts/generate_interpretability_figures.py --device cuda
```

Expected files:

```text
results/end/interpretability/ablation_metrics.csv
results/end/interpretability/scenario_traces_sh.csv
results/end/interpretability/scenario_traces_nas.csv
results/end/interpretability/controller_trace_sh.csv
results/end/interpretability/controller_trace_nas.csv
results/end/interpretability/switch_event_sh.csv
results/end/interpretability/switch_event_nas.csv
results/end/interpretability/fig1_ablation_summary.png
results/end/interpretability/fig2_equity_curves.png
results/end/interpretability/fig3_controller_timeline.png
results/end/interpretability/fig4_controller_signal_relationships.png
results/end/interpretability/fig5_switch_event_study.png
results/end/interpretability/paper_conclusions_zh.md
```

- [ ] **Step 3: Inspect generated images**

Run:

```bash
file results/end/interpretability/fig*.png
```

Expected: each file is a valid PNG image with nonzero dimensions.

- [ ] **Step 4: Check trace sanity**

Run:

```bash
/home/tongwenxuan/conda/envs/xuangu/bin/python - <<'PY'
import pandas as pd
for market in ["sh", "nas"]:
    df = pd.read_csv(f"results/end/interpretability/controller_trace_{market}.csv")
    print(market, len(df), int(df["is_switch"].sum()), int(df["is_free_switch"].sum()), df["exit_prob"].mean())
PY
```

Expected: both markets have nonempty traces, nonzero switch counts, and finite `exit_prob` means.

- [ ] **Step 5: Commit implementation**

Commit only files created for this task:

```bash
git add scripts/generate_interpretability_figures.py tests/test_interpretability_figures.py docs/superpowers/plans/2026-06-21-interpretability-figures.md
git commit -m "Add interpretability figure generation"
```

Do not add generated `results/end/interpretability/` artifacts unless the user explicitly asks to version them.

---

## Verification

The final work is complete when:

- Helper tests pass.
- The script generates all CSV/PNG/Markdown artifacts.
- `ablation_metrics.csv` reproduces the archived final metrics.
- Controller trace CSVs contain daily diagnostics and free switch events for both markets.
- The final response lists the exact artifact paths and the main quantitative conclusions.

## Next Skill

Use `$superpower-executing-plans` for inline execution in this session.
