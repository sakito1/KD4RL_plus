import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd


CHECKPOINT_NAMES = ("hrl_fixed_best", "controller_best", "best_model")


@dataclass
class CheckpointInfo:
    name: str
    path: Path
    exists: bool


@dataclass
class RunInfo:
    market: str
    seed: int
    run_dir: Path
    command_json: Path
    checkpoints: Dict[str, CheckpointInfo]


def parse_seed_specs(seed_specs: Sequence[str], markets: Sequence[str] = None) -> Dict[str, List[int]]:
    if not seed_specs:
        default = {"sh": [90], "nas": [49]}
        return {m: default[m] for m in markets} if markets else default
    parsed: Dict[str, List[int]] = {}
    for spec in seed_specs:
        if ":" in str(spec):
            market, seed = str(spec).split(":", 1)
            parsed.setdefault(market, []).append(int(seed))
        else:
            if not markets:
                raise ValueError("bare seed requires --markets")
            for market in markets:
                parsed.setdefault(market, []).append(int(spec))
    return parsed


def discover_runs(results_root: Path, *, markets: Sequence[str], seed_map: Dict[str, List[int]]) -> List[RunInfo]:
    results_root = Path(results_root)
    runs = []
    for market in markets:
        for seed in seed_map.get(market, []):
            run_dir = results_root / f"{market}_seed{seed}"
            command_json = run_dir / f"seed_{seed}_command.json"
            ckpt_dir = run_dir / "checkpoints"
            checkpoints = {
                name: CheckpointInfo(name, ckpt_dir / f"{name}.pth", (ckpt_dir / f"{name}.pth").exists())
                for name in CHECKPOINT_NAMES
            }
            if not run_dir.exists():
                warnings.warn(f"missing run directory: {run_dir}", RuntimeWarning)
            if not command_json.exists():
                warnings.warn(f"missing command json: {command_json}", RuntimeWarning)
            for info in checkpoints.values():
                if not info.exists:
                    warnings.warn(f"missing checkpoint: {info.path}", RuntimeWarning)
            runs.append(RunInfo(market, int(seed), run_dir, command_json, checkpoints))
    return runs


def ensure_output_dirs(output_dir: Path) -> Dict[str, Path]:
    root = Path(output_dir)
    paths = {
        "root": root,
        "traces": root / "traces",
        "metrics": root / "metrics",
        "figures": root / "figures",
        "tables": root / "tables",
        "logs": root / "logs",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def normalize_portfolio_trace(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    out = df.copy()
    out["portfolio_value"] = pd.to_numeric(out["portfolio_value"], errors="coerce")
    if "portfolio_value_before" not in out:
        out["portfolio_value_before"] = out["portfolio_value"].shift(1).fillna(out["portfolio_value"].iloc[0])
    out["portfolio_value_before"] = pd.to_numeric(out["portfolio_value_before"], errors="coerce")
    if "daily_simple_return" not in out:
        if "daily_return" in out:
            out["daily_simple_return"] = pd.to_numeric(out["daily_return"], errors="coerce")
        else:
            out["daily_simple_return"] = out["portfolio_value"].pct_change().fillna(0.0)
    out["daily_log_return"] = np.log1p(pd.to_numeric(out["daily_simple_return"], errors="coerce").fillna(0.0))
    if "base_log_return" not in out:
        out["base_log_return"] = out["daily_log_return"]
    if "inner_alpha" not in out:
        out["inner_alpha"] = out["daily_log_return"] - pd.to_numeric(out["base_log_return"], errors="coerce")
    first = float(out["portfolio_value"].iloc[0]) if len(out) else 1.0
    out["cumulative_return"] = out["portfolio_value"] / max(first, 1e-12) - 1.0
    peak = out["portfolio_value"].cummax().clip(lower=1e-12)
    out["drawdown"] = (peak - out["portfolio_value"]) / peak
    for col in ["turnover", "cost_rate", "is_switch", "is_free_switch", "is_forced_switch", "is_forced_hold"]:
        if col not in out:
            out[col] = 0.0
    if "holding_duration" not in out:
        out["holding_duration"] = out.get("hold_duration", np.nan)
    if "switch_count" not in out:
        out["switch_count"] = pd.to_numeric(out["is_switch"], errors="coerce").fillna(0).cumsum()
    if "free_switch_count" not in out:
        out["free_switch_count"] = pd.to_numeric(out["is_free_switch"], errors="coerce").fillna(0).cumsum()
    return out


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")


def parse_json_curve(value) -> np.ndarray:
    if isinstance(value, str) and value:
        try:
            return np.asarray(json.loads(value), dtype="float64")
        except json.JSONDecodeError:
            return np.ones(1, dtype="float64")
    if isinstance(value, Iterable):
        return np.asarray(list(value), dtype="float64")
    return np.ones(1, dtype="float64")

