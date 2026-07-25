#!/usr/bin/env python3
"""Rank staged CMTFlow seed-screening outputs against DeepTrader/DeepAries gates."""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd


STAGE_CURVE_PATTERNS = {
    "outer": "test_s2_*_NoInner.csv",
    "controller": "test_s5_ControllerNoInner.csv",
    "inner": "test_s3_AllModules.csv",
}


@dataclass(frozen=True)
class BaselineTarget:
    market: str
    deeptrader_return_pct: float
    deeparies_sharpe: float


def _float(row: dict, key: str) -> float:
    value = row.get(key, "")
    return float(value) if value not in ("", None) else float("nan")


def load_baseline_targets(manifest_path: Path, *, value_source: str = "table") -> Dict[str, BaselineTarget]:
    manifest_path = Path(manifest_path)
    return_col = f"{value_source}_return_pct"
    sharpe_col = f"{value_source}_sharpe"
    rows_by_market_method = {}
    with manifest_path.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            rows_by_market_method[(row["market"], row["method"].lower())] = row

    markets = sorted({market for market, _method in rows_by_market_method})
    targets = {}
    for market in markets:
        deeptrader = rows_by_market_method.get((market, "deeptrader"))
        deeparies = rows_by_market_method.get((market, "deeparies"))
        if deeptrader is None or deeparies is None:
            continue
        targets[market] = BaselineTarget(
            market=market,
            deeptrader_return_pct=_float(deeptrader, return_col),
            deeparies_sharpe=_float(deeparies, sharpe_col),
        )
    return targets


def compute_curve_metrics(curve_path: Path) -> dict:
    df = pd.read_csv(curve_path)
    if "value" not in df.columns:
        raise ValueError(f"{curve_path} must contain a value column")
    values = pd.to_numeric(df["value"], errors="coerce").dropna()
    if values.empty:
        raise ValueError(f"{curve_path} contains no numeric values")
    returns = values.pct_change().fillna(0.0)
    total_ret = float((values.iloc[-1] - values.iloc[0]) / values.iloc[0])
    ann_ret = float(returns.mean() * 252.0)
    ann_vol = float(returns.std(ddof=1) * math.sqrt(252.0)) if len(returns) > 1 else 0.0
    sharpe = float(ann_ret / (ann_vol + 1e-8)) if ann_vol > 1e-8 else 0.0
    roll_max = values.cummax()
    max_dd = float(((roll_max - values) / roll_max).max())
    cr = float(ann_ret / max_dd) if max_dd > 1e-8 else float("inf")
    return {
        "return_pct": total_ret * 100.0,
        "sharpe": sharpe,
        "maxdd_pct": max_dd * 100.0,
        "cr": cr,
        "ann_return_pct": ann_ret * 100.0,
        "ann_vol_pct": ann_vol * 100.0,
    }


def _seed_from_dir(seed_dir: Path) -> int:
    name = seed_dir.name
    if not name.startswith("seed_"):
        raise ValueError(f"Unexpected seed directory: {seed_dir}")
    return int(name.split("_", 1)[1])


def _stage_curve(seed_dir: Path, stage: str) -> Path | None:
    pattern = STAGE_CURVE_PATTERNS[stage]
    matches = sorted(seed_dir.glob(pattern))
    return matches[0] if matches else None


def _iter_seed_dirs(run_root: Path) -> Iterable[tuple[str, Path]]:
    for market_dir in sorted(Path(run_root).iterdir()):
        if not market_dir.is_dir():
            continue
        ppo_dir = market_dir / "ppo"
        if not ppo_dir.is_dir():
            continue
        for seed_dir in sorted(ppo_dir.glob("seed_*")):
            if seed_dir.is_dir():
                yield market_dir.name, seed_dir


def rank_stage(
        *,
        run_root: Path,
        stage: str,
        baseline_manifest: Path,
        value_source: str = "table",
) -> List[dict]:
    if stage not in STAGE_CURVE_PATTERNS:
        raise ValueError(f"Unknown stage: {stage}")
    targets = load_baseline_targets(baseline_manifest, value_source=value_source)
    rows = []
    for market, seed_dir in _iter_seed_dirs(Path(run_root)):
        if market not in targets:
            continue
        curve_path = _stage_curve(seed_dir, stage)
        if curve_path is None:
            continue
        metrics = compute_curve_metrics(curve_path)
        target = targets[market]
        pass_return = metrics["return_pct"] > target.deeptrader_return_pct
        pass_sharpe = metrics["sharpe"] > target.deeparies_sharpe
        checkpoint = seed_dir / "checkpoints" / "best_model.pth"
        rows.append({
            "stage": stage,
            "market": market,
            "seed": _seed_from_dir(seed_dir),
            "checkpoint": str(checkpoint),
            "curve_path": str(curve_path),
            "return_pct": metrics["return_pct"],
            "sharpe": metrics["sharpe"],
            "maxdd_pct": metrics["maxdd_pct"],
            "cr": metrics["cr"],
            "deeptrader_return_pct": target.deeptrader_return_pct,
            "deeparies_sharpe": target.deeparies_sharpe,
            "pass_deeptrader_return": pass_return,
            "pass_deeparies_sharpe": pass_sharpe,
            "pass_target": pass_return and pass_sharpe,
        })
    return sorted(rows, key=lambda r: (not r["pass_target"], -float(r["return_pct"]), -float(r["sharpe"])))


def write_rank_csv(rows: List[dict], output_path: Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "stage",
        "market",
        "seed",
        "checkpoint",
        "curve_path",
        "return_pct",
        "sharpe",
        "maxdd_pct",
        "cr",
        "deeptrader_return_pct",
        "deeparies_sharpe",
        "pass_deeptrader_return",
        "pass_deeparies_sharpe",
        "pass_target",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_root", required=True)
    parser.add_argument("--stage", choices=sorted(STAGE_CURVE_PATTERNS), required=True)
    parser.add_argument(
        "--baseline_manifest",
        default="paper_experiments_outputs/baseline_matched/manifest/baseline_sources.csv",
    )
    parser.add_argument("--value_source", choices=["table", "recomputed"], default="table")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = rank_stage(
        run_root=Path(args.run_root),
        stage=args.stage,
        baseline_manifest=Path(args.baseline_manifest),
        value_source=args.value_source,
    )
    write_rank_csv(rows, Path(args.output))
    print(f"Wrote {len(rows)} rows to {args.output}", flush=True)


if __name__ == "__main__":
    main()
