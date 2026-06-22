import argparse
from pathlib import Path

import numpy as np
import pandas as pd


METHOD_NAMES = {
    "fixed_hrl_no_inner": "Fixed HRL w/o Inner",
    "fixed_hrl": "Fixed HRL",
    "full_controller": "Final E2E",
    "random_switch_matched_count": "Random Switch",
}


def _fmt(value, kind="float"):
    if pd.isna(value):
        return "missing"
    if kind == "pct":
        return f"{float(value) * 100:.2f}%"
    if kind == "int":
        return f"{int(round(float(value)))}"
    return f"{float(value):.3f}"


def _format_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "scenario" in out:
        out["method"] = out["scenario"].map(METHOD_NAMES).fillna(out["scenario"])
    for col in out.columns:
        if col in {"total_return", "annualized_return", "annualized_volatility", "max_drawdown", "daily_win_rate", "cumulative_inner_alpha", "mean_inner_alpha", "exit_gap"}:
            out[col] = out[col].map(lambda x: _fmt(x, "pct"))
        elif col in {"switch_count", "free_switch_count", "forced_switch_count", "n_free_decisions", "n_free_switches"}:
            out[col] = out[col].map(lambda x: _fmt(x, "int"))
        elif col in {"sharpe", "sortino", "calmar", "corr_exit_prob_switch_advantage", "corr_policy_logit_switch_advantage"}:
            out[col] = out[col].map(_fmt)
    return out


def _write_table(df: pd.DataFrame, output_dir: Path, name: str, caption: str):
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / f"{name}.csv", index=False)
    df.to_markdown(output_dir / f"{name}.md", index=False)
    tex = df.to_latex(index=False, escape=False, caption=caption)
    (output_dir / f"{name}.tex").write_text(tex, encoding="utf-8")


def main_from_paths(input_dir: Path, output_dir: Path):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    metrics_dir = input_dir / "metrics"
    specs = [
        ("stage_progression", "main_stage_progression", "Stage-wise checkpoint progression supports Claim A."),
        ("inference_ablation", "inference_ablation", "Inference ablation supports Claims B and C."),
        ("inner_alpha_summary", "inner_alpha_summary", "Inner alpha summary supports Claim B."),
        ("switch_alignment_summary", "switch_alignment_summary", "Switch alignment summary supports Claim E."),
        ("random_switch_comparison", "random_switch_summary", "Random switch comparison supports Claim D."),
    ]
    for source, target, caption in specs:
        path = metrics_dir / f"{source}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        _write_table(_format_table(df), output_dir, target, caption)


def parse_args():
    parser = argparse.ArgumentParser(description="Regenerate paper explanation tables.")
    parser.add_argument("--input_dir", default="paper_experiments_outputs/end_to_end_explain")
    parser.add_argument("--output_dir", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir / "tables"
    main_from_paths(input_dir, output_dir)


if __name__ == "__main__":
    main()

