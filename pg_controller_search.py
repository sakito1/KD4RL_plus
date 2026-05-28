"""Small orchestration helper for PG controller validation searches."""

import argparse
import json
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PYTHON = "/home/tongwenxuan/conda/envs/xuangu/bin/python"


def fmt_float(value):
    text = f"{float(value):g}"
    return text.replace(".", "p").replace("-", "m")


def default_configs():
    base = {
        "lambda_min": 100,
        "lambda_max": 100,
        "constraint_loss_scale": 1,
        "grad_clip": 10,
        "schedule_penalty": 0.5,
        "near_max_penalty": 0.5,
        "late_hold_start": 0.7,
        "late_hold_loss_scale": 10,
        "pg_objective": "episode_sharpe",
        "reward_gamma": 1.0,
    }
    configs = []
    for bias, late_bias in [(4, 0), (4, 1), (4, 2), (5, 1), (6, 1), (6, 2)]:
        cfg = dict(base)
        cfg.update({
            "hidden_dim": 32,
            "fusion_hidden": 64,
            "lr": 1e-3,
            "ent_coef": 0.01,
            "constraint_logit_bias": bias,
            "late_hold_logit_bias": late_bias,
        })
        configs.append(cfg)
    for hidden_dim, fusion_hidden, lr, ent_coef, bias in [
        (16, 64, 1e-3, 0.01, 10),
        (64, 128, 5e-4, 0.01, 10),
        (32, 64, 5e-4, 0.02, 8),
        (64, 128, 1e-3, 0.02, 8),
    ]:
        cfg = dict(base)
        cfg.update({
            "hidden_dim": hidden_dim,
            "fusion_hidden": fusion_hidden,
            "lr": lr,
            "ent_coef": ent_coef,
            "constraint_logit_bias": bias,
            "late_hold_logit_bias": 1,
        })
        configs.append(cfg)
    return configs


def run_id(stage, market, episodes, cfg):
    pieces = [
        "pg", stage, market, f"ep{episodes}",
        f"h{cfg['hidden_dim']}", f"f{cfg['fusion_hidden']}",
        f"lr{fmt_float(cfg['lr'])}", f"ent{fmt_float(cfg['ent_coef'])}",
        f"bias{fmt_float(cfg['constraint_logit_bias'])}",
        f"late{fmt_float(cfg['late_hold_logit_bias'])}",
        f"lm{fmt_float(cfg['lambda_min'])}",
        f"lx{fmt_float(cfg['lambda_max'])}",
    ]
    return "_".join(pieces)


def run_one(stage, market, mode, episodes, cfg, validation_only, rerun):
    rid = run_id(stage, market, episodes, cfg)
    out_dir = ROOT / "results" / "pg_controller" / market / rid
    summary_path = out_dir / "summary.json"
    if summary_path.exists() and not rerun:
        with open(summary_path) as file:
            return json.load(file)
    cmd = [
        PYTHON, "pg_controller_experiment.py",
        "--market", market,
        "--mode", mode,
        "--episodes", str(episodes),
        "--run-id", rid,
        "--hidden-dim", str(cfg["hidden_dim"]),
        "--fusion-hidden", str(cfg["fusion_hidden"]),
        "--lr", str(cfg["lr"]),
        "--lambda-min", str(cfg["lambda_min"]),
        "--lambda-max", str(cfg["lambda_max"]),
        "--ent-coef", str(cfg["ent_coef"]),
        "--constraint-loss-scale", str(cfg["constraint_loss_scale"]),
        "--constraint-logit-bias", str(cfg["constraint_logit_bias"]),
        "--late-hold-start", str(cfg["late_hold_start"]),
        "--late-hold-logit-bias", str(cfg["late_hold_logit_bias"]),
        "--late-hold-loss-scale", str(cfg["late_hold_loss_scale"]),
        "--grad-clip", str(cfg["grad_clip"]),
        "--schedule-penalty", str(cfg["schedule_penalty"]),
        "--near-max-penalty", str(cfg["near_max_penalty"]),
        "--pg-objective", str(cfg["pg_objective"]),
        "--reward-gamma", str(cfg["reward_gamma"]),
    ]
    if validation_only:
        cmd.append("--validation-only")
    subprocess.run(cmd, cwd=ROOT, check=True)
    with open(summary_path) as file:
        return json.load(file)


def score(summary):
    val = summary["best_validation"]
    violations = val.get("early_violation_count", 0) + val.get("long_violation_count", 0)
    scheduled = val.get("scheduled_switch_rate", 0.0)
    near_max = val.get("near_max_switch_rate", 0.0)
    return (
        -violations,
        -scheduled,
        -near_max,
        summary.get("best_validation_objective", val["sharpe"]),
        val["sharpe"],
        val["total_ret"],
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--markets", nargs="+", default=["nas100", "sh"])
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--stage", default="search1")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--final-test", action="store_true")
    parser.add_argument("--rerun", action="store_true")
    args = parser.parse_args()

    configs = default_configs()
    if args.max_runs:
        configs = configs[:args.max_runs]
    all_summaries = []
    for market in args.markets:
        market_summaries = []
        for cfg in configs:
            summary = run_one(
                args.stage,
                market,
                args.mode,
                args.episodes,
                cfg,
                validation_only=not args.final_test,
                rerun=args.rerun,
            )
            val = summary["best_validation"]
            print(
                f"{summary['run_id']} {market} "
                f"obj={summary.get('best_validation_objective')} "
                f"sharpe={val['sharpe']:.4f} ret={val['total_ret']:.4f} "
                f"switch={val['switch_count']} "
                f"early={val['early_violation_count']} long={val['long_violation_count']} "
                f"sched={val.get('scheduled_switch_rate', 0.0):.2f} "
                f"near_max={val.get('near_max_switch_rate', 0.0):.2f}"
            )
            market_summaries.append(summary)
            all_summaries.append(summary)
        best = sorted(market_summaries, key=score, reverse=True)[0]
        val = best["best_validation"]
        print(
            f"BEST {market}: {best['run_id']} "
            f"obj={best.get('best_validation_objective')} "
            f"sharpe={val['sharpe']:.4f} ret={val['total_ret']:.4f} "
            f"switch={val['switch_count']} "
            f"early={val['early_violation_count']} long={val['long_violation_count']} "
            f"sched={val.get('scheduled_switch_rate', 0.0):.2f} "
            f"near_max={val.get('near_max_switch_rate', 0.0):.2f}"
        )

    report_path = ROOT / "results" / "pg_controller" / f"{args.stage}_summary.json"
    os.makedirs(report_path.parent, exist_ok=True)
    with open(report_path, "w") as file:
        json.dump(all_summaries, file, indent=2)


if __name__ == "__main__":
    main()
