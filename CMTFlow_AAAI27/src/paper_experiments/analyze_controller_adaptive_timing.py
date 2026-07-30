"""Statistical validation of the Controller's learned switch/hold timing."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
import numpy as np
import pandas as pd
import scipy.stats
from sklearn.metrics import (
    balanced_accuracy_score,
    matthews_corrcoef,
    roc_auc_score,
)

from paper_experiments.analyze_inner_outer_statistical_validation import (
    _block_mean_interval as block_mean_interval,
)
from paper_experiments.analyze_inner_outer_statistical_validation import (
    benjamini_hochberg,
    circular_block_bootstrap,
    newey_west_mean_test,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_curve(value: str) -> np.ndarray:
    """Parse and validate one normalized counterfactual wealth curve."""
    try:
        parsed = np.asarray(json.loads(value), dtype="float64")
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("counterfactual curve is not valid JSON") from exc
    if parsed.ndim != 1 or len(parsed) < 2 or not np.isfinite(parsed).all():
        raise ValueError("counterfactual curve must be a finite vector")
    if not np.isclose(parsed[0], 1.0, atol=1e-7):
        raise ValueError("counterfactual curve must start at one")
    return parsed


def parse_controller_decisions(
    actions: pd.DataFrame,
    *,
    max_horizon: int = 30,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Keep learned free decisions and attach validated counterfactual curves."""
    required = {
        "date",
        "step",
        "decision_type",
        "is_switch",
        "is_free_switch",
        "duration_before_decision",
        "exit_prob",
        "policy_logit",
        f"hold_curve_{max_horizon}",
        f"switch_curve_{max_horizon}",
    }
    missing = sorted(required.difference(actions.columns))
    if missing:
        raise ValueError(f"action trace is missing columns: {missing}")

    frame = actions.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values(["date", "step"]).reset_index(drop=True)
    if frame["date"].duplicated().any() or frame["step"].duplicated().any():
        raise ValueError("free-decision trace has duplicate dates or steps")

    free = frame.loc[frame["decision_type"] == "free_decision"].copy()
    free["action"] = pd.to_numeric(
        free["is_switch"], errors="raise"
    ).astype(int)
    free_switch = pd.to_numeric(
        free["is_free_switch"], errors="raise"
    ).astype(int)
    if not np.array_equal(free["action"].to_numpy(), free_switch.to_numpy()):
        raise ValueError("action flags disagree on free decisions")
    if not free["action"].isin([0, 1]).all():
        raise ValueError("action flags must be binary")

    duration = pd.to_numeric(
        free["duration_before_decision"], errors="raise"
    ).astype(int)
    if ((duration < 1) | (duration >= max_horizon)).any():
        raise ValueError("duration is outside the adaptive-horizon range")
    free["duration_before_decision"] = duration
    free["adaptive_horizon"] = (max_horizon - duration).clip(lower=1)

    hold_curves: list[np.ndarray] = []
    switch_curves: list[np.ndarray] = []
    invalid = 0
    for _, row in free.iterrows():
        try:
            hold = parse_curve(row[f"hold_curve_{max_horizon}"])
            switch = parse_curve(row[f"switch_curve_{max_horizon}"])
            horizon = int(row["adaptive_horizon"])
            if len(hold) <= horizon or len(switch) <= horizon:
                raise ValueError("counterfactual curve is shorter than horizon")
        except (TypeError, ValueError, json.JSONDecodeError):
            invalid += 1
            hold = switch = np.asarray([], dtype="float64")
        hold_curves.append(hold)
        switch_curves.append(switch)
    if invalid:
        raise ValueError(f"{invalid} invalid counterfactual curve rows")

    free["hold_curve"] = hold_curves
    free["switch_curve"] = switch_curves
    free = free.reset_index(drop=True)
    audit = {
        "input_rows": int(len(actions)),
        "free_decisions": int(len(free)),
        "valid_free_decisions": int(len(free)),
        "invalid_curve_rows": int(invalid),
    }
    return free, audit


def max_drawdown(curve: np.ndarray) -> float:
    """Return maximum peak-to-trough drawdown as a non-negative fraction."""
    peaks = np.maximum.accumulate(curve)
    return float(np.max((peaks - curve) / np.maximum(peaks, 1e-12)))


def compute_adaptive_outcomes(decisions: pd.DataFrame) -> pd.DataFrame:
    """Score each chosen action against its unchosen counterfactual."""
    rows = []
    for row in decisions.itertuples(index=False):
        horizon = int(row.adaptive_horizon)
        hold = row.hold_curve[: horizon + 1]
        switch = row.switch_curve[: horizon + 1]
        cumulative_log_advantage = float(
            np.log(max(switch[-1], 1e-12))
            - np.log(max(hold[-1], 1e-12))
        )
        return_advantage = cumulative_log_advantage / horizon
        mdd_advantage = max_drawdown(hold) - max_drawdown(switch)
        direction = 1.0 if int(row.action) == 1 else -1.0
        rows.append(
            {
                **row._asdict(),
                "cumulative_return_advantage_log": cumulative_log_advantage,
                "return_advantage_log_per_day": return_advantage,
                "return_advantage_bp_day": return_advantage * 10000.0,
                "mdd_advantage": mdd_advantage,
                "return_decision_value_log_per_day": direction
                * return_advantage,
                "return_decision_value_bp_day": direction
                * return_advantage
                * 10000.0,
                "mdd_decision_value": direction * mdd_advantage,
            }
        )
    return pd.DataFrame(rows)


def compute_horizon_outcomes(
    decisions: pd.DataFrame,
    horizons: Sequence[int] = (5, 10, 20, 30),
) -> pd.DataFrame:
    """Expand decisions over fixed horizons for sensitivity checks only."""
    expanded = []
    for _, row in decisions.iterrows():
        for horizon_value in horizons:
            horizon = int(horizon_value)
            if len(row["hold_curve"]) <= horizon:
                continue
            item = row.copy()
            item["adaptive_horizon"] = horizon
            item["sensitivity_horizon"] = horizon
            expanded.append(item)
    if not expanded:
        return pd.DataFrame()
    return compute_adaptive_outcomes(
        pd.DataFrame(expanded).reset_index(drop=True)
    )


def summarize_decision_value(
    outcomes: pd.DataFrame,
    *,
    block_length: int,
    bootstrap_reps: int,
    seed: int,
) -> dict[str, float | int]:
    """Summarize whether chosen actions beat their unchosen alternatives."""
    ret = outcomes["return_decision_value_bp_day"].to_numpy(dtype="float64")
    mdd = outcomes["mdd_decision_value"].to_numpy(dtype="float64")
    ret_hac = newey_west_mean_test(ret, maxlags=5)
    mdd_hac = newey_west_mean_test(mdd, maxlags=5)
    ret_ci = block_mean_interval(
        ret,
        block_length=min(block_length, len(ret)),
        reps=bootstrap_reps,
        seed=seed,
    )
    mdd_ci = block_mean_interval(
        mdd,
        block_length=min(block_length, len(mdd)),
        reps=bootstrap_reps,
        seed=seed + 1,
    )
    horizon = outcomes["adaptive_horizon"].to_numpy(dtype="float64")
    action = outcomes["action"].to_numpy(dtype=int)
    return {
        "free_decisions": int(len(outcomes)),
        "free_switches": int(action.sum()),
        "switch_rate": float(action.mean()),
        "mean_adaptive_horizon": float(horizon.mean()),
        "median_adaptive_horizon": float(np.median(horizon)),
        "mean_return_decision_value_bp_day": float(ret.mean()),
        "median_return_decision_value_bp_day": float(np.median(ret)),
        "return_value_ci_low": ret_ci[0],
        "return_value_ci_high": ret_ci[1],
        "return_value_nw_t": ret_hac["t_stat"],
        "return_value_nw_p": ret_hac["p_value"],
        "positive_return_decision_ratio": float(np.mean(ret > 0)),
        "mean_mdd_decision_value": float(mdd.mean()),
        "mdd_value_ci_low": mdd_ci[0],
        "mdd_value_ci_high": mdd_ci[1],
        "mdd_value_nw_t": mdd_hac["t_stat"],
        "mdd_value_nw_p": mdd_hac["p_value"],
        "positive_mdd_decision_ratio": float(np.mean(mdd > 0)),
    }


def summarize_switch_hold(
    outcomes: pd.DataFrame,
    *,
    block_length: int,
    bootstrap_reps: int,
    seed: int,
) -> pd.DataFrame:
    """Decompose switch and hold outcomes without discarding either action."""
    rows = []
    for action_value, label in ((1, "switch"), (0, "hold")):
        group = outcomes.loc[outcomes["action"] == action_value]
        advantage = group["return_advantage_bp_day"].to_numpy(
            dtype="float64"
        )
        mdd = group["mdd_advantage"].to_numpy(dtype="float64")
        direction = 1.0 if action_value == 1 else -1.0
        ci = block_mean_interval(
            advantage,
            block_length=min(block_length, len(advantage)),
            reps=bootstrap_reps,
            seed=seed + action_value,
        )
        rows.append(
            {
                "action": label,
                "n": int(len(group)),
                "mean_return_advantage_bp_day": float(advantage.mean()),
                "return_advantage_ci_low": ci[0],
                "return_advantage_ci_high": ci[1],
                "favorable_return_ratio": float(
                    np.mean(direction * advantage > 0)
                ),
                "mean_mdd_advantage": float(mdd.mean()),
                "favorable_mdd_ratio": float(np.mean(direction * mdd > 0)),
            }
        )
    return pd.DataFrame(rows)


def _safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or np.unique(x).size < 2 or np.unique(y).size < 2:
        return np.nan
    return float(scipy.stats.spearmanr(x, y).statistic)


def summarize_exit_probability(
    outcomes: pd.DataFrame,
    *,
    block_length: int,
    bootstrap_reps: int,
    seed: int,
) -> tuple[dict[str, float | int], pd.DataFrame]:
    """Assess whether exit probability ranks switch-vs-hold advantage."""
    clean = outcomes.dropna(
        subset=["exit_prob", "policy_logit", "return_advantage_log_per_day"]
    ).copy()
    clean["return_advantage_bp_day"] = (
        clean["return_advantage_log_per_day"] * 10000.0
    )
    if "mdd_advantage" not in clean:
        clean["mdd_advantage"] = np.nan
    target = (clean["return_advantage_log_per_day"] > 0).astype(int)
    exit_prob = clean["exit_prob"].to_numpy(dtype="float64")
    policy_logit = clean["policy_logit"].to_numpy(dtype="float64")
    advantage = clean["return_advantage_log_per_day"].to_numpy(
        dtype="float64"
    )
    exit_rho = _safe_spearman(exit_prob, advantage)
    logit_rho = _safe_spearman(policy_logit, advantage)
    auroc = (
        float(roc_auc_score(target, exit_prob))
        if target.nunique() == 2
        else np.nan
    )
    predicted = clean["action"].astype(int)

    probability_quintile = pd.qcut(
        clean["exit_prob"].rank(method="first"),
        q=5,
        labels=["Q1", "Q2", "Q3", "Q4", "Q5"],
    )
    clean["probability_quintile"] = probability_quintile
    quintiles = (
        clean.groupby("probability_quintile", observed=False)
        .agg(
            n=("action", "size"),
            mean_exit_prob=("exit_prob", "mean"),
            mean_return_advantage_bp_day=(
                "return_advantage_bp_day",
                "mean",
            ),
            mean_mdd_advantage=("mdd_advantage", "mean"),
            switch_rate=("action", "mean"),
        )
        .reset_index()
    )
    q1 = quintiles.loc[
        quintiles["probability_quintile"] == "Q1",
        "mean_return_advantage_bp_day",
    ].iloc[0]
    q5 = quintiles.loc[
        quintiles["probability_quintile"] == "Q5",
        "mean_return_advantage_bp_day",
    ].iloc[0]

    def ranking_stats(
        prob_sample: np.ndarray,
        advantage_sample: np.ndarray,
    ) -> np.ndarray:
        ranks = scipy.stats.rankdata(prob_sample, method="ordinal")
        low = advantage_sample[ranks <= len(ranks) / 5.0]
        high = advantage_sample[ranks > 4.0 * len(ranks) / 5.0]
        return np.asarray(
            [
                _safe_spearman(prob_sample, advantage_sample),
                float((high.mean() - low.mean()) * 10000.0),
            ]
        )

    draws = circular_block_bootstrap(
        [exit_prob, advantage],
        ranking_stats,
        block_length=min(block_length, len(clean)),
        reps=bootstrap_reps,
        seed=seed,
    )
    rho_ci = np.nanquantile(draws[:, 0], [0.025, 0.975])
    spread_ci = np.nanquantile(draws[:, 1], [0.025, 0.975])
    return {
        "n": int(len(clean)),
        "spearman_exit_prob_advantage": exit_rho,
        "spearman_exit_prob_ci_low": float(rho_ci[0]),
        "spearman_exit_prob_ci_high": float(rho_ci[1]),
        "spearman_policy_logit_advantage": logit_rho,
        "auroc_positive_advantage": auroc,
        "balanced_accuracy": float(
            balanced_accuracy_score(target, predicted)
        ),
        "matthews_correlation": float(
            matthews_corrcoef(target, predicted)
        ),
        "q5_minus_q1_return_advantage_bp_day": float(q5 - q1),
        "q5_minus_q1_ci_low": float(spread_ci[0]),
        "q5_minus_q1_ci_high": float(spread_ci[1]),
    }, quintiles


def matched_action_permutation(
    outcomes: pd.DataFrame,
    *,
    reps: int,
    seed: int,
) -> tuple[dict[str, float | int], pd.DataFrame]:
    """Permute actions within duration/volatility strata."""
    rng = np.random.default_rng(seed)
    frame = outcomes.reset_index(drop=True).copy()
    advantage_bp = (
        frame["return_advantage_bp_day"].to_numpy(dtype="float64")
        if "return_advantage_bp_day" in frame
        else frame["return_advantage_log_per_day"].to_numpy(
            dtype="float64"
        )
        * 10000.0
    )
    action = frame["action"].to_numpy(dtype=int)
    direction = 2.0 * action - 1.0
    frame["return_decision_value_bp_day"] = direction * advantage_bp
    frame["mdd_decision_value"] = (
        direction * frame["mdd_advantage"].to_numpy(dtype="float64")
    )
    groups = [
        np.asarray(index, dtype=int)
        for index in frame.groupby(
            ["duration_stratum", "volatility_stratum"],
            observed=False,
            dropna=False,
        ).groups.values()
    ]
    observed_return = float(
        frame["return_decision_value_bp_day"].mean()
    )
    observed_mdd = float(frame["mdd_decision_value"].mean())
    draws = []
    invalid = 0
    for _ in range(int(reps)):
        permuted = action.copy()
        for index in groups:
            before = int(permuted[index].sum())
            permuted[index] = rng.permutation(permuted[index])
            invalid += int(int(permuted[index].sum()) != before)
        permuted_direction = 2.0 * permuted - 1.0
        draws.append(
            {
                "return_value_bp_day": float(
                    np.mean(permuted_direction * advantage_bp)
                ),
                "mdd_value": float(
                    np.mean(
                        permuted_direction
                        * frame["mdd_advantage"].to_numpy(dtype="float64")
                    )
                ),
            }
        )
    draw_frame = pd.DataFrame(draws)
    summary = {
        "placebo_reps": int(reps),
        "observed_return_value_bp_day": observed_return,
        "placebo_mean_return_value_bp_day": float(
            draw_frame["return_value_bp_day"].mean()
        ),
        "return_permutation_p": float(
            (
                1
                + np.sum(
                    draw_frame["return_value_bp_day"] >= observed_return
                )
            )
            / (reps + 1)
        ),
        "observed_mdd_value": observed_mdd,
        "placebo_mean_mdd_value": float(draw_frame["mdd_value"].mean()),
        "mdd_permutation_p": float(
            (1 + np.sum(draw_frame["mdd_value"] >= observed_mdd))
            / (reps + 1)
        ),
        "invalid_permutations": int(invalid),
    }
    return summary, draw_frame


def summarize_holding_spells(decisions: pd.DataFrame) -> pd.DataFrame:
    """Return completed learned holding spells (free switch decisions)."""
    action = pd.to_numeric(decisions["action"], errors="raise").astype(int)
    switched = decisions.loc[action == 1].copy()
    return pd.DataFrame(
        {
            "end_date": pd.to_datetime(switched["date"]).to_numpy(),
            "end_step": pd.to_numeric(
                switched["step"], errors="raise"
            ).astype(int).to_numpy(),
            "completed_duration": pd.to_numeric(
                switched["duration_before_decision"], errors="raise"
            ).astype(int).to_numpy(),
        }
    )


def holding_duration_hazard(decisions: pd.DataFrame) -> pd.DataFrame:
    """Empirical switch hazard conditional on reaching each duration."""
    frame = decisions.loc[:, ["duration_before_decision", "action"]].copy()
    frame["duration_before_decision"] = pd.to_numeric(
        frame["duration_before_decision"], errors="raise"
    ).astype(int)
    frame["action"] = pd.to_numeric(
        frame["action"], errors="raise"
    ).astype(int)
    result = (
        frame.groupby("duration_before_decision", as_index=False)
        .agg(at_risk=("action", "size"), switches=("action", "sum"))
        .rename(columns={"duration_before_decision": "duration"})
    )
    result["hazard"] = result["switches"] / result["at_risk"]
    return result


def tercile_labels(values: pd.Series) -> pd.Series:
    """Assign stable low/mid/high labels, including tied-cut fallback."""
    numeric = pd.to_numeric(values, errors="coerce")
    finite = numeric[np.isfinite(numeric)]
    result = pd.Series(pd.NA, index=values.index, dtype="object")
    if finite.empty:
        return result
    lower, upper = finite.quantile([1.0 / 3.0, 2.0 / 3.0]).to_numpy()
    if np.isclose(lower, upper):
        result.loc[np.isfinite(numeric)] = "mid"
        return result
    result.loc[numeric <= lower] = "low"
    result.loc[(numeric > lower) & (numeric <= upper)] = "mid"
    result.loc[numeric > upper] = "high"
    return result


def attach_observable_states(
    decisions: pd.DataFrame,
    portfolio: pd.DataFrame,
    equal_weight_market_return: pd.Series,
    *,
    lookback: int = 20,
) -> pd.DataFrame:
    """Attach only state information observed through each decision date."""
    values = portfolio.loc[:, ["date", "portfolio_value"]].copy()
    values["date"] = pd.to_datetime(values["date"])
    values = values.sort_values("date").set_index("date")
    market = equal_weight_market_return.copy().sort_index().astype("float64")
    market.index = pd.to_datetime(market.index)
    rows = []
    for row in decisions.itertuples(index=False):
        date = pd.Timestamp(row.date)
        history = values.loc[:date, "portfolio_value"].tail(lookback + 1)
        market_history = market.loc[:date].tail(lookback)
        pre_return = (
            float(history.iloc[-1] / history.iloc[0] - 1.0)
            if len(history) == lookback + 1
            else np.nan
        )
        pre_drawdown = (
            max_drawdown(history.to_numpy(dtype="float64"))
            if len(history) == lookback + 1
            else np.nan
        )
        market_volatility = (
            float(market_history.std(ddof=1) * np.sqrt(252.0))
            if len(market_history) == lookback
            else np.nan
        )
        rows.append(
            {
                "pre_return_20": pre_return,
                "pre_drawdown_20": pre_drawdown,
                "market_volatility_20": market_volatility,
            }
        )
    result = decisions.reset_index(drop=True).join(pd.DataFrame(rows))
    result["duration_stratum"] = tercile_labels(
        result["duration_before_decision"]
    )
    result["volatility_stratum"] = tercile_labels(
        result["market_volatility_20"]
    )
    result["pre_return_stratum"] = tercile_labels(result["pre_return_20"])
    result["pre_drawdown_stratum"] = tercile_labels(
        result["pre_drawdown_20"]
    )
    return result


def fit_switch_state_model(decisions: pd.DataFrame) -> pd.DataFrame:
    """Fit an explanatory HAC-logit model for switch behavior."""
    import statsmodels.api as sm

    columns = [
        "duration_before_decision",
        "pre_return_20",
        "pre_drawdown_20",
        "market_volatility_20",
    ]
    clean = decisions.dropna(subset=columns + ["action"]).copy()
    design = clean[columns].astype("float64")
    design = (design - design.mean()) / design.std(ddof=1).replace(0.0, 1.0)
    design["duration_squared"] = (
        design["duration_before_decision"] ** 2
    )
    design = sm.add_constant(design)
    fit = sm.Logit(clean["action"].astype(int), design).fit(
        disp=False,
        cov_type="HAC",
        cov_kwds={"maxlags": min(5, max(len(clean) - 1, 0))},
    )
    interval = fit.conf_int()
    return pd.DataFrame(
        {
            "term": fit.params.index,
            "coefficient": fit.params.to_numpy(),
            "odds_ratio": np.exp(fit.params.to_numpy()),
            "std_error": fit.bse.to_numpy(),
            "p_value": fit.pvalues.to_numpy(),
            "ci_low": np.exp(interval[0].to_numpy()),
            "ci_high": np.exp(interval[1].to_numpy()),
        }
    )


def summarize_state_conditional_value(
    decisions: pd.DataFrame,
) -> pd.DataFrame:
    """Describe action rates and counterfactual value by observable state."""
    specs = {
        "duration": "duration_stratum",
        "pre_return": "pre_return_stratum",
        "pre_drawdown": "pre_drawdown_stratum",
        "market_volatility": "volatility_stratum",
    }
    rows = []
    for state_name, column in specs.items():
        for level, group in decisions.groupby(
            column, observed=False, dropna=False
        ):
            rows.append(
                {
                    "state": state_name,
                    "level": str(level),
                    "n": int(len(group)),
                    "switch_rate": float(group["action"].mean()),
                    "mean_exit_prob": float(group["exit_prob"].mean()),
                    "mean_return_decision_value_bp_day": float(
                        group["return_decision_value_bp_day"].mean()
                    ),
                    "positive_return_decision_ratio": float(
                        (group["return_decision_value_bp_day"] > 0).mean()
                    ),
                    "mean_mdd_decision_value": float(
                        group["mdd_decision_value"].mean()
                    ),
                }
            )
    return pd.DataFrame(rows)


def load_equal_weight_market_returns(
    prices_root: Path,
    market: str,
) -> pd.Series:
    """Load the experiment universe and construct equal-weight daily returns."""
    path = prices_root / market / f"{market}_data.csv"
    frame = pd.read_csv(path, usecols=["date", "tic", "adjclose"])
    frame["date"] = pd.to_datetime(frame["date"])
    panel = (
        frame.pivot(index="date", columns="tic", values="adjclose")
        .sort_index()
        .ffill()
    )
    return panel.pct_change(fill_method=None).mean(axis=1)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_frame(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _plot_outputs(
    decision_table: pd.DataFrame,
    quintiles: pd.DataFrame,
    permutation_draws: dict[str, pd.DataFrame],
    hazard: pd.DataFrame,
    output_dir: Path,
) -> None:
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    colors = {"nas": "#0072B2", "sh": "#D55E00"}

    figure, axis = plt.subplots(figsize=(7.2, 3.8))
    positions = np.arange(len(decision_table))
    means = decision_table["mean_return_decision_value_bp_day"].to_numpy()
    lower = means - decision_table["return_value_ci_low"].to_numpy()
    upper = decision_table["return_value_ci_high"].to_numpy() - means
    axis.errorbar(
        means,
        positions,
        xerr=np.vstack([lower, upper]),
        fmt="o",
        capsize=4,
        color="#0072B2",
    )
    axis.axvline(0.0, color="black", linewidth=1, linestyle="--")
    axis.set_yticks(positions, decision_table["market_label"])
    axis.set_xlabel("Chosen-action value (bp/day; 95% block CI)")
    axis.set_title("Adaptive-horizon Controller decision value")
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(
            figure_dir / f"decision_value_forest.{suffix}", dpi=180
        )
    plt.close(figure)

    figure, axes = plt.subplots(
        1, max(1, len(permutation_draws)), figsize=(8.5, 3.6), squeeze=False
    )
    for axis, (market, draws) in zip(axes[0], permutation_draws.items()):
        axis.hist(
            draws["return_value_bp_day"],
            bins=35,
            color=colors.get(market, "#009E73"),
            alpha=0.75,
        )
        observed = decision_table.loc[
            decision_table["market"] == market,
            "mean_return_decision_value_bp_day",
        ].iloc[0]
        axis.axvline(0.0, color="black", linewidth=1, linestyle=":")
        axis.axvline(observed, color="#CC79A7", linewidth=2)
        axis.set_title(_market_label(market))
        axis.set_xlabel("Matched-placebo value (bp/day)")
    axes[0, 0].set_ylabel("Permutation count")
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(
            figure_dir / f"matched_action_placebo.{suffix}", dpi=180
        )
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7.2, 4.0))
    for market, group in quintiles.groupby("market"):
        axis.plot(
            group["probability_quintile"].astype(str),
            group["mean_return_advantage_bp_day"],
            marker="o",
            color=colors.get(market, "#009E73"),
            label=_market_label(market),
        )
    axis.axhline(0.0, color="black", linewidth=1, linestyle="--")
    axis.set_xlabel("Exit-probability quintile")
    axis.set_ylabel("Switch advantage (bp/day)")
    axis.set_title("Exit probability versus counterfactual switch advantage")
    axis.legend(frameon=False)
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(
            figure_dir / f"exit_probability_quintiles.{suffix}", dpi=180
        )
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7.2, 4.0))
    for market, group in hazard.groupby("market"):
        axis.plot(
            group["duration"],
            group["hazard"],
            marker="o",
            markersize=3,
            color=colors.get(market, "#009E73"),
            label=_market_label(market),
        )
    axis.axhline(0.0, color="black", linewidth=1, linestyle=":")
    axis.set_xlabel("Holding duration before decision (days)")
    axis.set_ylabel("Empirical switch hazard")
    axis.set_ylim(bottom=0)
    axis.set_title("Learned free-switch hazard")
    axis.legend(frameon=False)
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(
            figure_dir / f"holding_duration_hazard.{suffix}", dpi=180
        )
    plt.close(figure)


def _market_label(market: str) -> str:
    return "NASDAQ-100" if market == "nas" else "CSI-300"


def _claim_label(
    estimate: float,
    ci_low: float,
    adjusted_p: float,
) -> str:
    if estimate > 0 and ci_low > 0 and adjusted_p < 0.05:
        return "SUPPORTED"
    if estimate > 0:
        return "DESCRIPTIVE"
    return "NOT SUPPORTED"


def _write_report(
    output_dir: Path,
    decision: pd.DataFrame,
    decomposition: pd.DataFrame,
    ranking: pd.DataFrame,
    permutation: pd.DataFrame,
) -> None:
    lines = [
        "# Controller Adaptive-Timing Statistical Validation",
        "",
        "This analysis uses every learned free switch/hold decision and excludes "
        "fixed-day policy comparisons.",
        "",
        "Intervals quantify time-series uncertainty within the test period for "
        "one selected checkpoint. They do not measure cross-seed training "
        "uncertainty.",
        "",
        "## Claim audit",
        "",
    ]
    for row in decision.itertuples(index=False):
        label = _claim_label(
            row.mean_return_decision_value_bp_day,
            row.return_value_ci_low,
            row.return_value_p_adjusted,
        )
        lines.append(
            f"- **{label} — {_market_label(row.market)} chosen-action "
            f"return value:** {row.mean_return_decision_value_bp_day:.3f} "
            f"bp/day (95% block CI {row.return_value_ci_low:.3f}, "
            f"{row.return_value_ci_high:.3f}; adjusted HAC "
            f"p={row.return_value_p_adjusted:.4g})."
        )
        mdd_label = _claim_label(
            row.mean_mdd_decision_value,
            row.mdd_value_ci_low,
            row.mdd_value_p_adjusted,
        )
        lines.append(
            f"- **{mdd_label} — {_market_label(row.market)} drawdown "
            f"decision value:** {row.mean_mdd_decision_value * 100:.3f} pp "
            f"of MDD reduction (95% block CI "
            f"{row.mdd_value_ci_low * 100:.3f}, "
            f"{row.mdd_value_ci_high * 100:.3f} pp; adjusted HAC "
            f"p={row.mdd_value_p_adjusted:.4g})."
        )
        parts = decomposition.loc[decomposition["market"] == row.market]
        switch_part = parts.loc[parts["action"] == "switch"].iloc[0]
        hold_part = parts.loc[parts["action"] == "hold"].iloc[0]
        lines.append(
            f"- **DESCRIPTIVE — {_market_label(row.market)} action "
            f"decomposition:** switch dates have "
            f"{switch_part['mean_return_advantage_bp_day']:.3f} bp/day "
            f"switch-minus-hold advantage; hold dates have "
            f"{hold_part['mean_return_advantage_bp_day']:.3f} bp/day. "
            f"The latter contributes positively to chosen-action value only "
            f"when it is negative, because hold was selected."
        )
        rank = ranking.loc[ranking["market"] == row.market].iloc[0]
        rank_label = (
            "SUPPORTED"
            if rank["spearman_exit_prob_advantage"] > 0
            and rank["spearman_exit_prob_ci_low"] > 0
            else (
                "DESCRIPTIVE"
                if rank["spearman_exit_prob_advantage"] > 0
                else "NOT SUPPORTED"
            )
        )
        lines.append(
            f"- **{rank_label} — {_market_label(row.market)} exit-probability "
            f"ranking:** Spearman rho="
            f"{rank['spearman_exit_prob_advantage']:.3f} "
            f"(95% block CI {rank['spearman_exit_prob_ci_low']:.3f}, "
            f"{rank['spearman_exit_prob_ci_high']:.3f}); Q5−Q1="
            f"{rank['q5_minus_q1_return_advantage_bp_day']:.3f} bp/day."
        )
        perm = permutation.loc[permutation["market"] == row.market].iloc[0]
        perm_label = (
            "SUPPORTED"
            if perm["observed_return_value_bp_day"]
            > perm["placebo_mean_return_value_bp_day"]
            and perm["return_permutation_p_adjusted"] < 0.05
            else (
                "DESCRIPTIVE"
                if perm["observed_return_value_bp_day"]
                > perm["placebo_mean_return_value_bp_day"]
                else "NOT SUPPORTED"
            )
        )
        lines.append(
            f"- **{perm_label} — {_market_label(row.market)} matched timing "
            f"placebo:** observed {perm['observed_return_value_bp_day']:.3f} "
            f"versus placebo mean "
            f"{perm['placebo_mean_return_value_bp_day']:.3f} bp/day "
            f"(adjusted permutation p="
            f"{perm['return_permutation_p_adjusted']:.4g})."
        )
        mdd_perm_label = (
            "SUPPORTED"
            if perm["observed_mdd_value"] > perm["placebo_mean_mdd_value"]
            and perm["mdd_permutation_p_adjusted"] < 0.05
            else (
                "DESCRIPTIVE"
                if perm["observed_mdd_value"]
                > perm["placebo_mean_mdd_value"]
                else "NOT SUPPORTED"
            )
        )
        lines.append(
            f"- **{mdd_perm_label} — {_market_label(row.market)} matched "
            f"drawdown timing placebo:** observed MDD value "
            f"{perm['observed_mdd_value'] * 100:.3f} pp versus placebo mean "
            f"{perm['placebo_mean_mdd_value'] * 100:.3f} pp "
            f"(adjusted permutation p="
            f"{perm['mdd_permutation_p_adjusted']:.4g})."
        )
    lines.extend(
        [
            "",
            "State models and conditional-state tables are explanatory "
            "descriptions, not causal evidence. A positive action-value "
            "estimate means the realized action beat its same-date unchosen "
            "counterfactual over the adaptive remaining horizon.",
            "",
        ]
    )
    (output_dir / "CONTROLLER_ADAPTIVE_TIMING_STATISTICAL_VALIDATION.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def _parse_seed_mapping(values: Sequence[str]) -> dict[str, int]:
    mapping = {}
    for value in values:
        market, seed = value.split(":", 1)
        mapping[market] = int(seed)
    return mapping


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate learned Controller switch/hold timing."
    )
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--portfolio_trace_dir", type=Path)
    parser.add_argument("--prices_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--markets", nargs="+", default=["nas", "sh"])
    parser.add_argument("--seeds", nargs="+", default=["nas:49", "sh:90"])
    parser.add_argument("--max_horizon", type=int, default=30)
    parser.add_argument(
        "--sensitivity_horizons",
        nargs="+",
        type=int,
        default=[5, 10, 20, 30],
    )
    parser.add_argument("--block_length", type=int, default=30)
    parser.add_argument(
        "--block_length_sensitivity",
        nargs="+",
        type=int,
        default=[20, 40, 60],
    )
    parser.add_argument("--bootstrap_reps", type=int, default=10000)
    parser.add_argument("--placebo_reps", type=int, default=5000)
    parser.add_argument("--random_seed", type=int, default=20260727)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the complete Controller adaptive-timing analysis."""
    args = _build_parser().parse_args(argv)
    portfolio_dir = args.portfolio_trace_dir or args.input_dir
    output_dir = args.output_dir
    tables_dir = output_dir / "tables"
    decision_rows = []
    decomposition_frames = []
    ranking_rows = []
    quintile_frames = []
    permutation_rows = []
    permutation_draws: dict[str, pd.DataFrame] = {}
    spell_frames = []
    hazard_frames = []
    model_frames = []
    conditional_frames = []
    robustness_rows = []
    decision_audits = []
    input_records = []
    seed_mapping = _parse_seed_mapping(args.seeds)

    for market_position, market in enumerate(args.markets):
        seed = seed_mapping[market]
        stem = (
            f"{market}_seed{seed}_full_controller_"
            f"horizon{args.max_horizon}"
        )
        action_path = args.input_dir / f"{stem}_actions.csv"
        portfolio_path = portfolio_dir / f"{stem}_portfolio.csv"
        price_path = args.prices_root / market / f"{market}_data.csv"
        for kind, path in (
            ("actions", action_path),
            ("portfolio", portfolio_path),
            ("prices", price_path),
        ):
            input_records.append(
                {
                    "market": market,
                    "kind": kind,
                    "path": str(path.resolve()),
                    "sha256": _sha256(path),
                }
            )
        actions = pd.read_csv(action_path)
        decisions, audit = parse_controller_decisions(
            actions, max_horizon=args.max_horizon
        )
        outcomes = compute_adaptive_outcomes(decisions)
        portfolio = pd.read_csv(portfolio_path)
        market_return = load_equal_weight_market_returns(
            args.prices_root, market
        )
        outcomes = attach_observable_states(
            outcomes, portfolio, market_return, lookback=20
        )
        market_seed = args.random_seed + market_position * 1000
        decision = summarize_decision_value(
            outcomes,
            block_length=args.block_length,
            bootstrap_reps=args.bootstrap_reps,
            seed=market_seed,
        )
        decision.update(
            {
                "market": market,
                "market_label": _market_label(market),
                "seed": seed,
            }
        )
        decision_rows.append(decision)

        decomposition = summarize_switch_hold(
            outcomes,
            block_length=args.block_length,
            bootstrap_reps=args.bootstrap_reps,
            seed=market_seed + 10,
        )
        decomposition.insert(0, "market", market)
        decomposition_frames.append(decomposition)

        ranking, quintiles = summarize_exit_probability(
            outcomes,
            block_length=args.block_length,
            bootstrap_reps=args.bootstrap_reps,
            seed=market_seed + 20,
        )
        ranking.update({"market": market, "seed": seed})
        ranking_rows.append(ranking)
        quintiles.insert(0, "market", market)
        quintile_frames.append(quintiles)

        permutation, draws = matched_action_permutation(
            outcomes,
            reps=args.placebo_reps,
            seed=market_seed + 30,
        )
        permutation.update({"market": market, "seed": seed})
        permutation_rows.append(permutation)
        permutation_draws[market] = draws

        spells = summarize_holding_spells(outcomes)
        spells.insert(0, "market", market)
        spell_frames.append(spells)
        hazard = holding_duration_hazard(outcomes)
        hazard.insert(0, "market", market)
        hazard_frames.append(hazard)
        try:
            model = fit_switch_state_model(outcomes)
        except (ValueError, np.linalg.LinAlgError) as exc:
            model = pd.DataFrame(
                {
                    "term": ["MODEL_FAILURE"],
                    "error": [str(exc)],
                }
            )
        model.insert(0, "market", market)
        model_frames.append(model)
        conditional = summarize_state_conditional_value(outcomes)
        conditional.insert(0, "market", market)
        conditional_frames.append(conditional)

        sensitivity = compute_horizon_outcomes(
            decisions, args.sensitivity_horizons
        )
        for horizon, group in sensitivity.groupby("sensitivity_horizon"):
            summary = summarize_decision_value(
                group,
                block_length=min(args.block_length, len(group)),
                bootstrap_reps=args.bootstrap_reps,
                seed=market_seed + 100 + int(horizon),
            )
            robustness_rows.append(
                {
                    "market": market,
                    "sensitivity_type": "fixed_horizon",
                    "value": int(horizon),
                    **summary,
                }
            )
        for block_length in args.block_length_sensitivity:
            summary = summarize_decision_value(
                outcomes,
                block_length=min(block_length, len(outcomes)),
                bootstrap_reps=args.bootstrap_reps,
                seed=market_seed + 200 + int(block_length),
            )
            robustness_rows.append(
                {
                    "market": market,
                    "sensitivity_type": "block_length",
                    "value": int(block_length),
                    **summary,
                }
            )
        decision_audits.append(
            {
                "market": market,
                "seed": seed,
                **audit,
                "free_switches": int(outcomes["action"].sum()),
                "action_curve_invariants": True,
            }
        )

    decision_table = pd.DataFrame(decision_rows)
    decision_table["return_value_p_adjusted"] = benjamini_hochberg(
        decision_table["return_value_nw_p"].to_numpy()
    )
    decision_table["mdd_value_p_adjusted"] = benjamini_hochberg(
        decision_table["mdd_value_nw_p"].to_numpy()
    )
    decomposition_table = pd.concat(decomposition_frames, ignore_index=True)
    ranking_table = pd.DataFrame(ranking_rows)
    quintile_table = pd.concat(quintile_frames, ignore_index=True)
    permutation_table = pd.DataFrame(permutation_rows)
    permutation_table["return_permutation_p_adjusted"] = benjamini_hochberg(
        permutation_table["return_permutation_p"].to_numpy()
    )
    permutation_table["mdd_permutation_p_adjusted"] = benjamini_hochberg(
        permutation_table["mdd_permutation_p"].to_numpy()
    )
    spells_table = pd.concat(spell_frames, ignore_index=True)
    duration_distribution = (
        spells_table.groupby(["market", "completed_duration"], as_index=False)
        .size()
        .rename(columns={"size": "completed_spells"})
    )
    hazard_table = pd.concat(hazard_frames, ignore_index=True)
    state_model = pd.concat(model_frames, ignore_index=True)
    if "p_value" in state_model:
        state_model["p_value_adjusted"] = np.nan
        for market, index in state_model.groupby("market").groups.items():
            values = state_model.loc[index, "p_value"]
            finite = values.notna()
            if finite.any():
                state_model.loc[
                    np.asarray(index)[finite.to_numpy()], "p_value_adjusted"
                ] = benjamini_hochberg(values.loc[finite].to_numpy())
    conditional_table = pd.concat(conditional_frames, ignore_index=True)
    robustness_table = pd.DataFrame(robustness_rows)

    outputs = {
        "adaptive_horizon_decision_value.csv": decision_table,
        "switch_hold_decomposition.csv": decomposition_table,
        "exit_probability_ranking.csv": ranking_table,
        "exit_probability_quintiles.csv": quintile_table,
        "matched_action_permutation.csv": permutation_table,
        "holding_duration_distribution.csv": duration_distribution,
        "holding_duration_hazard.csv": hazard_table,
        "controller_state_model.csv": state_model,
        "controller_state_conditional_value.csv": conditional_table,
        "horizon_robustness.csv": robustness_table,
        "decision_audit.csv": pd.DataFrame(decision_audits),
    }
    for filename, frame in outputs.items():
        _write_frame(frame, tables_dir / filename)
    for market, draws in permutation_draws.items():
        _write_frame(
            draws,
            tables_dir / f"{market}_matched_action_permutation_draws.csv",
        )

    _plot_outputs(
        decision_table,
        quintile_table,
        permutation_draws,
        hazard_table,
        output_dir,
    )
    _write_report(
        output_dir,
        decision_table,
        decomposition_table,
        ranking_table,
        permutation_table,
    )
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit = "unknown"
    manifest = {
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "code_commit": commit,
        "market_seed_mapping": seed_mapping,
        "input_files": input_records,
        "audit": decision_audits,
        "random_seed": args.random_seed,
    }
    metadata_dir = output_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    (metadata_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
