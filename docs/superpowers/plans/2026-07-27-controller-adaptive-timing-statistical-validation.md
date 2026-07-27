# Controller Adaptive-Timing Statistical Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use $superpower-subagents (recommended) or $superpower-executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking via update_plan.

**Goal:** Build and run a reproducible, non-fixed-day statistical analysis that tests whether the learned Controller's adaptive switch/hold timing has counterfactual decision value.

**Architecture:** Add one standalone analysis module that consumes the existing horizon-30 Controller traces and project price data without modifying the current paper figure pipeline. Pure functions parse and validate traces, construct adaptive-horizon outcomes, perform dependent-time-series inference, evaluate exit-probability ranking and matched timing placebos, summarize holding behavior and observable states, then a CLI writes tables, figures, a manifest, and a claim-bounded report.

**Tech Stack:** Python 3, pandas, NumPy, SciPy, statsmodels, scikit-learn, matplotlib, pytest.

---

## File Structure

- Create `paper_experiments/analyze_controller_adaptive_timing.py`: all pure
  analysis functions, plotting helpers, report writer, and CLI.
- Create `tests/test_controller_adaptive_timing.py`: focused synthetic tests for
  every new behavior and a small end-to-end artifact test.
- Read only
  `paper_experiments_outputs/paper_experiments_final/_cache/counterfactual_horizon30/*.csv`
  and `DeepAries/data/{nas,sh}/{nas,sh}_data.csv`.
- Write only
  `reproduced_outputs/controller_adaptive_timing_statistical_validation/`.

### Task 1: Parse and validate adaptive counterfactual decisions

**Files:**
- Create: `tests/test_controller_adaptive_timing.py`
- Create: `paper_experiments/analyze_controller_adaptive_timing.py`

- [ ] **Step 1: Write failing tests for curve parsing and free-decision validation**

```python
import json

import numpy as np
import pandas as pd
import pytest

from paper_experiments.analyze_controller_adaptive_timing import (
    parse_controller_decisions,
)


def _decision_frame():
    return pd.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-03", "2024-01-04"],
            "step": [1, 2, 3],
            "decision_type": ["free_decision"] * 3,
            "is_switch": [1, 0, 1],
            "is_free_switch": [1, 0, 1],
            "duration_before_decision": [10, 20, 29],
            "exit_prob": [0.8, 0.2, 0.7],
            "policy_logit": [1.2, -1.1, 0.9],
            "hold_curve_30": [
                json.dumps([1.0] + [1.0] * 30),
                json.dumps([1.0] + [1.001] * 30),
                json.dumps([1.0] + [1.0] * 30),
            ],
            "switch_curve_30": [
                json.dumps([1.0] + [1.002] * 30),
                json.dumps([1.0] + [0.999] * 30),
                json.dumps([1.0] + [1.001] * 30),
            ],
        }
    )


def test_parse_controller_decisions_keeps_all_free_rows_and_adaptive_horizons():
    parsed, audit = parse_controller_decisions(_decision_frame(), max_horizon=30)
    assert parsed["adaptive_horizon"].tolist() == [20, 10, 1]
    assert parsed["action"].tolist() == [1, 0, 1]
    assert audit["input_rows"] == 3
    assert audit["valid_free_decisions"] == 3
    assert audit["invalid_curve_rows"] == 0


def test_parse_controller_decisions_rejects_disagreeing_action_flags():
    frame = _decision_frame()
    frame.loc[0, "is_free_switch"] = 0
    with pytest.raises(ValueError, match="action flags disagree"):
        parse_controller_decisions(frame, max_horizon=30)
```

- [ ] **Step 2: Run the tests and verify the missing module failure**

Run:

```bash
/home/tongwenxuan/conda/envs/xuangu/bin/python -m pytest \
  tests/test_controller_adaptive_timing.py -q
```

Expected: collection fails because
`paper_experiments.analyze_controller_adaptive_timing` does not exist.

- [ ] **Step 3: Implement curve parsing and validation**

Add:

```python
def parse_curve(value) -> np.ndarray:
    parsed = np.asarray(json.loads(value), dtype="float64")
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
    required = {
        "date", "step", "decision_type", "is_switch", "is_free_switch",
        "duration_before_decision", "exit_prob", "policy_logit",
        f"hold_curve_{max_horizon}", f"switch_curve_{max_horizon}",
    }
    missing = sorted(required.difference(actions.columns))
    if missing:
        raise ValueError(f"action trace is missing columns: {missing}")
    frame = actions.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values(["date", "step"])
    if frame["date"].duplicated().any() or frame["step"].duplicated().any():
        raise ValueError("free-decision trace has duplicate dates or steps")
    free = frame[frame["decision_type"] == "free_decision"].copy()
    free["action"] = pd.to_numeric(free["is_switch"], errors="raise").astype(int)
    free_switch = pd.to_numeric(
        free["is_free_switch"], errors="raise"
    ).astype(int)
    if not np.array_equal(free["action"].to_numpy(), free_switch.to_numpy()):
        raise ValueError("action flags disagree on free decisions")
    duration = pd.to_numeric(
        free["duration_before_decision"], errors="raise"
    ).astype(int)
    if ((duration < 1) | (duration >= max_horizon)).any():
        raise ValueError("duration is outside the adaptive-horizon range")
    free["adaptive_horizon"] = (max_horizon - duration).clip(lower=1)
    hold_curves, switch_curves = [], []
    invalid = 0
    for row in free.itertuples(index=False):
        try:
            hold = parse_curve(getattr(row, f"hold_curve_{max_horizon}"))
            switch = parse_curve(getattr(row, f"switch_curve_{max_horizon}"))
            horizon = int(row.adaptive_horizon)
            if len(hold) <= horizon or len(switch) <= horizon:
                raise ValueError("counterfactual curve is shorter than horizon")
        except (TypeError, ValueError, json.JSONDecodeError):
            invalid += 1
            hold = switch = None
        hold_curves.append(hold)
        switch_curves.append(switch)
    if invalid:
        raise ValueError(f"{invalid} invalid counterfactual curve rows")
    free["hold_curve"] = hold_curves
    free["switch_curve"] = switch_curves
    audit = {
        "input_rows": int(len(actions)),
        "free_decisions": int(len(free)),
        "valid_free_decisions": int(len(free)),
        "invalid_curve_rows": int(invalid),
    }
    return free, audit
```

- [ ] **Step 4: Run the focused tests**

Expected: both tests pass.

- [ ] **Step 5: Commit Task 1**

```bash
git add paper_experiments/analyze_controller_adaptive_timing.py \
  tests/test_controller_adaptive_timing.py
git commit -m "Add controller counterfactual trace validation"
```

### Task 2: Compute adaptive-horizon decision outcomes

**Files:**
- Modify: `tests/test_controller_adaptive_timing.py`
- Modify: `paper_experiments/analyze_controller_adaptive_timing.py`

- [ ] **Step 1: Add failing tests for return, drawdown, and action-value signs**

```python
from paper_experiments.analyze_controller_adaptive_timing import (
    compute_adaptive_outcomes,
)


def test_compute_adaptive_outcomes_scores_switch_and_hold_correctly():
    parsed, _ = parse_controller_decisions(_decision_frame(), max_horizon=30)
    result = compute_adaptive_outcomes(parsed)
    assert result.loc[0, "return_advantage_log_per_day"] > 0
    assert result.loc[0, "return_decision_value_log_per_day"] > 0
    assert result.loc[1, "return_advantage_log_per_day"] < 0
    assert result.loc[1, "return_decision_value_log_per_day"] > 0
    assert result.loc[2, "adaptive_horizon"] == 1


def test_compute_adaptive_outcomes_uses_hold_minus_switch_for_mdd():
    frame = _decision_frame().iloc[[0]].copy()
    frame.loc[frame.index[0], "hold_curve_30"] = json.dumps(
        [1.0, 0.90] + [0.90] * 29
    )
    frame.loc[frame.index[0], "switch_curve_30"] = json.dumps(
        [1.0, 0.97] + [0.97] * 29
    )
    parsed, _ = parse_controller_decisions(frame, max_horizon=30)
    result = compute_adaptive_outcomes(parsed)
    assert result.loc[0, "mdd_advantage"] == pytest.approx(0.07)
    assert result.loc[0, "mdd_decision_value"] == pytest.approx(0.07)
```

- [ ] **Step 2: Run tests and verify `compute_adaptive_outcomes` is missing**

Expected: import or attribute failure for the new function.

- [ ] **Step 3: Implement adaptive outcomes and fixed-horizon sensitivity**

```python
def max_drawdown(curve: np.ndarray) -> float:
    peaks = np.maximum.accumulate(curve)
    return float(np.max((peaks - curve) / np.maximum(peaks, 1e-12)))


def compute_adaptive_outcomes(decisions: pd.DataFrame) -> pd.DataFrame:
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
                "return_decision_value_log_per_day": direction * return_advantage,
                "return_decision_value_bp_day": direction * return_advantage * 10000.0,
                "mdd_decision_value": direction * mdd_advantage,
            }
        )
    return pd.DataFrame(rows)


def compute_horizon_outcomes(
    decisions: pd.DataFrame,
    horizons: Sequence[int] = (5, 10, 20, 30),
) -> pd.DataFrame:
    expanded = []
    for _, row in decisions.iterrows():
        for horizon in horizons:
            horizon = int(horizon)
            if len(row["hold_curve"]) <= horizon:
                continue
            item = row.copy()
            item["adaptive_horizon"] = horizon
            item["sensitivity_horizon"] = horizon
            expanded.append(item)
    if not expanded:
        return pd.DataFrame()
    return compute_adaptive_outcomes(pd.DataFrame(expanded).reset_index(drop=True))
```

- [ ] **Step 4: Run focused tests**

Expected: adaptive-outcome tests pass.

- [ ] **Step 5: Commit Task 2**

```bash
git add paper_experiments/analyze_controller_adaptive_timing.py \
  tests/test_controller_adaptive_timing.py
git commit -m "Add adaptive controller decision outcomes"
```

### Task 3: Add dependent-series inference and decision summaries

**Files:**
- Modify: `tests/test_controller_adaptive_timing.py`
- Modify: `paper_experiments/analyze_controller_adaptive_timing.py`

- [ ] **Step 1: Add failing tests for block intervals and action decomposition**

```python
from paper_experiments.analyze_controller_adaptive_timing import (
    circular_block_bootstrap,
    summarize_decision_value,
    summarize_switch_hold,
)


def test_circular_block_bootstrap_is_deterministic_and_paired():
    x = np.arange(12.0)
    y = x + 3.0
    draws1 = circular_block_bootstrap(
        [x, y], lambda a, b: np.mean(b - a),
        block_length=4, reps=100, seed=7,
    )
    draws2 = circular_block_bootstrap(
        [x, y], lambda a, b: np.mean(b - a),
        block_length=4, reps=100, seed=7,
    )
    assert np.array_equal(draws1, draws2)
    assert np.allclose(draws1, 3.0)


def test_switch_hold_summary_uses_all_actions():
    parsed, _ = parse_controller_decisions(_decision_frame(), max_horizon=30)
    outcomes = compute_adaptive_outcomes(parsed)
    summary = summarize_switch_hold(
        outcomes, block_length=2, bootstrap_reps=200, seed=9
    )
    assert set(summary["action"]) == {"switch", "hold"}
    assert summary["n"].sum() == len(outcomes)
```

- [ ] **Step 2: Run tests and verify the new functions are missing**

- [ ] **Step 3: Reuse or implement inference helpers**

Import the already tested implementations rather than duplicating them:

```python
from paper_experiments.analyze_inner_outer_statistical_validation import (
    _block_mean_interval as block_mean_interval,
    circular_block_bootstrap,
    newey_west_mean_test,
)
```

Add:

```python
def summarize_decision_value(
    outcomes: pd.DataFrame,
    *,
    block_length: int,
    bootstrap_reps: int,
    seed: int,
) -> dict[str, float | int]:
    ret = outcomes["return_decision_value_bp_day"].to_numpy(dtype="float64")
    mdd = outcomes["mdd_decision_value"].to_numpy(dtype="float64")
    ret_hac = newey_west_mean_test(ret, maxlags=5)
    mdd_hac = newey_west_mean_test(mdd, maxlags=5)
    ret_ci = block_mean_interval(
        ret, block_length=block_length, reps=bootstrap_reps, seed=seed
    )
    mdd_ci = block_mean_interval(
        mdd, block_length=block_length, reps=bootstrap_reps, seed=seed + 1
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
    rows = []
    for action_value, label in [(1, "switch"), (0, "hold")]:
        group = outcomes[outcomes["action"] == action_value]
        advantage = group["return_advantage_bp_day"].to_numpy(dtype="float64")
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
                "favorable_return_ratio": float(np.mean(direction * advantage > 0)),
                "mean_mdd_advantage": float(mdd.mean()),
                "favorable_mdd_ratio": float(np.mean(direction * mdd > 0)),
            }
        )
    return pd.DataFrame(rows)
```

- [ ] **Step 4: Run Task 3 tests**

- [ ] **Step 5: Commit Task 3**

```bash
git add paper_experiments/analyze_controller_adaptive_timing.py \
  tests/test_controller_adaptive_timing.py
git commit -m "Add controller decision value inference"
```

### Task 4: Add exit-probability ranking and matched permutation placebo

**Files:**
- Modify: `tests/test_controller_adaptive_timing.py`
- Modify: `paper_experiments/analyze_controller_adaptive_timing.py`

- [ ] **Step 1: Add failing ranking and permutation tests**

```python
from paper_experiments.analyze_controller_adaptive_timing import (
    matched_action_permutation,
    summarize_exit_probability,
)


def test_exit_probability_summary_detects_perfect_ranking():
    frame = pd.DataFrame(
        {
            "exit_prob": np.linspace(0.01, 0.99, 100),
            "policy_logit": np.linspace(-3.0, 3.0, 100),
            "return_advantage_log_per_day": np.linspace(-0.01, 0.01, 100),
            "action": (np.arange(100) >= 50).astype(int),
        }
    )
    summary, quintiles = summarize_exit_probability(
        frame, block_length=10, bootstrap_reps=200, seed=11
    )
    assert summary["spearman_exit_prob_advantage"] == pytest.approx(1.0)
    assert summary["auroc_positive_advantage"] == pytest.approx(1.0)
    assert quintiles.iloc[-1]["mean_return_advantage_bp_day"] > quintiles.iloc[0][
        "mean_return_advantage_bp_day"
    ]


def test_matched_permutation_preserves_switch_count_within_strata():
    frame = pd.DataFrame(
        {
            "action": [0, 1, 0, 1, 0, 1, 0, 1],
            "duration_stratum": ["short"] * 4 + ["long"] * 4,
            "volatility_stratum": ["low", "low", "high", "high"] * 2,
            "return_advantage_log_per_day": np.linspace(-0.01, 0.01, 8),
            "mdd_advantage": np.linspace(-0.02, 0.02, 8),
        }
    )
    result, draws = matched_action_permutation(frame, reps=100, seed=13)
    assert result["placebo_reps"] == 100
    assert len(draws) == 100
    assert result["invalid_permutations"] == 0
```

- [ ] **Step 2: Run tests and verify failures for missing functions**

- [ ] **Step 3: Implement ranking metrics**

Use `scipy.stats.spearmanr` and
`sklearn.metrics.{roc_auc_score, balanced_accuracy_score, matthews_corrcoef}`.
Handle a constant advantage label by returning NaN AUROC and recording the
reason in the summary.

```python
def summarize_exit_probability(
    outcomes: pd.DataFrame,
    *,
    block_length: int,
    bootstrap_reps: int,
    seed: int,
) -> tuple[dict[str, float | int], pd.DataFrame]:
    clean = outcomes.dropna(
        subset=["exit_prob", "policy_logit", "return_advantage_log_per_day"]
    ).copy()
    target = (clean["return_advantage_log_per_day"] > 0).astype(int)
    exit_rho = scipy.stats.spearmanr(
        clean["exit_prob"], clean["return_advantage_log_per_day"]
    ).statistic
    logit_rho = scipy.stats.spearmanr(
        clean["policy_logit"], clean["return_advantage_log_per_day"]
    ).statistic
    auroc = (
        roc_auc_score(target, clean["exit_prob"])
        if target.nunique() == 2 else np.nan
    )
    predicted = clean["action"].astype(int)
    quintile = pd.qcut(
        clean["exit_prob"].rank(method="first"), q=5,
        labels=["Q1", "Q2", "Q3", "Q4", "Q5"],
    )
    clean["probability_quintile"] = quintile
    quintiles = clean.groupby("probability_quintile", observed=False).agg(
        n=("action", "size"),
        mean_exit_prob=("exit_prob", "mean"),
        mean_return_advantage_bp_day=("return_advantage_bp_day", "mean"),
        mean_mdd_advantage=("mdd_advantage", "mean"),
        switch_rate=("action", "mean"),
    ).reset_index()
    return {
        "spearman_exit_prob_advantage": float(exit_rho),
        "spearman_policy_logit_advantage": float(logit_rho),
        "auroc_positive_advantage": float(auroc),
        "balanced_accuracy": float(balanced_accuracy_score(target, predicted)),
        "matthews_correlation": float(matthews_corrcoef(target, predicted)),
    }, quintiles
```

- [ ] **Step 4: Implement matched action permutation**

```python
def matched_action_permutation(
    outcomes: pd.DataFrame,
    *,
    reps: int,
    seed: int,
) -> tuple[dict[str, float | int], pd.DataFrame]:
    rng = np.random.default_rng(seed)
    frame = outcomes.copy()
    groups = [
        index.to_numpy()
        for _, index in frame.groupby(
            ["duration_stratum", "volatility_stratum"],
            observed=False,
        ).groups.items()
    ]
    observed_return = float(frame["return_decision_value_bp_day"].mean())
    observed_mdd = float(frame["mdd_decision_value"].mean())
    draws = []
    invalid = 0
    for _ in range(int(reps)):
        permuted = frame["action"].to_numpy(dtype=int).copy()
        for index in groups:
            before = int(permuted[index].sum())
            permuted[index] = rng.permutation(permuted[index])
            invalid += int(permuted[index].sum() != before)
        direction = 2.0 * permuted - 1.0
        draws.append(
            {
                "return_value_bp_day": float(np.mean(
                    direction * frame["return_advantage_bp_day"].to_numpy()
                )),
                "mdd_value": float(np.mean(
                    direction * frame["mdd_advantage"].to_numpy()
                )),
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
            (1 + np.sum(
                draw_frame["return_value_bp_day"] >= observed_return
            )) / (reps + 1)
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
```

- [ ] **Step 5: Run Task 4 tests**

- [ ] **Step 6: Commit Task 4**

```bash
git add paper_experiments/analyze_controller_adaptive_timing.py \
  tests/test_controller_adaptive_timing.py
git commit -m "Add controller timing ranking and placebo tests"
```

### Task 5: Add holding behavior and observable-state analysis

**Files:**
- Modify: `tests/test_controller_adaptive_timing.py`
- Modify: `paper_experiments/analyze_controller_adaptive_timing.py`

- [ ] **Step 1: Add failing tests for spells, hazard, and lagged states**

```python
from paper_experiments.analyze_controller_adaptive_timing import (
    attach_observable_states,
    holding_duration_hazard,
    summarize_holding_spells,
)


def test_holding_spells_end_on_switches():
    decisions = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=6),
            "step": np.arange(6),
            "action": [0, 0, 1, 0, 1, 0],
            "duration_before_decision": [1, 2, 3, 1, 2, 1],
        }
    )
    spells = summarize_holding_spells(decisions)
    assert spells["completed_duration"].tolist() == [3, 2]


def test_hazard_is_switches_divided_by_at_risk_decisions():
    decisions = pd.DataFrame(
        {"duration_before_decision": [1, 1, 2, 2], "action": [0, 1, 1, 0]}
    )
    hazard = holding_duration_hazard(decisions)
    assert hazard.loc[hazard["duration"] == 1, "hazard"].iloc[0] == pytest.approx(0.5)


def test_observable_states_use_only_information_through_decision_date():
    dates = pd.date_range("2024-01-01", periods=30, freq="D")
    decisions = pd.DataFrame({"date": dates[20:22]})
    portfolio = pd.DataFrame(
        {"date": dates, "portfolio_value": np.arange(1.0, 31.0)}
    )
    market = pd.Series(np.arange(30.0) / 1000.0, index=dates)
    result = attach_observable_states(decisions, portfolio, market, lookback=20)
    assert result.loc[0, "pre_return_20"] == pytest.approx(21.0 / 1.0 - 1.0)
```

- [ ] **Step 2: Run tests and verify missing-function failures**

- [ ] **Step 3: Implement holding summaries and state attachment**

```python
def summarize_holding_spells(decisions: pd.DataFrame) -> pd.DataFrame:
    switched = decisions[pd.to_numeric(
        decisions["action"], errors="raise"
    ).astype(int) == 1].copy()
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
    frame = decisions.loc[:, ["duration_before_decision", "action"]].copy()
    frame["duration_before_decision"] = pd.to_numeric(
        frame["duration_before_decision"], errors="raise"
    ).astype(int)
    frame["action"] = pd.to_numeric(frame["action"], errors="raise").astype(int)
    result = (
        frame.groupby("duration_before_decision", as_index=False)
        .agg(at_risk=("action", "size"), switches=("action", "sum"))
        .rename(columns={"duration_before_decision": "duration"})
    )
    result["hazard"] = result["switches"] / result["at_risk"]
    return result


def attach_observable_states(
    decisions: pd.DataFrame,
    portfolio: pd.DataFrame,
    equal_weight_market_return: pd.Series,
    *,
    lookback: int = 20,
) -> pd.DataFrame:
    values = portfolio.loc[:, ["date", "portfolio_value"]].copy()
    values["date"] = pd.to_datetime(values["date"])
    values = values.sort_values("date").set_index("date")
    market = equal_weight_market_return.copy().sort_index().astype("float64")
    rows = []
    for row in decisions.itertuples(index=False):
        date = pd.Timestamp(row.date)
        history = values.loc[:date, "portfolio_value"].tail(lookback + 1)
        market_history = market.loc[:date].tail(lookback)
        pre_return = (
            float(history.iloc[-1] / history.iloc[0] - 1.0)
            if len(history) == lookback + 1 else np.nan
        )
        pre_drawdown = (
            max_drawdown(history.to_numpy(dtype="float64"))
            if len(history) >= 2 else np.nan
        )
        market_volatility = (
            float(market_history.std(ddof=1) * np.sqrt(252.0))
            if len(market_history) == lookback else np.nan
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
    return result
```

Implement `tercile_labels` with the finite 1/3 and 2/3 quantiles and labels
`low`, `mid`, and `high`; when both cut points coincide, return `mid` for every
finite observation so the permutation code never creates invalid empty bins.

- [ ] **Step 4: Add and test explanatory state summaries**

```python
def fit_switch_state_model(decisions: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "duration_before_decision", "pre_return_20",
        "pre_drawdown_20", "market_volatility_20",
    ]
    clean = decisions.dropna(subset=columns + ["action"]).copy()
    design = clean[columns].astype("float64")
    design = (design - design.mean()) / design.std(ddof=1).replace(0.0, 1.0)
    design["duration_squared"] = design["duration_before_decision"] ** 2
    design = statsmodels.api.add_constant(design)
    fit = statsmodels.api.Logit(clean["action"].astype(int), design).fit(
        disp=False,
        cov_type="HAC",
        cov_kwds={"maxlags": 5},
    )
    return pd.DataFrame(
        {
            "term": fit.params.index,
            "odds_ratio": np.exp(fit.params.to_numpy()),
            "std_error": fit.bse.to_numpy(),
            "p_value": fit.pvalues.to_numpy(),
            "ci_low": np.exp(fit.conf_int()[0].to_numpy()),
            "ci_high": np.exp(fit.conf_int()[1].to_numpy()),
        }
    )


def summarize_state_conditional_value(decisions: pd.DataFrame) -> pd.DataFrame:
    specs = {
        "duration": "duration_stratum",
        "pre_return": "pre_return_stratum",
        "pre_drawdown": "pre_drawdown_stratum",
        "market_volatility": "volatility_stratum",
    }
    rows = []
    for state_name, column in specs.items():
        for level, group in decisions.groupby(column, observed=False):
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
```

Add a synthetic test in which higher volatility deterministically increases
switching and verify the volatility odds ratio exceeds one.

- [ ] **Step 5: Run Task 5 tests**

- [ ] **Step 6: Commit Task 5**

```bash
git add paper_experiments/analyze_controller_adaptive_timing.py \
  tests/test_controller_adaptive_timing.py
git commit -m "Add controller holding and state analysis"
```

### Task 6: Add CLI, artifacts, figures, and claim-bounded report

**Files:**
- Modify: `tests/test_controller_adaptive_timing.py`
- Modify: `paper_experiments/analyze_controller_adaptive_timing.py`

- [ ] **Step 1: Add a failing small end-to-end artifact test**

```python
from paper_experiments.analyze_controller_adaptive_timing import main


def test_main_writes_required_artifacts(tmp_path, synthetic_input_bundle):
    exit_code = main(
        [
            "--input_dir", str(synthetic_input_bundle["input_dir"]),
            "--prices_root", str(synthetic_input_bundle["prices_root"]),
            "--output_dir", str(tmp_path),
            "--markets", "nas",
            "--seeds", "nas:49",
            "--bootstrap_reps", "50",
            "--placebo_reps", "50",
        ]
    )
    assert exit_code == 0
    required = [
        "tables/adaptive_horizon_decision_value.csv",
        "tables/switch_hold_decomposition.csv",
        "tables/exit_probability_ranking.csv",
        "tables/matched_action_permutation.csv",
        "CONTROLLER_ADAPTIVE_TIMING_STATISTICAL_VALIDATION.md",
        "metadata/run_manifest.json",
    ]
    for relative in required:
        assert (tmp_path / relative).exists()
```

The fixture must write a complete miniature horizon-30 action trace, portfolio
trace, and price panel with deterministic results.

- [ ] **Step 2: Run the end-to-end test and verify artifact failure**

- [ ] **Step 3: Implement CLI and output assembly**

CLI arguments:

```text
--input_dir
--portfolio_trace_dir
--prices_root
--output_dir
--markets
--seeds
--max_horizon 30
--sensitivity_horizons 5 10 20 30
--block_length 30
--block_length_sensitivity 20 40 60
--bootstrap_reps 10000
--placebo_reps 5000
--random_seed 20260727
```

Write all tables required by the design and daily/decision-level audit tables.
Use a colorblind-safe palette and zero-reference lines in:

- decision-value forest;
- matched-action placebo distribution;
- exit-probability quintile plot;
- holding-duration hazard.

Save every figure as PNG and PDF.

- [ ] **Step 4: Implement manifest and report**

The manifest records:

- CLI arguments;
- code commit;
- input paths and SHA-256 hashes;
- market/seed mapping;
- input/valid/invalid row counts;
- decision and switch counts;
- curve and action invariants;
- random seeds.

The Markdown report must label every claim:

- `SUPPORTED`;
- `DESCRIPTIVE`;
- `NOT SUPPORTED`.

It must explicitly state that intervals describe test-period uncertainty for one
selected checkpoint and not cross-seed training uncertainty.

- [ ] **Step 5: Run Task 6 tests**

- [ ] **Step 6: Run the full project test suite**

Run:

```bash
/home/tongwenxuan/conda/envs/xuangu/bin/python -m pytest -q
```

Expected: all tests pass.

- [ ] **Step 7: Commit Task 6**

```bash
git add paper_experiments/analyze_controller_adaptive_timing.py \
  tests/test_controller_adaptive_timing.py
git commit -m "Add controller adaptive timing report pipeline"
```

### Task 7: Run the selected Controller analysis and audit results

**Files:**
- Generate:
  `reproduced_outputs/controller_adaptive_timing_statistical_validation/`

- [ ] **Step 1: Run the production analysis**

```bash
/home/tongwenxuan/conda/envs/xuangu/bin/python \
  paper_experiments/analyze_controller_adaptive_timing.py \
  --input_dir \
    paper_experiments_outputs/paper_experiments_final/_cache/counterfactual_horizon30 \
  --portfolio_trace_dir \
    paper_experiments_outputs/paper_experiments_final/_cache/counterfactual_horizon30 \
  --prices_root DeepAries/data \
  --output_dir \
    reproduced_outputs/controller_adaptive_timing_statistical_validation \
  --markets nas sh \
  --seeds nas:49 sh:90 \
  --max_horizon 30 \
  --sensitivity_horizons 5 10 20 30 \
  --block_length 30 \
  --block_length_sensitivity 20 40 60 \
  --bootstrap_reps 10000 \
  --placebo_reps 5000 \
  --random_seed 20260727
```

Expected: exit code 0 and all required tables, figures, report, and manifest.

- [ ] **Step 2: Audit row coverage and invariants**

Verify:

```text
Nasdaq valid free decisions = 1,334
Nasdaq free switches = 231
CSI-300 valid free decisions = 1,220
CSI-300 free switches = 92
invalid curve rows = 0
permutation switch-count violations = 0
```

- [ ] **Step 3: Audit statistical claims**

For every `SUPPORTED` line, verify:

1. the point estimate has the expected direction;
2. the primary 95% block interval excludes the null;
3. the applicable adjusted p-value is below 0.05;
4. the number matches the source CSV.

Downgrade any inconsistent claim to `DESCRIPTIVE` or `NOT SUPPORTED`.

- [ ] **Step 4: Inspect every generated PNG**

Use image inspection to verify:

- labels and units are readable;
- zero/reference lines are visible;
- Nasdaq and CSI-300 labels are correct;
- no confidence bars or annotations are clipped.

- [ ] **Step 5: Run final verification**

```bash
/home/tongwenxuan/conda/envs/xuangu/bin/python -m pytest -q
git status --short
```

Expected: tests pass; only intended source/test changes and generated untracked
outputs are present.

- [ ] **Step 6: Produce the evidence interpretation**

Summarize:

- which decision-level timing claims are supported;
- whether `exit_prob` contains ranking information;
- whether timing beats matched random decisions;
- which states trigger switching and whether those states have positive value;
- which results are merely descriptive;
- which intended explanations are contradicted by the data.

Do not use fixed-day performance comparisons in the Controller interpretation.

## Verification

- Every new calculation is covered by a failing-then-passing synthetic test.
- All free decisions are included in the primary decision-value analysis.
- No existing Controller figure or table is overwritten.
- Bootstrap resampling is paired and deterministic for a fixed seed.
- Matched permutations preserve switch counts within every stratum.
- All multiple-comparison families report adjusted p-values.
- The production report is traceable to source CSVs and input hashes.

## Next skill

Use `$superpower-executing-plans` for inline implementation because this
runtime has no user-authorized subagent delegation.
