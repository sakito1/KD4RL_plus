# Fixed-Path Transaction-Cost Sensitivity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use $superpower-subagents (recommended) or $superpower-executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking via update_plan.

**Goal:** Replay the paper-selected Nasdaq-100 seed 49 and CSI-300 seed 90 executed-weight paths at five transaction-cost rates and produce auditable performance tables.

**Architecture:** Add one standalone paper-experiment module that loads the existing action traces and adjusted prices, reconstructs drift-aware turnover and gross returns once, and applies every requested cost rate to that common path. Reuse the existing portfolio metric definitions, then write daily replay data, a summary CSV, a compact Markdown report, and a hashed run manifest.

**Tech Stack:** Python 3, NumPy, pandas, pytest, existing `paper_experiments` utilities.

---

### Task 1: Implement and test drift-aware fixed-path replay

**Files:**
- Create: `tests/test_transaction_cost_sensitivity.py`
- Create: `paper_experiments/analyze_transaction_cost_sensitivity.py`

- [ ] **Step 1: Write failing replay tests**

Add tests that construct a two-asset executed path and price panel, then assert:

```python
from paper_experiments.analyze_transaction_cost_sensitivity import (
    replay_executed_weight_path,
)


def test_replay_uses_price_drifted_previous_weights():
    weights = pd.DataFrame(
        [[0.5, 0.5], [0.5, 0.5]],
        index=pd.to_datetime(["2020-01-01", "2020-01-02"]),
        columns=["A", "B"],
    )
    prices = pd.DataFrame(
        [[100.0, 100.0], [110.0, 100.0], [121.0, 100.0]],
        index=pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]),
        columns=["A", "B"],
    )
    replay = replay_executed_weight_path(weights, prices, [0.0, 0.001])
    assert replay.loc[pd.Timestamp("2020-01-02"), "turnover"] == pytest.approx(
        abs(0.5 - 110.0 / 210.0) + abs(0.5 - 100.0 / 210.0)
    )


def test_higher_cost_cannot_improve_same_fixed_path():
    replay = replay_executed_weight_path(weights, prices, [0.00005, 0.0005])
    assert (
        replay["net_growth_tc_0p0500pct"]
        <= replay["net_growth_tc_0p0050pct"] + 1e-15
    ).all()
```

Also assert that a zero-turnover row has identical net growth at all cost rates
and that all cost columns use the same dates, turnover, and gross return.

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
/home/tongwenxuan/conda/envs/xuangu/bin/python -m pytest \
  tests/test_transaction_cost_sensitivity.py -q
```

Expected: collection fails because
`paper_experiments.analyze_transaction_cost_sensitivity` does not exist.

- [ ] **Step 3: Implement the replay kernel**

Create:

```python
def cost_label(rate: float) -> str:
    return f"tc_{rate * 100:.4f}pct".replace(".", "p")


def replay_executed_weight_path(
    executed: pd.DataFrame,
    prices: pd.DataFrame,
    cost_rates: Sequence[float],
) -> pd.DataFrame:
```

For every executed target after the first trace row:

1. drift the previous target with the previous-date-to-current-date price ratio;
2. compute `turnover = abs(current_target - drifted).sum()`;
3. compute the current-date-to-next-date gross portfolio growth;
4. compute `net_growth = (1 - rate * turnover) * gross_growth`;
5. store gross return, turnover, cost rate, net log return, and cumulative wealth
   for every rate.

Reject misaligned assets, negative cost rates, non-positive prices, and
non-positive net growth.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run:

```bash
/home/tongwenxuan/conda/envs/xuangu/bin/python -m pytest \
  tests/test_transaction_cost_sensitivity.py -q
```

Expected: all replay tests pass.

- [ ] **Step 5: Commit replay kernel**

```bash
git add paper_experiments/analyze_transaction_cost_sensitivity.py \
  tests/test_transaction_cost_sensitivity.py
git commit -m "feat: add fixed path cost replay"
```

### Task 2: Add summary metrics and artifact generation

**Files:**
- Modify: `tests/test_transaction_cost_sensitivity.py`
- Modify: `paper_experiments/analyze_transaction_cost_sensitivity.py`

- [ ] **Step 1: Write failing summary tests**

Add a test for:

```python
from paper_experiments.analyze_transaction_cost_sensitivity import summarize_replay

summary = summarize_replay(
    replay,
    market="nas",
    seed=49,
    cost_rates=[0.00005, 0.00010, 0.00050],
    reference_rate=0.00005,
)
assert list(summary["transaction_cost_pct"]) == [0.005, 0.010, 0.050]
assert summary["total_return"].is_monotonic_decreasing
assert summary.loc[0, "delta_total_return_pp"] == pytest.approx(0.0)
assert np.isfinite(summary[["sharpe", "max_drawdown", "calmar"]]).all().all()
```

Add a CLI artifact test using temporary action/price fixtures and assert that
the summary CSV, daily CSV, Markdown report, and JSON manifest are created.

- [ ] **Step 2: Run tests and verify RED**

Run the focused test file and expect failure because `summarize_replay` and the
CLI artifact writer are missing.

- [ ] **Step 3: Implement summary and CLI**

Implement:

```python
def summarize_replay(
    replay: pd.DataFrame,
    *,
    market: str,
    seed: int,
    cost_rates: Sequence[float],
    reference_rate: float,
) -> pd.DataFrame:
```

Use `portfolio_path_metrics` from
`paper_experiments.analyze_inner_outer_statistical_validation` for TR, SR, MDD,
and CR. Add replay days, mean turnover, cumulative charged cost rate, and metric
changes relative to the 0.005% row.

Add CLI arguments for:

```text
--full_actions_root
--prices_root
--results_root
--output_dir
--markets nas sh
--seeds nas:49 sh:90
--cost_rates 0.00005 0.00010 0.00015 0.00020 0.00050
--reference_rate 0.00005
```

Write the artifacts and include SHA-256 hashes for action traces, price files,
checkpoints, and command JSON files in the manifest.

- [ ] **Step 4: Run focused tests and full suite**

Run:

```bash
/home/tongwenxuan/conda/envs/xuangu/bin/python -m pytest \
  tests/test_transaction_cost_sensitivity.py -q
/home/tongwenxuan/conda/envs/xuangu/bin/python -m pytest -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit artifact pipeline**

```bash
git add paper_experiments/analyze_transaction_cost_sensitivity.py \
  tests/test_transaction_cost_sensitivity.py
git commit -m "feat: report transaction cost sensitivity"
```

### Task 3: Run the selected-model replay and audit results

**Files:**
- Create under: `reproduced_outputs/fixed_path_transaction_cost_sensitivity/`

- [ ] **Step 1: Run the selected trajectories**

```bash
/home/tongwenxuan/conda/envs/xuangu/bin/python \
  paper_experiments/analyze_transaction_cost_sensitivity.py \
  --full_actions_root \
  reproduced_outputs/inner_daily_stats_paper_selected/_cache/inner_base_adjustment \
  --prices_root DeepAries/data \
  --results_root reproduced_inputs/paper_selected/results_root \
  --output_dir reproduced_outputs/fixed_path_transaction_cost_sensitivity \
  --markets nas sh \
  --seeds nas:49 sh:90 \
  --cost_rates 0.00005 0.00010 0.00015 0.00020 0.00050 \
  --reference_rate 0.00005
```

Expected: exit code 0 and all required artifacts exist.

- [ ] **Step 2: Audit monotonicity and common paths**

Read the summary and daily CSVs and verify:

- total return is non-increasing with cost for each market;
- every rate has the same replay-day count;
- gross return and turnover are rate-independent;
- all wealth values are positive and finite;
- the manifest hashes match the selected seed 49 and seed 90 inputs.

- [ ] **Step 3: Run final verification**

Run:

```bash
/home/tongwenxuan/conda/envs/xuangu/bin/python -m pytest -q
git status --short
```

Report the exact test count, the five-rate metrics for both markets, and the
fixed-path interpretation boundary.

