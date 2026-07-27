# Inner–Outer Statistical Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use $superpower-subagents (recommended) or $superpower-executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking via update_plan.

**Goal:** Build and run a reproducible analysis that measures Inner's configuration refinement, frozen-path direct effect, and closed-loop contribution relative to Outer on the paper-selected NAS and SH checkpoints.

**Architecture:** Add one standalone analysis module under `paper_experiments/` that consumes existing action traces, can generate missing Full/No-Inner traces through the existing evaluator, computes paired HAC/block-bootstrap statistics, and writes tables, figures, metadata, and a Markdown report. Add focused unit tests for weight invariants, ex-ante risk, fair transaction-cost accounting, paired bootstrap behavior, and deterministic placebo generation; do not modify model or training code.

**Tech Stack:** Python 3, pandas, NumPy, SciPy, statsmodels, scikit-learn covariance when available, matplotlib, pytest.

---

## File Structure

- Create `paper_experiments/analyze_inner_outer_statistical_validation.py`
  - CLI, trace loading/evaluation, configuration metrics, frozen-path counterfactual, closed-loop paired statistics, placebo analysis, plots, manifest, report.
- Create `tests/test_inner_outer_statistical_validation.py`
  - Deterministic unit and integration-sized tests built from synthetic weight/return traces.
- Create at runtime `reproduced_outputs/inner_outer_statistical_validation/`
  - Generated artifacts only; no generated output is committed by implementation tasks.
- Reference `docs/superpowers/specs/2026-07-27-inner-outer-statistical-validation-design.md`
  - Locked statistical design; implementation must not silently change primary metrics.

### Task 1: Weight parsing and mechanism invariants

**Files:**
- Create: `tests/test_inner_outer_statistical_validation.py`
- Create: `paper_experiments/analyze_inner_outer_statistical_validation.py`

- [ ] **Step 1: Write failing parsing and invariant tests**

```python
def test_parse_weight_trace_and_validate_support():
    actions = synthetic_actions(
        base=[[0.6, 0.4, 0.0], [0.5, 0.5, 0.0]],
        executed=[[0.5, 0.5, 0.0], [0.4, 0.6, 0.0]],
    )
    parsed = parse_weight_trace(actions)
    validation = validate_weight_invariants(parsed)
    assert list(parsed.base.columns) == ["A", "B", "C"]
    assert validation["max_abs_tilt_identity_error"] < 1e-12
    assert validation["max_abs_weight_sum_error"] < 1e-12
    assert validation["support_violation_count"] == 0
```

Also test that a positive executed weight on an asset whose base weight is zero increments `support_violation_count`.

- [ ] **Step 2: Run the focused test and confirm RED**

Run:

```bash
/home/tongwenxuan/conda/envs/xuangu/bin/python -m pytest \
  tests/test_inner_outer_statistical_validation.py::test_parse_weight_trace_and_validate_support -q
```

Expected: collection/import failure because the analysis module does not exist.

- [ ] **Step 3: Implement parsing and validation**

Define:

```python
@dataclass(frozen=True)
class WeightTrace:
    base: pd.DataFrame
    executed: pd.DataFrame
    tilt: pd.DataFrame

def parse_weight_trace(actions: pd.DataFrame) -> WeightTrace: ...

def validate_weight_invariants(
    trace: WeightTrace,
    *,
    atol: float = 1e-7,
) -> dict[str, float | int]: ...
```

Use `asset_names_json` for column order and `date` for a sorted `DatetimeIndex`. Reject duplicated dates, inconsistent vector lengths, non-finite weights, and missing JSON fields with `ValueError`. Compute identity, normalization, negativity, and support-violation diagnostics without mutating inputs.

- [ ] **Step 4: Run Task 1 tests and confirm GREEN**

Run the focused tests; expected all Task 1 tests pass.

- [ ] **Step 5: Commit Task 1**

```bash
git add paper_experiments/analyze_inner_outer_statistical_validation.py \
  tests/test_inner_outer_statistical_validation.py
git commit -m "Add inner outer weight trace validation"
```

### Task 2: Configuration refinement and ex-ante risk

**Files:**
- Modify: `paper_experiments/analyze_inner_outer_statistical_validation.py`
- Modify: `tests/test_inner_outer_statistical_validation.py`

- [ ] **Step 1: Write failing configuration-metric tests**

```python
def test_configuration_metrics_match_hand_calculation():
    base = pd.DataFrame([[0.6, 0.4]], columns=["A", "B"])
    executed = pd.DataFrame([[0.5, 0.5]], columns=["A", "B"])
    result = configuration_shape_metrics(base, executed)
    assert result.loc[0, "active_share"] == pytest.approx(0.1)
    assert result.loc[0, "base_hhi"] == pytest.approx(0.52)
    assert result.loc[0, "exec_hhi"] == pytest.approx(0.50)
    assert result.loc[0, "delta_effective_n"] > 0
```

```python
def test_ex_ante_risk_uses_only_past_returns():
    result = ex_ante_risk_metrics(
        base=base_weights,
        executed=exec_weights,
        asset_returns=returns,
        lookback=3,
    )
    changed_future = returns.copy()
    changed_future.iloc[-1] = 99.0
    rerun = ex_ante_risk_metrics(
        base=base_weights.iloc[:-1],
        executed=exec_weights.iloc[:-1],
        asset_returns=changed_future,
        lookback=3,
    )
    pd.testing.assert_series_equal(
        result.iloc[:-1]["delta_ex_ante_vol"],
        rerun["delta_ex_ante_vol"],
        check_names=False,
    )
```

Include a constant-return test proving the covariance fallback returns finite zeros.

- [ ] **Step 2: Run Task 2 tests and confirm RED**

Expected: missing `configuration_shape_metrics` and `ex_ante_risk_metrics`.

- [ ] **Step 3: Implement shape and risk metrics**

Define:

```python
def configuration_shape_metrics(
    base: pd.DataFrame,
    executed: pd.DataFrame,
) -> pd.DataFrame: ...

def estimate_covariance(window: np.ndarray) -> np.ndarray: ...

def ex_ante_risk_metrics(
    base: pd.DataFrame,
    executed: pd.DataFrame,
    asset_returns: pd.DataFrame,
    *,
    lookback: int,
) -> pd.DataFrame: ...
```

`estimate_covariance` first tries `sklearn.covariance.LedoitWolf`; on import/numerical failure use

```python
sample = np.cov(window, rowvar=False, ddof=1)
diag = np.diag(np.diag(sample))
cov = 0.9 * sample + 0.1 * diag
```

and symmetrize/clip negative numerical eigenvalues. The return window for decision date `t` must end at `t`, with each row representing information from dates no later than `t`; never use the return from `t` to `t+1`. Compute ordinary covariance and negative-return semicovariance, then annualize volatility by `sqrt(252)`.

- [ ] **Step 4: Implement state terciles**

Define:

```python
def attach_market_volatility_regime(
    daily: pd.DataFrame,
    equal_weight_market_return: pd.Series,
    *,
    lookback: int = 20,
) -> pd.DataFrame: ...
```

Use fixed sample terciles labelled `low`, `mid`, `high`, store numeric cut points in the manifest, and do not search alternate thresholds.

- [ ] **Step 5: Run Task 2 tests and confirm GREEN**

Run all tests in the new test file.

- [ ] **Step 6: Commit Task 2**

```bash
git add paper_experiments/analyze_inner_outer_statistical_validation.py \
  tests/test_inner_outer_statistical_validation.py
git commit -m "Add configuration refinement risk metrics"
```

### Task 3: Frozen-path fair direct effect

**Files:**
- Modify: `paper_experiments/analyze_inner_outer_statistical_validation.py`
- Modify: `tests/test_inner_outer_statistical_validation.py`

- [ ] **Step 1: Write failing fair-cost tests**

Construct a two-asset, three-date path where base remains `[0.5, 0.5]`, executed changes from `[0.5, 0.5]` to `[0.7, 0.3]`, and prices are known. Assert:

```python
assert direct.loc[date, "exec_turnover"] == pytest.approx(expected_exec_turnover)
assert direct.loc[date, "base_turnover"] == pytest.approx(expected_base_turnover)
assert direct.loc[date, "delta_net_log_return"] == pytest.approx(
    expected_exec_net - expected_base_net
)
```

Add a test showing equal target paths produce zero paired effect and equal costs.

- [ ] **Step 2: Run Task 3 tests and confirm RED**

Expected: missing `frozen_path_direct_effect`.

- [ ] **Step 3: Implement fair paired returns**

Define:

```python
def drift_weights(previous_target: np.ndarray, gross_ratio: np.ndarray) -> np.ndarray: ...

def frozen_path_direct_effect(
    base: pd.DataFrame,
    executed: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    transaction_cost_pct: float,
) -> pd.DataFrame: ...
```

For each path independently:

```python
drift = normalize(previous_target * ratio_previous_to_today)
turnover = abs(target_today - drift).sum()
net_growth = (1.0 - cost * turnover) * (target_today * ratio_today_to_next).sum()
net_log_return = log(max(net_growth, 1e-12))
```

Exclude the first row from inferential summaries because previous holdings are unavailable. Add equal-weight market next-day return and a preregistered worst-market-5% flag.

- [ ] **Step 4: Implement HAC and block-bootstrap primitives**

Define:

```python
def newey_west_mean_test(values: ArrayLike, *, maxlags: int = 5) -> dict[str, float]: ...

def circular_block_bootstrap(
    arrays: Sequence[np.ndarray],
    statistic: Callable[..., np.ndarray | float],
    *,
    block_length: int,
    reps: int,
    seed: int,
) -> np.ndarray: ...
```

Tests must confirm deterministic output for a fixed seed, paired arrays use identical sampled indices, and a constant zero series yields mean/CI zero without crashing.

- [ ] **Step 5: Implement direct summaries**

Define `summarize_frozen_path` returning mean alpha bp/day, HAC t/p, block CI, positive days, annualized alpha Sharpe, cumulative alpha, turnover/cost differences, daily-volatility difference, 5% ES difference, and worst-market-5% paired return difference.

- [ ] **Step 6: Run Task 3 tests and confirm GREEN**

Run the new test file and `tests/test_paper_experiments.py`.

- [ ] **Step 7: Commit Task 3**

```bash
git add paper_experiments/analyze_inner_outer_statistical_validation.py \
  tests/test_inner_outer_statistical_validation.py
git commit -m "Add frozen path inner direct effect"
```

### Task 4: Full/No-Inner trace acquisition and closed-loop comparison

**Files:**
- Modify: `paper_experiments/analyze_inner_outer_statistical_validation.py`
- Modify: `tests/test_inner_outer_statistical_validation.py`

- [ ] **Step 1: Write failing trace-cache tests**

Monkeypatch evaluator calls and assert:

- existing valid CSV bundles are reused unless `force_eval=True`;
- `full_controller` passes `disable_inner=False`;
- `controller_outer` passes `disable_inner=True`;
- invalid/missing portfolio columns trigger regeneration;
- Full and No-Inner dates are inner-joined and sorted.

- [ ] **Step 2: Run Task 4 tests and confirm RED**

Expected: missing `ensure_closed_loop_trace` and `align_closed_loop_returns`.

- [ ] **Step 3: Implement evaluator reuse**

Define:

```python
def ensure_closed_loop_trace(
    *,
    results_root: Path,
    output_dir: Path,
    market: str,
    seed: int,
    scenario: Literal["full_controller", "controller_outer"],
    device: str,
    force_eval: bool,
) -> dict[str, pd.DataFrame]: ...
```

Reuse `build_loaded_trainer`, `load_checkpoint_into_trainer`, `collect_eval_trace`, and `discover_runs`. Cache the returned `portfolio`, `actions`, and `switch_events` under `traces/`. Record checkpoint path/SHA256 and command JSON SHA256 in metadata.

- [ ] **Step 4: Implement closed-loop path statistics**

Define:

```python
def align_closed_loop_returns(
    full_portfolio: pd.DataFrame,
    no_inner_portfolio: pd.DataFrame,
) -> pd.DataFrame: ...

def portfolio_path_metrics(log_returns: np.ndarray) -> dict[str, float]: ...

def summarize_closed_loop(
    paired: pd.DataFrame,
    *,
    block_length: int,
    bootstrap_reps: int,
    seed: int,
) -> tuple[dict[str, float], pd.DataFrame]: ...
```

For every paired block-bootstrap draw, reconstruct both wealth paths and recompute total return, annualized Sharpe, MDD, Calmar/CR, and 5% ES. Output observed Full, observed No-Inner, difference, and paired 95% CI.

- [ ] **Step 5: Run Task 4 tests and confirm GREEN**

Run the full new test file.

- [ ] **Step 6: Commit Task 4**

```bash
git add paper_experiments/analyze_inner_outer_statistical_validation.py \
  tests/test_inner_outer_statistical_validation.py
git commit -m "Add closed loop inner ablation statistics"
```

### Task 5: Placebo, CLI, report, and figures

**Files:**
- Modify: `paper_experiments/analyze_inner_outer_statistical_validation.py`
- Modify: `tests/test_inner_outer_statistical_validation.py`

- [ ] **Step 1: Write failing placebo and CLI tests**

Assert that `permute_tilt_within_support`:

- is deterministic for a fixed seed;
- preserves each day's sorted tilt values on the nonzero base support;
- preserves zero weights outside support;
- produces normalized nonnegative executed weights or rejects an invalid permutation.

Add a CLI smoke test using temporary synthetic traces with `--skip_eval --bootstrap_reps 50 --placebo_reps 20`.

- [ ] **Step 2: Run Task 5 tests and confirm RED**

Expected: missing placebo/CLI entry points.

- [ ] **Step 3: Implement placebo**

Generate 5,000 within-support permutations by default. If `base + permuted_tilt` would be negative, resample that day up to a fixed maximum and otherwise mark the draw invalid; report the invalid-draw count. Compare observed cumulative fair net alpha, mean ex-ante risk change, and worst-market-5% return difference with the empirical placebo distribution using add-one corrected p-values.

- [ ] **Step 4: Implement CLI and artifact writing**

Arguments:

```text
--results_root
--full_actions_root
--output_dir
--markets nas sh
--seeds nas:49 sh:90
--device cpu
--transaction_cost_pct 0.00005
--risk_windows 60 20
--block_length 20
--bootstrap_reps 10000
--placebo_reps 5000
--force_eval
--skip_eval
```

Write:

- `metadata/run_manifest.json`
- Full/No-Inner traces
- five CSV tables from the design
- four PDF and PNG figures
- `INNER_OUTER_STATISTICAL_VALIDATION.md`

The report must explicitly distinguish `SUPPORTED`, `NOT SUPPORTED`, and `DESCRIPTIVE ONLY`, and state that one checkpoint per market does not measure training-seed uncertainty.

- [ ] **Step 5: Apply BH-FDR and generate plots**

Apply BH-FDR within the predefined secondary-result families, preserve raw p-values, and plot:

- Active Share, delta HHI, delta 60-day ex-ante volatility distributions
- Full/No-Inner cumulative wealth and cumulative paired log-return difference

- [ ] **Step 6: Run Task 5 tests and confirm GREEN**

Run:

```bash
/home/tongwenxuan/conda/envs/xuangu/bin/python -m pytest \
  tests/test_inner_outer_statistical_validation.py \
  tests/test_paper_experiments.py -q
```

- [ ] **Step 7: Commit Task 5**

```bash
git add paper_experiments/analyze_inner_outer_statistical_validation.py \
  tests/test_inner_outer_statistical_validation.py
git commit -m "Add inner outer validation report pipeline"
```

### Task 6: Run paper-selected verification

**Files:**
- Generate: `reproduced_outputs/inner_outer_statistical_validation/`
- Modify only if required by verified defects: analysis module/tests

- [ ] **Step 1: Run the selected-model analysis**

```bash
/home/tongwenxuan/conda/envs/xuangu/bin/python \
  paper_experiments/analyze_inner_outer_statistical_validation.py \
  --results_root reproduced_inputs/paper_selected/results_root \
  --full_actions_root reproduced_outputs/inner_daily_stats_paper_selected/_cache/inner_base_adjustment \
  --output_dir reproduced_outputs/inner_outer_statistical_validation \
  --markets nas sh \
  --seeds nas:49 sh:90 \
  --device cpu \
  --transaction_cost_pct 0.00005 \
  --risk_windows 60 20 \
  --block_length 20 \
  --bootstrap_reps 10000 \
  --placebo_reps 5000
```

Expected: exit code 0; both markets have Full and No-Inner traces and all required tables/report files.

- [ ] **Step 2: Verify model/data identities**

Compare recorded checkpoint hashes with:

```text
NAS e63e4f8d7748e89553cace37d9eb1c7718f47a9a713a9a8d48d881d55ec7de4d
SH  9abb3c8907caf8a9e999d3c4d008755ccbc81d13af71d8527140cfa8d0178f94
```

Verify dataset hashes against `REPRODUCE_CONTROLLER_INNER_FIGURES.md`.

- [ ] **Step 3: Audit required outputs**

Run a small read-only audit that checks:

- no required CSV is empty;
- all inferential rows contain effect size, sample size, raw p-value when applicable, and CI;
- no primary result was dropped;
- report values equal CSV values after documented rounding;
- mechanism invariant violations are zero or explicitly reported as a failure.

- [ ] **Step 4: Run full verification suite**

```bash
/home/tongwenxuan/conda/envs/xuangu/bin/python -m pytest \
  tests/test_inner_outer_statistical_validation.py \
  tests/test_paper_experiments.py \
  tests/test_controller_counterfactual_pg.py -q
```

Expected: zero failures.

- [ ] **Step 5: Inspect generated figures**

Open the four generated PNG files and confirm labels, units, market names, and Full/No-Inner ordering are correct.

- [ ] **Step 6: Report evidence without result shopping**

Summarize:

- whether configuration refinement is nontrivial;
- whether ex-ante risk decreases;
- whether frozen-path alpha is distinguishable from zero;
- whether closed-loop Full improves return/MDD/Sharpe relative to No-Inner;
- which claims are supported, unsupported, or descriptive only.

Do not modify hypotheses, windows, or displayed markets after seeing results.

## Verification

- Unit tests prove deterministic parsing, cost accounting, covariance timing, paired bootstrap, and placebo constraints.
- Existing paper-experiment tests guard evaluator compatibility.
- Selected-model run records hashes and complete artifacts.
- Final claims are read from generated CSV/report, not transcribed from console output.

## Next skill

Use `$superpower-executing-plans` for inline implementation. Subagents are not used because the user did not request delegation.
