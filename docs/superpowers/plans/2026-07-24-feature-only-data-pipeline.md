# Feature-Only Data Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use $superpower-subagents (recommended) or $superpower-executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking via update_plan.

**Goal:** Remove the complete SSM runtime path and make CMTFlow plus all active baselines consume model-appropriate inputs derived from `dataset["feature_path"]`.

**Architecture:** A single feature-only portfolio loader aligns stock CSVs and derives the configured adjusted fields. CMTFlow observations expose only asset histories and portfolio state, while each deep baseline keeps its own adapter: AlphaStock uses seven adjusted fields, DeepTrader uses six asset fields, and DeepAries builds raw OHLCV-compatible input before invoking its native preprocessor. Obsolete SSM models, CLI options, buffers, and rule-switch paths are deleted.

**Tech Stack:** Python 3.10, pandas, NumPy, PyTorch, pytest, Bash.

**Workspace preservation:** The repository already contains unrelated user
changes, including changes in several target files. Before every task commit,
inspect the staged diff. If a target file contains pre-existing user work that
cannot be separated safely at hunk level, do not commit that task; leave the
verified change in the working tree and report it explicitly.

---

## File Structure

- `utils/PriceMatrix.py`: feature derivation and aligned feature-only portfolio loading.
- `env/PPO_env.py`: feature-only CMTFlow environment and observations.
- `Components/PPO_model.py`: learned Controller API without SSM fallback inputs.
- `agent/PPO_agent.py`: feature-only action and PPO update paths.
- `Train/PPO_train.py`: feature-only rollout/controller records and training orchestration.
- `run_hrl_training.py`: feature-only CLI, metadata, and model construction.
- `utils/config.py`, `utils/config_Nas.py`, `utils/config_SH.py`: active data configuration without SSM paths.
- `Train/baseline.py`, `Baseline/__init__.py`: active baseline registry without SSM-only.
- `utils/PriceMatrix.py`, `Train/deep_baseline.py`: AlphaStock feature-path integration.
- `create_deeptrader_data.py`: DeepTrader-specific six-field adapter.
- `create_DeepAries_data.py`, `Train/deep_baseline.py`: DeepAries raw adapter and native preprocessing handoff.
- `train_sh/*.sh`, `scripts/*.sh`, `run_deeparies_baseline.py`: remove obsolete SSM arguments and messages where present.
- Delete `Baseline/SSM/` and `SSM_pipeline.py`.
- `tests/test_feature_only_data_pipeline.py`: focused regression tests for the new data contract.
- Existing controller, command, and end-to-end tests: update calls to the new Controller and network APIs.

### Task 1: Feature-Only Portfolio Loader

**Files:**
- Modify: `utils/PriceMatrix.py`
- Create: `tests/test_feature_only_data_pipeline.py`

- [ ] **Step 1: Write failing loader tests**

```python
import numpy as np
import pandas as pd

from utils.PriceMatrix import load_feature_files


def _write_stock(path, scale):
    pd.DataFrame({
        "Date": ["2020-01-02", "2020-01-03"],
        "open": [10.0, 11.0],
        "high": [12.0, 13.0],
        "low": [9.0, 10.0],
        "close": [11.0, 12.0],
        "volume": [100.0, 120.0],
        "adjfactor": [scale, scale],
    }).to_csv(path, index=False)


def test_load_feature_files_derives_fields_without_ssm(tmp_path):
    _write_stock(tmp_path / "AAA.csv", 2.0)
    _write_stock(tmp_path / "BBB.csv", 3.0)
    cols = ["adjopen", "adjhigh", "adjlow", "adjclose", "amount", "amp", "body"]

    loaded = load_feature_files(tmp_path, ["AAA", "BBB"], cols)

    assert loaded["data"].shape == (2, 2, 7)
    assert loaded["prices"].shape == (2, 2, 2)
    assert np.isfinite(loaded["data"]).all()
    assert set(loaded) == {"data", "prices", "dates", "id2stock", "stock2id"}
```

- [ ] **Step 2: Verify RED**

Run:

```bash
/home/tongwenxuan/conda/envs/xuangu/bin/python -m pytest \
  tests/test_feature_only_data_pipeline.py::test_load_feature_files_derives_fields_without_ssm -v
```

Expected: FAIL because `load_feature_files` does not exist.

- [ ] **Step 3: Replace the SSM loader**

In `utils/PriceMatrix.py`:

- delete `Datamatrix_ssm_hidden`;
- replace `process_files` with `load_feature_files(file_paths, stocks, feature_cols)`;
- retain `_standardize_feature_columns`;
- return only aligned `data`, `prices`, `dates`, and stock mappings;
- raise a descriptive error for a missing CSV, empty common calendar, missing derived field, or non-finite tensor.

The implementation must not read `ssm3_*` columns or `*_ssm3_states.pt`.

- [ ] **Step 4: Verify GREEN**

Run:

```bash
/home/tongwenxuan/conda/envs/xuangu/bin/python -m pytest \
  tests/test_feature_only_data_pipeline.py::test_load_feature_files_derives_fields_without_ssm -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add utils/PriceMatrix.py tests/test_feature_only_data_pipeline.py
git commit -m "refactor: add feature-only portfolio loader"
```

### Task 2: Feature-Only CMTFlow Environment

**Files:**
- Modify: `env/PPO_env.py`
- Modify: `tests/test_feature_only_data_pipeline.py`

- [ ] **Step 1: Write a failing environment contract test**

Add a focused test using a temporary dataset and patched config:

```python
def test_cmtflow_observation_contains_no_ssm(monkeypatch, tmp_path):
    env = build_tiny_feature_env(monkeypatch, tmp_path)
    obs = env.reset()

    assert "outer_state" in obs
    assert "inner_state" in obs
    assert "ssm" not in obs
    assert "held_p" not in obs
```

`build_tiny_feature_env` must create two stock CSVs, a stock-list file, and
patch `config.dataset["feature_path"]`, date bounds, and CPU device.

- [ ] **Step 2: Verify RED**

```bash
/home/tongwenxuan/conda/envs/xuangu/bin/python -m pytest \
  tests/test_feature_only_data_pipeline.py::test_cmtflow_observation_contains_no_ssm -v
```

Expected: FAIL because the environment still resolves `ssm_data_path` and emits SSM fields.

- [ ] **Step 3: Rewrite environment loading and observations**

In `env/PPO_env.py`:

- import and call `load_feature_files(dataset["feature_path"], ...)`;
- delete `h_tensor`, `z_tensor`, `p_tensor`, `q_bear_tensor`, and `q_bull_tensor`;
- delete `ssm_dict` and `held_p`;
- make `get_outer_state` and `get_inner_state` return only normalized feature tensors;
- retain feature normalization, prices, returns, portfolio state, reward, cost,
  and holding-period accounting unchanged.

- [ ] **Step 4: Verify GREEN**

Run the focused environment test and the existing environment/controller tests:

```bash
/home/tongwenxuan/conda/envs/xuangu/bin/python -m pytest \
  tests/test_feature_only_data_pipeline.py \
  tests/test_controller_counterfactual_pg.py -q
```

Expected: the new test passes; existing tests may now fail only at old Controller signatures, which Task 3 addresses.

- [ ] **Step 5: Commit**

```bash
git add env/PPO_env.py tests/test_feature_only_data_pipeline.py
git commit -m "refactor: make CMTFlow environment feature only"
```

### Task 3: Remove SSM From the Learned Controller

**Files:**
- Modify: `Components/PPO_model.py`
- Modify: `Train/PPO_train.py`
- Modify: `agent/PPO_agent.py`
- Modify: `tests/test_controller_counterfactual_pg.py`
- Modify: `tests/test_feature_only_data_pipeline.py`

- [ ] **Step 1: Write failing Controller API tests**

```python
def test_controller_accepts_asset_features_without_ssm():
    controller = MonitorAC(
        port_state_dim=6,
        hidden_dim=8,
        asset_in_dim=7,
        controller_window=5,
    )
    stats = controller.decision_stats(
        asset_state=torch.randn(2, 3, 5, 7),
        weights_drift=torch.full((2, 3), 1 / 3),
        port_state=torch.zeros(2, 6),
        switch_action=torch.full((2, 3), 1 / 3),
    )
    assert stats["exit_prob"].shape == (2,)
```

Also assert `HRL_Buffer().data` has no `"ssm"` key and a stored transition can
be batched without an SSM dictionary.

- [ ] **Step 2: Verify RED**

```bash
/home/tongwenxuan/conda/envs/xuangu/bin/python -m pytest \
  tests/test_feature_only_data_pipeline.py::test_controller_accepts_asset_features_without_ssm -v
```

Expected: FAIL because `MonitorAC` still requires `z_dim` and `h_dim`.

- [ ] **Step 3: Simplify `MonitorAC`**

In `Components/PPO_model.py`:

- remove `z_dim`, `h_dim`, and `fallback_projection`;
- make `asset_in_dim` required;
- change `decision_stats`, `encode`, `pi`, `value`, and `forward` to receive
  `asset_state`, `weights_drift`, `port_state`, and optional `switch_action`;
- make `_encode_asset_sequence` accept only `asset_state`;
- retain attention, portfolio/action features, auxiliary heads, thresholding,
  and output dictionary unchanged.

- [ ] **Step 4: Remove SSM from action/update/buffer paths**

In `agent/PPO_agent.py` and `Train/PPO_train.py`:

- remove `"ssm"` from `HRL_Buffer`;
- remove special SSM stacking logic;
- call Controller methods with feature-only keyword arguments;
- remove SSM entries from daily transitions and detached Controller records;
- update Controller auxiliary replay and counterfactual PG records accordingly.

- [ ] **Step 5: Update existing tests and verify GREEN**

Replace old Controller calls in tests with:

```python
stats = controller.decision_stats(
    asset_state=asset_state,
    weights_drift=weights,
    port_state=port_state,
    switch_action=candidate,
)
```

Run:

```bash
/home/tongwenxuan/conda/envs/xuangu/bin/python -m pytest \
  tests/test_feature_only_data_pipeline.py \
  tests/test_controller_counterfactual_pg.py \
  tests/test_controller_joint_baseline.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add Components/PPO_model.py agent/PPO_agent.py Train/PPO_train.py \
  tests/test_controller_counterfactual_pg.py tests/test_controller_joint_baseline.py \
  tests/test_feature_only_data_pipeline.py
git commit -m "refactor: remove SSM controller interface"
```

### Task 4: Remove SSM and Rule-Switch Training Configuration

**Files:**
- Modify: `run_hrl_training.py`
- Modify: `utils/config.py`
- Modify: `utils/config_Nas.py`
- Modify: `utils/config_SH.py`
- Modify: `Train/PPO_train.py`
- Modify: `train_sh/run_end_to_end_hrl_controller_joint_nas49_sh90.sh`
- Modify: relevant scripts under `train_sh/` and `scripts/`
- Modify: `tests/test_run_hrl_training_command.py`
- Modify: `tests/test_end_to_end_hrl_controller_joint_script.py`

- [ ] **Step 1: Write failing command/config tests**

Add assertions:

```python
def test_end_to_end_command_has_no_ssm_arguments(command):
    joined = " ".join(command)
    assert "--ssm_dim" not in joined
    assert "--ssm_data_path" not in joined
    assert "--nas_ssm_data_path" not in joined
    assert "--sh_ssm_data_path" not in joined


def test_runtime_config_uses_feature_path():
    assert "feature_path" in config.dataset
    assert "ssm_data_path" not in config.dataset
```

- [ ] **Step 2: Verify RED**

```bash
/home/tongwenxuan/conda/envs/xuangu/bin/python -m pytest \
  tests/test_run_hrl_training_command.py \
  tests/test_end_to_end_hrl_controller_joint_script.py -q
```

Expected: FAIL on current SSM arguments.

- [ ] **Step 3: Remove obsolete runtime configuration**

- change `HRL_Networks.__init__` to `(num_stocks, cfg)`;
- remove `ssm_dim` and all SSM path CLI arguments and metadata;
- log `dataset["feature_path"]`;
- delete `ssm_data_path`, `ssm_feature`, and `ssm_features` from the three
  active config files;
- remove rule-switch CLI/config/test execution and `held_p` diagnostics;
- remove obsolete shell variables and arguments without changing active
  end-to-end hyperparameters.

- [ ] **Step 4: Verify GREEN**

```bash
/home/tongwenxuan/conda/envs/xuangu/bin/python -m pytest \
  tests/test_run_hrl_training_command.py \
  tests/test_end_to_end_hrl_controller_joint_script.py \
  tests/test_controller_end_replay_scripts.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add run_hrl_training.py Train/PPO_train.py utils/config.py utils/config_Nas.py \
  utils/config_SH.py train_sh scripts tests/test_run_hrl_training_command.py \
  tests/test_end_to_end_hrl_controller_joint_script.py \
  tests/test_controller_end_replay_scripts.py
git commit -m "refactor: remove SSM training configuration"
```

### Task 5: Remove the SSM-Only Baseline and Pipeline

**Files:**
- Modify: `Train/baseline.py`
- Modify: `Baseline/__init__.py`
- Delete: `Baseline/SSM/run.py`
- Delete: `SSM_pipeline.py`
- Modify: `tests/test_feature_only_data_pipeline.py`

- [ ] **Step 1: Write a failing baseline registry test**

```python
def test_baseline_registry_has_no_ssm_only_model():
    source = Path("Train/baseline.py").read_text()
    assert "baseline_ssm" not in source
    assert not Path("Baseline/SSM").exists()
    assert not Path("SSM_pipeline.py").exists()
```

- [ ] **Step 2: Verify RED**

```bash
/home/tongwenxuan/conda/envs/xuangu/bin/python -m pytest \
  tests/test_feature_only_data_pipeline.py::test_baseline_registry_has_no_ssm_only_model -v
```

Expected: FAIL because the SSM-only baseline and pipeline exist.

- [ ] **Step 3: Delete the obsolete code**

- remove `baseline_ssm` import/export/call;
- delete `Baseline/SSM/`;
- delete `SSM_pipeline.py`;
- retain the current test tree unchanged here because repository search finds
  no test importing `SSM_pipeline`, `baseline_ssm`, or `Baseline.SSM`.

- [ ] **Step 4: Verify GREEN**

Run the focused test and import check:

```bash
/home/tongwenxuan/conda/envs/xuangu/bin/python -m pytest \
  tests/test_feature_only_data_pipeline.py::test_baseline_registry_has_no_ssm_only_model -v
/home/tongwenxuan/conda/envs/xuangu/bin/python -c \
  "from Train.baseline import baseline; from Baseline import baseline_BH"
```

Expected: PASS and clean import.

- [ ] **Step 5: Commit**

```bash
git add Train/baseline.py Baseline tests/test_feature_only_data_pipeline.py
git add -u SSM_pipeline.py
git commit -m "refactor: remove obsolete SSM baseline pipeline"
```

### Task 6: AlphaStock and DeepTrader Feature Adapters

**Files:**
- Modify: `Train/deep_baseline.py`
- Modify: `utils/PriceMatrix.py`
- Modify: `create_deeptrader_data.py`
- Modify: `tests/test_feature_only_data_pipeline.py`

- [ ] **Step 1: Write failing adapter tests**

```python
def test_alphastock_files_derives_nasdaq_fields(tmp_path):
    _write_stock(tmp_path / "AAA.csv", 2.0)
    data, dates, _, _, prices = alphastock_files(
        tmp_path,
        ["AAA"],
        ["adjopen", "adjhigh", "adjlow", "adjclose", "amount", "amp", "body"],
    )
    assert data.shape[-1] == 7
    assert np.isfinite(data).all()


def test_deeptrader_uses_six_model_specific_fields(monkeypatch, tmp_path):
    outputs = deeptrader_files(output_path=tmp_path / "out")
    features = np.load(tmp_path / "out" / "features.npy")
    assert features.shape[-1] == 6
```

The test fixture must provide both `feature_path` and a deliberately invalid
`ssm_data_path`; success proves the adapter did not resolve the old path.

- [ ] **Step 2: Verify RED**

Run the two focused tests. Expected: AlphaStock raises missing-column `KeyError`
and DeepTrader resolves `ssm_data_path` or emits seven features.

- [ ] **Step 3: Implement AlphaStock adaptation**

- remove the `feature_path = ssm_data_path` override in `Train/deep_baseline.py`;
- call `_standardize_feature_columns(df, feature_cols)` in `alphastock_files`
  before selecting features and prices.

- [ ] **Step 4: Implement DeepTrader adaptation**

In `create_deeptrader_data.py`:

- resolve only `dataset["feature_path"]`;
- define:

```python
DEEPTRADER_FEATURES = [
    "adjopen", "adjhigh", "adjlow", "adjclose", "amount", "amp"
]
```

- standardize each stock DataFrame before computing returns and extracting
  features;
- preserve date intersection, return calculation, relation matrix, and split
  files;
- accept an optional `output_path` to make the adapter testable without
  mutating configured output directories.

- [ ] **Step 5: Verify GREEN**

```bash
/home/tongwenxuan/conda/envs/xuangu/bin/python -m pytest \
  tests/test_feature_only_data_pipeline.py -k "alphastock or deeptrader" -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add Train/deep_baseline.py utils/PriceMatrix.py create_deeptrader_data.py \
  tests/test_feature_only_data_pipeline.py
git commit -m "fix: load AlphaStock and DeepTrader from feature data"
```

### Task 7: Restore Native DeepAries Feature Processing

**Files:**
- Modify: `create_DeepAries_data.py`
- Modify: `Train/deep_baseline.py`
- Modify: `run_deeparies_baseline.py`
- Modify: `tests/test_feature_only_data_pipeline.py`

- [ ] **Step 1: Write a failing DeepAries handoff test**

```python
def test_deeparies_builds_raw_ohlcv_then_native_features(tmp_path):
    summary = save_deeparies_data(
        market="nas",
        output_root=tmp_path,
        feature_path=feature_dir,
        stocks_path=stocks_file,
        start_date="2020-01-02",
        end_date="2020-01-03",
    )
    raw = pd.read_csv(summary["raw_path"])
    assert list(raw.columns) == [
        "date", "tic", "open", "high", "low", "close", "adjclose", "volume"
    ]
    assert not Path(summary["processed_path"]).exists()

    YfinancePreprocessor(
        summary["raw_path"], summary["processed_path"]
    ).make_feature()
    processed = pd.read_csv(summary["processed_path"])
    assert {"zopen", "zhigh", "zlow", "zclose", "zd_5", "zd_60"} <= set(processed)
```

Add a second fixture containing only SH-style adjusted fields. Its expected raw
mapping is:

```text
open=adjopen, high=adjhigh, low=adjlow, close=adjclose,
adjclose=adjclose, volume=amount/adjclose
```

- [ ] **Step 2: Verify RED**

Run the DeepAries tests. Expected: FAIL because the current adapter writes
seven adjusted features and pre-creates the processed file.

- [ ] **Step 3: Rewrite the raw adapter**

In `create_DeepAries_data.py`:

- resolve only `feature_path`;
- replace the shared seven-field export with
  `build_deeparies_raw_dataframe`;
- for NAS-style files use raw OHLCV and adjusted close;
- for SH-style files derive the raw-compatible mapping above;
- write only `<market>_data.csv`;
- remove a stale generated `<market>_general_data.csv` inside the run-specific
  output directory before launching DeepAries;
- retain dates, stock alignment, and summary metadata.

Update `Train/deep_baseline.py` and `run_deeparies_baseline.py` messages to say
`feature_path`, then let `DeepAries/main.py` invoke `YfinancePreprocessor`.

- [ ] **Step 4: Verify GREEN**

```bash
/home/tongwenxuan/conda/envs/xuangu/bin/python -m pytest \
  tests/test_feature_only_data_pipeline.py -k deeparies -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add create_DeepAries_data.py Train/deep_baseline.py run_deeparies_baseline.py \
  tests/test_feature_only_data_pipeline.py
git commit -m "fix: restore native DeepAries feature preprocessing"
```

### Task 8: Full Regression and Smoke Verification

**Files:**
- Modify only files required by failures found in this task.

- [ ] **Step 1: Audit active runtime references**

Run:

```bash
rg -n "feature_ssm|ssm_data_path|ssm_dim|ssm3_|held_p|use_rule_switch|baseline_ssm" \
  Components env agent Train Baseline utils run_hrl_training.py \
  create_deeptrader_data.py create_DeepAries_data.py run_deeparies_baseline.py \
  train_sh scripts --glob '*.py' --glob '*.sh'
```

Expected: no active runtime matches.

- [ ] **Step 2: Run focused regression tests**

```bash
/home/tongwenxuan/conda/envs/xuangu/bin/python -m pytest \
  tests/test_feature_only_data_pipeline.py \
  tests/test_controller_counterfactual_pg.py \
  tests/test_controller_joint_baseline.py \
  tests/test_run_hrl_training_command.py \
  tests/test_end_to_end_hrl_controller_joint_script.py \
  tests/test_controller_end_replay_scripts.py -q
```

Expected: PASS.

- [ ] **Step 3: Run feature-only CMTFlow smoke training**

```bash
CUDA_VISIBLE_DEVICES=0 \
/home/tongwenxuan/conda/envs/xuangu/bin/python run_hrl_training.py \
  --markets nas sh \
  --smoke \
  --output_root results/feature_only_smoke \
  --run_name feature_only_pipeline \
  --continue_on_error
```

Expected: both markets construct from `feature_path`, perform at least one
training update, write checkpoints/results, and emit no SSM-path warning.

- [ ] **Step 4: Run deep-baseline adapter smoke checks**

Run the adapter tests plus the existing DeepAries smoke entry with one epoch
and a small stock subset. Expected: AlphaStock data construction, DeepTrader
NPY generation, and DeepAries native preprocessing all complete without
reading `feature_ssm`.

- [ ] **Step 5: Inspect the final diff**

```bash
git diff --check
git status --short
```

Expected: no whitespace errors; only intended source/test deletions and the
user's pre-existing unrelated changes remain.

- [ ] **Step 6: Commit final verification fixes**

```bash
git add Components env agent Train Baseline utils run_hrl_training.py \
  create_deeptrader_data.py create_DeepAries_data.py run_deeparies_baseline.py \
  train_sh scripts tests
git commit -m "test: verify feature-only training pipeline"
```
