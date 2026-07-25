# 仅使用 Feature 的数据链重写实施计划

> 实施时按任务顺序执行，每一项都遵循“先写失败测试，再进行最小修改，最后运行回归测试”。所有代码修改只保留在本地工作区，不执行 Git 提交或推送。

## 目标

彻底移除 CMTFlow 和当前 baseline 运行链中的 SSM 依赖，使所有有效模型从
`config.dataset["feature_path"]` 获取数据，并由各模型自己的适配器生成所需输入。

## 总体设计

- CMTFlow 使用统一的普通特征加载器，输入只包含特征、交易价格、日期和股票映射。
- Outer Actor、Inner Actor 和 learned Controller 均使用普通资产特征窗口。
- Controller 删除 `h/z/p/q_bear/q_bull` 输入及 SSM fallback。
- AlphaStock、DeepTrader 和 DeepAries 分别保留自己的字段和预处理方式。
- 删除 SSM-only baseline、SSM pipeline、rule-switch 及相关参数。
- 不兼容旧 checkpoint，新模型必须重新训练。
- 保护工作区已有修改，只编辑本任务涉及的代码，不执行 Git 操作。

---

## 任务一：建立仅使用 Feature 的统一加载器

### 涉及文件

- 修改：`utils/PriceMatrix.py`
- 新增：`tests/test_feature_only_data_pipeline.py`

### 实施步骤

1. 新增失败测试，使用临时 Nasdaq 风格 CSV：

```python
def test_load_feature_files_derives_adjusted_fields(tmp_path):
    # CSV 只提供 OHLCV 和 adjfactor
    # 加载后必须生成：
    # adjopen、adjhigh、adjlow、adjclose、amount、amp、body
    loaded = load_feature_files(tmp_path, ["AAA", "BBB"], FEATURE_COLUMNS)
    assert loaded["data"].shape == (2, 2, 7)
    assert loaded["prices"].shape == (2, 2, 2)
    assert np.isfinite(loaded["data"]).all()
    assert set(loaded) == {
        "data", "prices", "dates", "id2stock", "stock2id"
    }
```

2. 运行测试，确认因 `load_feature_files` 尚不存在而失败：

```bash
/home/tongwenxuan/conda/envs/xuangu/bin/python -m pytest \
  tests/test_feature_only_data_pipeline.py::test_load_feature_files_derives_adjusted_fields -v
```

3. 在 `utils/PriceMatrix.py` 中：

   - 保留 `_standardize_feature_columns()`；
   - 删除 `Datamatrix_ssm_hidden()`；
   - 用 `load_feature_files(file_paths, stocks, feature_cols)` 取代 `process_files()`；
   - 只读取每只股票的 CSV；
   - 对齐所有股票的共同日期；
   - 返回特征、`adjopen/adjclose`、日期和股票映射；
   - 不读取 `ssm3_*` 列和 `*_ssm3_states.pt`；
   - 缺少文件、字段不可推导、共同日期为空或输出存在非有限值时，给出明确错误。

4. 重新运行测试，确认通过。

---

## 任务二：将 CMTFlow 环境切换到 Feature

### 涉及文件

- 修改：`env/PPO_env.py`
- 修改：`tests/test_feature_only_data_pipeline.py`

### 实施步骤

1. 新增环境契约测试：

```python
def test_cmtflow_observation_has_no_ssm(monkeypatch, tmp_path):
    env = build_tiny_feature_env(monkeypatch, tmp_path)
    obs = env.reset()
    assert "outer_state" in obs
    assert "inner_state" in obs
    assert "ssm" not in obs
    assert "held_p" not in obs
```

2. 运行测试，确认当前代码因为读取 `ssm_data_path` 或返回 `ssm` 而失败。

3. 修改 `PPO_Env`：

   - 调用 `load_feature_files(dataset["feature_path"], ...)`；
   - 删除 `h_tensor`、`z_tensor`、`p_tensor`、`q_bear_tensor`、`q_bull_tensor`；
   - observation 删除 `ssm` 和 `held_p`；
   - `get_outer_state()` 只返回标准化后的 Outer 特征窗口；
   - `get_inner_state()` 只返回标准化后的 Inner 特征窗口；
   - 保留价格漂移、收益、交易成本、持仓年龄、回撤和反事实收益计算。

4. 运行新环境测试和现有 Controller 测试，记录后续仅由 Controller 旧接口导致的失败。

---

## 任务三：删除 Controller 的 SSM 接口

### 涉及文件

- 修改：`Components/PPO_model.py`
- 修改：`agent/PPO_agent.py`
- 修改：`Train/PPO_train.py`
- 修改：`tests/test_controller_counterfactual_pg.py`
- 修改：`tests/test_controller_joint_baseline.py`
- 修改：`tests/test_feature_only_data_pipeline.py`

### 实施步骤

1. 新增失败测试，期望 Controller 只接受普通特征：

```python
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

2. 确认测试因 `z_dim/h_dim` 必填而失败。

3. 修改 `MonitorAC`：

   - 删除 `z_dim`、`h_dim` 和 `fallback_projection`；
   - `asset_in_dim` 改为必要参数；
   - `decision_stats()`、`pi()`、`value()`、`forward()` 只接收：

```text
asset_state
weights_drift
port_state
switch_action
```

   - 保留两阶段注意力、组合状态、候选组合差异、切换概率、收益/风险/优势辅助头。

4. 修改 `HRL_Buffer`：

   - 删除 `"ssm"` 数据项；
   - 删除 SSM 字典的特殊堆叠逻辑。

5. 修改 Agent 和 Trainer：

   - 所有 Controller 前向调用只传普通特征；
   - transition 删除 `ssm`；
   - Controller PG、辅助预训练和反事实记录删除 SSM 字段；
   - 保持 Controller reward、switch advantage 和验证逻辑不变。

6. 更新现有测试中的 Controller 调用。

7. 运行：

```bash
/home/tongwenxuan/conda/envs/xuangu/bin/python -m pytest \
  tests/test_feature_only_data_pipeline.py \
  tests/test_controller_counterfactual_pg.py \
  tests/test_controller_joint_baseline.py -q
```

预期全部通过。

---

## 任务四：删除训练入口中的 SSM 和 Rule-Switch 参数

### 涉及文件

- 修改：`run_hrl_training.py`
- 修改：`utils/config.py`
- 修改：`utils/config_Nas.py`
- 修改：`utils/config_SH.py`
- 修改：`Train/PPO_train.py`
- 修改：`train_sh/` 和 `scripts/` 下当前仍在使用的训练脚本
- 修改：`tests/test_run_hrl_training_command.py`
- 修改：`tests/test_end_to_end_hrl_controller_joint_script.py`
- 修改：`tests/test_controller_end_replay_scripts.py`

### 实施步骤

1. 新增失败断言：

```python
joined = " ".join(command)
assert "--ssm_dim" not in joined
assert "--ssm_data_path" not in joined
assert "--nas_ssm_data_path" not in joined
assert "--sh_ssm_data_path" not in joined
```

同时检查：

```python
assert "feature_path" in config.dataset
assert "ssm_data_path" not in config.dataset
```

2. 运行相关测试，确认当前命令仍包含旧参数。

3. 修改训练入口：

   - `HRL_Networks` 构造函数改为 `(num_stocks, cfg)`；
   - 删除 `ssm_dim`；
   - 删除 `ssm_data_path` 及两个市场的覆盖参数；
   - 运行日志和 metadata 改为记录 `feature_path`。

4. 修改三个配置：

   - 删除 `ssm_data_path`；
   - 删除 `ssm_feature`；
   - 删除 `ssm_features`；
   - 保留 `feature_path`、股票池、七个模型输入字段和交易成本配置。

5. 删除 rule-switch：

   - 删除 `use_rule_switch` 分支；
   - 删除 `rule_switch_threshold`、连续低 `held_p` 等参数；
   - 删除 Scenario 4 rule-switch 测试；
   - 保留 learned Controller 和固定周期测试。

6. 清理 shell 脚本中的旧参数，但不改变现有 epoch、窗口、seed、成本和输出路径逻辑。

7. 运行：

```bash
/home/tongwenxuan/conda/envs/xuangu/bin/python -m pytest \
  tests/test_run_hrl_training_command.py \
  tests/test_end_to_end_hrl_controller_joint_script.py \
  tests/test_controller_end_replay_scripts.py -q
```

---

## 任务五：移除 SSM-only Baseline 和旧 Pipeline

### 涉及文件

- 修改：`Train/baseline.py`
- 修改：`Baseline/__init__.py`
- 删除：`Baseline/SSM/`
- 删除：`SSM_pipeline.py`
- 修改：`tests/test_feature_only_data_pipeline.py`

### 实施步骤

1. 新增失败测试：

```python
source = Path("Train/baseline.py").read_text()
assert "baseline_ssm" not in source
assert not Path("Baseline/SSM").exists()
assert not Path("SSM_pipeline.py").exists()
```

2. 确认测试失败。

3. 删除：

   - `baseline_ssm` 的导入、导出和执行；
   - `Baseline/SSM/`；
   - `SSM_pipeline.py`。

4. 验证 baseline 入口仍能正常导入：

```bash
/home/tongwenxuan/conda/envs/xuangu/bin/python -c \
  "from Train.baseline import baseline; from Baseline import baseline_BH"
```

---

## 任务六：适配 AlphaStock 和 DeepTrader

### 涉及文件

- 修改：`Train/deep_baseline.py`
- 修改：`utils/PriceMatrix.py`
- 修改：`create_deeptrader_data.py`
- 修改：`tests/test_feature_only_data_pipeline.py`

### AlphaStock

1. 新增 Nasdaq 风格字段推导测试。
2. 删除 `_run_alphastock()` 中将 `feature_path` 强制改成 `ssm_data_path` 的代码。
3. 在 `alphastock_files()` 选择字段前调用 `_standardize_feature_columns()`。
4. 输入字段保持：

```text
adjopen, adjhigh, adjlow, adjclose, amount, amp, body
```

### DeepTrader

1. 新增测试，配置一个无效的 `ssm_data_path`，确认模型仍从 `feature_path` 成功生成输入。
2. 固定 DeepTrader 自己的六维输入：

```python
DEEPTRADER_FEATURES = [
    "adjopen",
    "adjhigh",
    "adjlow",
    "adjclose",
    "amount",
    "amp",
]
```

3. `create_deeptrader_data.py`：

   - 只读取 `dataset["feature_path"]`；
   - 每只股票先做字段适配；
   - 再生成 `features.npy`、`rets.npy`、`relation.npy` 和 `split_idx.txt`；
   - 增加可选输出目录参数，测试时不污染正式目录。

4. 验证：

```bash
/home/tongwenxuan/conda/envs/xuangu/bin/python -m pytest \
  tests/test_feature_only_data_pipeline.py -k "alphastock or deeptrader" -v
```

---

## 任务七：恢复 DeepAries 原生特征处理

### 涉及文件

- 修改：`create_DeepAries_data.py`
- 修改：`Train/deep_baseline.py`
- 修改：`run_deeparies_baseline.py`
- 修改：`tests/test_feature_only_data_pipeline.py`

### 实施步骤

1. 新增测试，要求适配器只输出原始输入：

```text
date, tic, open, high, low, close, adjclose, volume
```

2. Nasdaq 风格文件直接使用原始 OHLCV 和 `adjclose`。

3. SH 风格文件缺少原始 OHLCV 时采用：

```text
open     = adjopen
high     = adjhigh
low      = adjlow
close    = adjclose
adjclose = adjclose
volume   = amount / adjclose
```

4. `create_DeepAries_data.py` 只生成 `<market>_data.csv`，不再提前生成
`<market>_general_data.csv`。

5. 在本次运行自己的输出目录中删除陈旧的 processed 文件，使
`DeepAries/main.py` 必定调用其原生 `YfinancePreprocessor`。

6. 测试原生处理后必须出现：

```text
zopen, zhigh, zlow, zadjcp, zclose,
zd_5, zd_10, zd_15, zd_20, zd_25, zd_30, zd_60
```

7. 运行：

```bash
/home/tongwenxuan/conda/envs/xuangu/bin/python -m pytest \
  tests/test_feature_only_data_pipeline.py -k deeparies -v
```

---

## 任务八：完整回归和 Smoke 验证

### 1. 检查旧运行引用

```bash
rg -n \
  "feature_ssm|ssm_data_path|ssm_dim|ssm3_|held_p|use_rule_switch|baseline_ssm" \
  Components env agent Train Baseline utils run_hrl_training.py \
  create_deeptrader_data.py create_DeepAries_data.py run_deeparies_baseline.py \
  train_sh scripts \
  --glob '*.py' --glob '*.sh'
```

预期：活动代码中没有匹配。

### 2. 运行重点测试

```bash
/home/tongwenxuan/conda/envs/xuangu/bin/python -m pytest \
  tests/test_feature_only_data_pipeline.py \
  tests/test_controller_counterfactual_pg.py \
  tests/test_controller_joint_baseline.py \
  tests/test_run_hrl_training_command.py \
  tests/test_end_to_end_hrl_controller_joint_script.py \
  tests/test_controller_end_replay_scripts.py -q
```

### 3. CMTFlow 双市场 Smoke

```bash
CUDA_VISIBLE_DEVICES=0 \
/home/tongwenxuan/conda/envs/xuangu/bin/python run_hrl_training.py \
  --markets nas sh \
  --smoke \
  --output_root results/feature_only_smoke \
  --run_name feature_only_pipeline \
  --continue_on_error
```

预期：

- 两个市场都从 `feature_path` 建立环境；
- 至少完成一次训练更新；
- 正常写出 checkpoint 和测试结果；
- 不出现 SSM 文件或路径错误。

### 4. Baseline 适配器 Smoke

- AlphaStock 能完成两个市场的数据构造；
- DeepTrader 输出最后一维为6；
- DeepAries 确实经过原生预处理器；
- 全部输入均来自 `feature`。

### 5. 最终检查

```bash
git diff --check
git status --short
```

这里只用于检查改动，不执行暂存、提交或推送。最终向用户列出：

- 修改和删除的文件；
- 通过的测试；
- smoke 结果；
- 仍需完整重训的模型。
