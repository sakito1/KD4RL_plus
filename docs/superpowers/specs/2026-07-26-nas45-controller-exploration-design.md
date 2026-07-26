# NAS seed 45 Controller 探索设计

## 目标

以 Nasdaq-100 seed 45 的 `hrl_fixed_best.pth` 为唯一 Outer+Inner
起点，冻结 Outer Actor 和 Inner Actor，仅训练 Controller。主要成功指标
是验证集累计收益相对固定 30 日 Outer+Inner 的提升；MDD、切换次数和
概率分布只用于排查异常。测试集不参与方案选择。

## 固定起点

- Checkpoint:
  `results/outer_inner_seed_sweep_k5/outer_inner_4_3_2_k5/nas/ppo/seed_45/checkpoints/hrl_fixed_best.pth`
- 市场：Nasdaq-100
- `trade_num=5`
- 交易成本：`0.0001`
- 最大持有期：30 个交易日
- Outer+Inner 测试终值仅作为既有记录，不用于 Controller 调参。

## 对照方案

脚本提供四种模式，输出目录相互隔离：

1. `probe`：只检查训练集经济标签，不训练。
2. `pg_only`：无监督预训练，直接训练 Controller PG，作为对照。
3. `sup_only`：只进行监督预训练，用于检查概率是否学出区分度。
4. `sup_pg`：监督预训练后进行 PG，作为推荐主方案。

## 主方案参数

- Controller rollout：300 日。
- 每日自由决策，最大持有期仍为 30 日。
- 风险阈值和成本调整后候选优势阈值均为 5%。
- 连续满足经济规则的日期全部标记为 switch label 1。
- 使用类别平衡 BCE 训练 switch label。
- 风险与优势辅助目标 scale 均为 20。
- Controller exit bias 初始化为 0。
- 监督预训练：采集 1 次，缓存后 replay 30 次。
- PG：3 epochs。
- PG 奖励：成本调整后的 switch--hold return uplift。
- Controller checkpoint 按验证集累计收益选择。

## 运行与筛选

先使用 Controller seed 45 顺序运行 `probe`、`pg_only`、`sup_only` 和
`sup_pg`。比较验证集累计收益、自由切换次数、切换概率分布以及
switch advantage。

## 输出要求

每种模式使用独立的 `run_name` 和结果目录，保留命令元数据、训练日志、
验证指标、诊断统计及最终 Controller checkpoint，不覆盖原始
Outer+Inner 文件。
