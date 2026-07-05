# CMTFlow 20 分钟中文汇报 PPT 大纲

本大纲用于 `codex-ppt` 生成整页图片式 PPT。目标是做一版视觉上接近 `SWAIB_TransG_slides.pptx` 模板、后期不需要太多修改的中文学术报告。模板风格参考 `assets/style/template_thumbnail.jpeg`：白底、深蓝边框/角标、红色强调标题、黑色正文、学术报告式紧凑排版。

重要说明：以下列出的本地图片是严格输入资产。生成时应保留图中数据、曲线、坐标轴、图例、标签、颜色和数值，不要重绘或替换为相似图。

## Slide 1: 标题页

- 页面角色：cover
- 关键点：
  - 标题：CMTFlow：控制器引导的分层投资组合管理框架
  - 副标题：Hierarchical Portfolio Management with Controller-Guided Base Revision and Daily Refinement
  - 作者、单位、日期占位
- 视觉想法：仿模板封面，白底、深蓝斜角边框、红色关键词 CMTFlow。
- Required images：无。

## Slide 2: 研究背景：投资组合管理的基本目标

- 页面角色：context / background
- 关键点：
  - 投资组合管理是在不确定市场中进行多资产资本配置。
  - 目标不仅是提高收益，还要控制波动、回撤和交易成本。
  - 强化学习适合长期序列决策，但金融市场状态持续变化。
- 视觉想法：收益、风险、成本三角结构，右侧用小型曲线表示非平稳市场。
- Required images：无。

## Slide 3: 研究背景：组合决策不是单一日频动作

- 页面角色：background / process
- 关键点：
  - 实际投资包含中期持仓、每日修正、异常状态退出。
  - 固定周期再平衡稳定但反应慢。
  - 纯日频调仓灵活但容易噪声化、换手过高。
  - 需要把持仓段和每日微调分开建模。
- 视觉想法：三阶段循环：Observe market -> Decide hold/switch -> Execute weights。
- Required images：无。

## Slide 4: 问题定义：带漂移和交易成本的动态组合

- 页面角色：problem definition
- 关键点：
  - 昨日组合经过价格变化后会产生 drifted portfolio。
  - 当日重新配置会带来换手和交易成本。
  - 策略目标是在长期收益、风险和成本之间取得平衡。
  - 指标包括 Total Return、Annual Return、Sharpe、MDD、CR。
- 视觉想法：横向公式流程：w_{t-1} -> drifted weights -> rebalance -> w_t -> reward。
- Required images：无。

## Slide 5: 问题定义：本文关注的三个核心决策

- 页面角色：concept explanation
- 关键点：
  - When：当前 base portfolio 是否已经失效？
  - What：如果切换，新的中期基准组合是什么？
  - How：在基准组合内部，如何每日权重微调？
  - 三个问题对应不同时间尺度，不适合压缩进单一动作。
- 视觉想法：三个并列问题卡片，对应 Controller、Outer Actor、Inner Actor。
- Required images：无。

## Slide 6: 相关工作：从静态优化到深度强化学习

- 页面角色：related work / comparison
- 关键点：
  - 传统投资组合优化依赖静态假设或单期目标。
  - 端到端 DRL 常将所有决策压缩成每日权重输出。
  - 层级或自适应方法通常缺少显式 base portfolio 记忆。
  - 本文从“基准组合何时被修正”切入。
- 视觉想法：三列对比：Classical Optimization、Deep RL Portfolio、Adaptive / HRL。
- Required images：无。

## Slide 7: 挑战：直接应用 RL 仍存在非平凡限制

- 页面角色：challenge
- 关键点：
  - 日频动作容易被短期噪声驱动。
  - 固定再平衡无法识别持仓状态是否已经恶化。
  - 换仓需要同时比较旧组合、候选组合、交易成本和持仓年龄。
  - 因此需要显式 controller 学习 hold/switch。
- 视觉想法：左侧列出三个挑战，右侧突出 Controller 的 hold/switch 决策点。
- Required images：无。

## Slide 8: 方案总览：CMTFlow 的统一分层结构

- 页面角色：overview / architecture
- 关键点：
  - CMTFlow 包含 Outer、Controller、Inner 三个核心模块。
  - Outer 生成候选 base portfolio。
  - Controller 判断 hold 或 switch。
  - Inner 围绕当前 base portfolio 做每日微调。
- 视觉想法：主体为架构图，右侧或底部加 3 个简短 callout。
- Required images:
  - CMTFlow overall architecture；strict input asset；preserve labels, modules, arrows, and hierarchy.

    ![CMTFlow Architecture](assets/figures/slide08_architecture.png)

## Slide 9: 强化学习建模：状态、动作与奖励

- 页面角色：formulation
- 关键点：
  - State：近期市场张量、漂移当前持仓、候选组合、holding-state vector、action-comparison features。
  - Action：候选基准组合、hold/switch 决策、最终执行权重。
  - Reward：组合 log return 扣除交易成本。
  - 目标是优化完整投资路径上的风险收益质量。
- 视觉想法：POMDP/MDP 风格闭环图。
- Required images：无。

## Slide 10: 方法架构：从候选组合到最终执行权重

- 页面角色：methodology / process
- 关键点：
  - Outer 生成候选组合 w_t^{cand}。
  - Controller 输出 switch probability，并决定是否替换当前 base。
  - Base selector 得到实际使用的 b_t。
  - Inner 输出最终执行权重 w_t。
- 视觉想法：主体为决策流程图，旁边突出公式 b_t = w_t^{cand} if switch else \tilde{b}_t。
- Required images:
  - CMTFlow decision flow；strict input asset；preserve branch logic, formulas, labels, and arrows.

    ![CMTFlow Decision Flow](assets/figures/slide10_decision_flow.png)

## Slide 11: 方法细节：Outer Actor 生成中期基准组合

- 页面角色：method detail
- 关键点：
  - Outer 面向持仓段级别决策，而不是每日最终交易。
  - 输入长期市场窗口和当前持仓漂移状态。
  - 输出稀疏 top-K candidate base portfolio。
  - 回答 What to hold next。
- 视觉想法：使用三模块图，并对 Outer 区域加 callout。
- Required images:
  - Three-module mechanism figure；strict input asset；preserve module names and relationships.

    ![CMTFlow Three Modules](assets/figures/slide11_three_modules.png)

## Slide 12: 方法细节：Controller 学习何时替换基准组合

- 页面角色：method detail / key mechanism
- 关键点：
  - Controller 是 CMTFlow 的核心自适应机制。
  - 输入近期市场张量、漂移当前持仓、候选组合、holding-state vector 和 action-comparison features。
  - 输出 exit probability 或 hold/switch 决策。
  - 把固定日历调仓变成状态依赖事件策略。
- 视觉想法：复用决策流程图，突出 Controller 分叉点。
- Required images:
  - Decision flow figure；strict input asset；preserve controller branch and labels.

    ![CMTFlow Decision Flow](assets/figures/slide10_decision_flow.png)

## Slide 13: 方法细节：Inner Actor 做每日局部微调

- 页面角色：method detail
- 关键点：
  - Inner 不重新决定股票池，而是在 active base 内调整权重。
  - 提供日频灵活性，但避免完全脱离中期基准组合。
  - 作用是局部 refinement，而不是主要换仓控制器。
- 视觉想法：复用三模块图，突出 Inner 局部 tilt。
- Required images:
  - Three-module mechanism figure；strict input asset；preserve module names and relationships.

    ![CMTFlow Three Modules](assets/figures/slide11_three_modules.png)

## Slide 14: 模型训练：固定 HRL 预训练与 Controller 学习

- 页面角色：training flow
- 关键点：
  - 先训练 fixed-segment HRL backbone。
  - 再训练每日 hold/switch Controller。
  - 最终测试不使用固定再平衡周期。
  - PPO 与辅助信号共同优化策略。
- 视觉想法：主体为训练流程图，右侧列出三阶段训练。
- Required images:
  - CMTFlow training flow；strict input asset；preserve stages, arrows, and labels.

    ![CMTFlow Training Flow](assets/figures/slide14_training_flow.png)

## Slide 15: 实验设置

- 页面角色：experimental setup
- 关键点：
  - 数据集：Nasdaq-100 与 CSI-300。
  - 时间顺序划分训练、验证和测试区间。
  - Baseline 包括传统策略、深度 RL 策略和固定窗口控制器。
  - 指标包括 TR、AR、Vol、Sharpe、MDD、CR。
- 视觉想法：数据集表格 + baseline/metric 卡片。
- Required images：无。

## Slide 16: 数值结果：主实验性能对比

- 页面角色：data evidence
- 关键点：
  - CMTFlow 在 Nasdaq-100 上取得最高 matched total return。
  - 在 CSI-300 上提升 Sharpe，并降低最大回撤。
  - 结果体现高收益和可控回撤的综合权衡。
  - 不表述为每个单项指标都绝对最优。
- 视觉想法：左侧累计财富曲线，右侧主指标柱状图，底部一句 takeaway。
- Required images:
  - Main equity curves；strict input asset；preserve all data, axes, labels, curves, legends, and colors.

    ![Main Equity Curves](assets/figures/slide16_main_equity_curves.png)

  - Main metric bars；strict input asset；preserve all data, axes, labels, bars, legends, and colors.

    ![Main Metric Bars](assets/figures/slide16_main_metric_bars.png)

## Slide 17: 数值结果：消融实验与机制验证

- 页面角色：data evidence / ablation
- 关键点：
  - Learned controller 优于固定窗口切换策略。
  - Controller 是主要自适应来源。
  - Inner Actor 更偏局部调权和风险控制。
  - 提升来自学习何时修正 base portfolio。
- 视觉想法：主体为消融指标图，旁边突出 Controller contribution。
- Required images:
  - Ablation metric bars；strict input asset；preserve all model names, metrics, values, bars, axes, labels, and colors.

    ![Ablation Metric Bars](assets/figures/slide17_ablation_metric_bars.png)

## Slide 18: 案例研究：Controller 与 Inner 的可解释行为

- 页面角色：case study / interpretability
- 关键点：
  - Controller 可在旧持仓恶化前触发切换，降低后续回撤。
  - 固定持仓窗口对比说明收益不是来自某个固定再平衡周期。
  - Inner Actor 在基准组合内部进行有方向的局部 tilt。
  - 该页负责把数值结果翻译成经济直觉。
- 视觉想法：三张图并排，分别为 switch case、fixed-window comparison、inner adjustment。
- Required images:
  - Controller switch cases；strict input asset；preserve all panels, labels, axes, curves, and annotations.

    ![Controller Switch Cases](assets/figures/slide18_controller_switch_cases.png)

  - Fixed-window comparison；strict input asset；preserve all panels, labels, axes, curves/bars, values, and colors.

    ![Fixed Window Comparison](assets/figures/slide18_fixed_window_comparison.png)

  - Inner actor base adjustment；strict input asset；preserve all panels, labels, axes, heatmaps/curves, and annotations.

    ![Inner Actor Base Adjustment](assets/figures/slide18_inner_actor_base_adjustment.png)

## Slide 19: 讨论与总结

- 页面角色：summary / conclusion
- 关键点：
  - CMTFlow 将组合管理拆成 base revision、base construction 和 daily refinement。
  - Controller 让换仓从固定周期规则变成可学习事件策略。
  - 实验和消融表明 controller 是动态修正行为的主要来源。
  - 未来工作：更强市场泛化、更复杂交易约束、更稳健风险控制。
- 视觉想法：三个贡献卡片 + 一条 future work。
- Required images：无。

## Slide 20: Q&A

- 页面角色：Q&A
- 关键点：
  - 谢谢，欢迎提问。
  - Takeaway：Learning when to revise is as important as learning what to hold.
- 视觉想法：仿模板结束页，白底深蓝边框、居中大标题。
- Required images：无。
