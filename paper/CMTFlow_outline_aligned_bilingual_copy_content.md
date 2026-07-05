# CMTFlow 20 分钟 PPT 中英对照可粘贴内容稿（按 outline 逐页意图重写）

对应文件：

- Outline：`D:\研二\KD4RL\实验result\cmtflow_20min_ppt_outline.md`
- 当前 PPT：`D:\研二\KD4RL\实验result\CTMFlow.pptx`

重要说明：

- 本稿以当前 PPT `CTMFlow.pptx` 的实际页序为准，并用论文 PDF 中的最终符号和实验结论校准。
- 当前 PPT 实际为 21 页：Slide 8-13 是方法部分，Slide 16 是 ablation 表格，Slide 17 是 random switch matched-count comparison，Slide 18 是 Controller switch case，Slide 19 是 Inner Actor case，Slide 20 是 Discussion & Conclusion，Slide 21 是 Q&A。
- 当前 PPT 中 `CTMFlow` 建议统一改为 `CMTFlow`；`Abalation` 建议改为 `Ablation`。
- Slide 12 中的 `\bar{\mathbf{w}}_t` 和 `\bar{\mathbf{w}}_t^{\mathrm{cand}}` 是 Controller 内部归一化后的比较特征，不是新的最终执行组合变量；讲述时需要和前面的 `\tilde{\mathbf{w}}_t`、`\tilde{\mathbf{b}}_t`、`\mathbf{w}_t^{\mathrm{cand}}`、`\mathbf{b}_t` 统一起来。

变量统一口径：

- 组合向量统一加粗：`\mathbf{w}_t`、`\tilde{\mathbf{w}}_t`、`\mathbf{b}_t`、`\tilde{\mathbf{b}}_t`、`\mathbf{w}_t^{\mathrm{cand}}`、`\mathbf{w}_t^{\mathrm{in}}`。
- 单个资产的分量不加粗：`w_{i,t}`、`b_{i,t}`、`w_{i,t}^{\mathrm{cand}}`。
- 市场输入、控制器辅助向量加粗：`\mathbf{X}_t^{\mathrm{out}}`、`\mathbf{X}_t^{\mathrm{in}}`、`\mathbf{X}_t^{\mathrm{ctrl}}`、`\mathbf{u}_t`、`\mathbf{a}_t^{\mathrm{ctrl}}`。
- 标量动作和概率不加粗：`g_t`、`e_t`、`\pi_t^{\mathrm{exit}}`、`\Delta w_{i,t}^{\mathrm{inner}}`。

---

## 每页扩充建议索引

这个索引用于让 PPT 页面更充分。每页可以从下面选择 2-4 条加入页面正文、图旁 callout、页脚 takeaway 或讲稿备注；不建议全部塞进同一页。

### Slide 1 扩充建议

- 页面可加一句主线：`Three questions: when to revise, what to hold, how to refine.`
- 页脚可加关键词：`Controller-guided revision / Segment-level base / Support-constrained refinement`
- 讲述补充：开场时先强调 CMTFlow 关注的是“组合是否应该继续持有”，而不是只做下一日收益预测。

### Slide 2 扩充建议

- 可加小框：`PM = long-horizon allocation under non-stationarity, risk, and cost.`
- 可加三角关系：`Return`、`Risk`、`Transaction cost` 三者共同决定策略质量。
- 讲述补充：引出“非平稳市场会让一个历史上有效的组合逐渐失效”，为 Controller 的必要性做铺垫。

### Slide 3 扩充建议

- 可加变量关系短句：`\tilde{\mathbf{w}}_t` 是被动漂移后的持仓，`\mathbf{w}_t` 是主动再平衡后的持仓。
- 可加成本解释：`\|\mathbf{w}_t-\tilde{\mathbf{w}}_t\|_1` 衡量 turnover，turnover 越大，交易成本越高。
- 页面 takeaway：`Portfolio management is a sequential wealth-control problem, not a one-step prediction task.`

### Slide 4 扩充建议

- 可加对比表：`Classical optimization` 强在可解释，弱在静态假设；`Daily RL` 强在端到端，弱在噪声敏感；`Hierarchical/adaptive RL` 强在灵活，弱在缺少 persistent base memory。
- 可加 research gap：`Existing methods rarely compare the active base with a candidate replacement every day.`
- 讲述补充：不要把 related work 讲成“别人都不好”，而是讲“已有方法分别解决一部分，但没有统一处理 when/what/how”。

### Slide 5 扩充建议

- 可加角色映射：`Portfolio manager -> what to hold`，`Trader -> how to refine`，`Risk controller -> when to revise`。
- 可加类比短句：`A real investment process separates planning, execution, and risk monitoring.`
- 讲述补充：这一页是直觉入口，用真实量化团队分工解释为什么模型也应该分层。

### Slide 6 扩充建议

- 可加三条 challenge title：`Noisy daily allocation`、`Rigid fixed holding`、`Missing real-time revision awareness`。
- 可加中心问题：`Can a policy learn both portfolio construction and revision timing?`
- 讲述补充：强调 fixed holding 的问题不是“低频一定不好”，而是固定日历无法匹配市场状态变化速度。

### Slide 7 扩充建议

- 可加模块对应关系：`Outer Actor = base construction`、`Controller = base revision`、`Inner Actor = daily refinement`。
- 图旁 callout：`The active base portfolio works like a memory state that can be held, drifted, compared, or replaced.`
- 讲述补充：先把三模块职责说清楚，后面公式才不会显得突然。

### Slide 8 扩充建议

- 可加 pipeline 短句：`\tilde{\mathbf{w}}_t -> \mathbf{w}_t^{\mathrm{cand}} -> g_t -> \mathbf{b}_t -> \mathbf{w}_t`
- 可加页面 takeaway：`The final portfolio is produced by coordination, not by one monolithic daily policy.`
- 讲述补充：这页适合用手指着图走一遍数据流，不要在这里展开训练目标。

### Slide 9 扩充建议

- 可加结构框：`long-horizon market window -> LSTM-HA -> CAAN -> MLP scores -> Top-K candidate base`
- 可加关键约束：`Output is a sparse long-only candidate base portfolio, not the final executed portfolio.`
- 讲述补充：Outer Actor 的价值是提供中期战略锚点，避免整个系统被短期价格噪声牵着走。

### Slide 10 扩充建议

- 可加公式注释：`\alpha` controls the strength of daily refinement.
- 可加约束短句：`Support constraint prevents the Inner Actor from overriding the base asset universe.`
- 讲述补充：Inner Actor 的定位是“在已选资产内部微调权重”，不是重新选股，也不是决定是否切仓。

### Slide 11 扩充建议

- 可加 hold/switch 视觉：`g_t=0: keep \tilde{\mathbf{b}}_t`，`g_t=1: replace with \mathbf{w}_t^{\mathrm{cand}}`。
- 可加机制解释：`30-day cap is a safety boundary; early revision is learned by the Controller.`
- 讲述补充：Controller 不直接输出最终组合，它输出的是事件决策，把一个 base 是否继续持有变成可学习问题。

### Slide 12 扩充建议

- 可加变量统一提示：`\bar{\mathbf{w}}_t` and `\bar{\mathbf{w}}_t^{\mathrm{cand}}` are normalized features, not new portfolio decisions.
- 可加状态四要素：`recent market`、`current holding quality`、`candidate properties`、`switching cost/overlap`。
- 页面 takeaway：`Exit probability is a nonlinear policy signal, not a direct return forecast.`

### Slide 13 扩充建议

- 可加训练逻辑图：`Outer warmup -> Inner warmup -> fixed-HRL stabilization -> Controller PG -> controller-active finetuning`
- 可加理由短句：`Different modules operate at different temporal resolutions and need staged stabilization.`
- 讲述补充：强调最后不是只训练 Controller，而是回到 daily controller-active protocol 下联合微调。

### Slide 14 扩充建议

- 可加实验协议框：`daily check / threshold 0.5 / no min-hold lock / 30-day max-hold cap`
- 可加指标说明：`TR measures terminal gain; MDD measures downside pain; CR measures return per drawdown.`
- 讲述补充：提醒听众后面结果要看 risk-return trade-off，不要只看 total return。

### Slide 15 扩充建议

- 可加 Nasdaq callout：`Highest matched TR = 265.53%, with MDD controlled at 18.62%.`
- 可加 CSI callout：`Slightly lower TR than DeepTrader, but higher Sharpe and much lower MDD.`
- 页面 takeaway：`CMTFlow improves return-path quality rather than simply chasing terminal wealth.`

### Slide 16 扩充建议

- 可加组件结论：`Controller brings the dominant gain; Inner Actor provides market-dependent refinement.`
- 可加 Nasdaq delta：`Outer-only -> Outer + Controller: MDD 32.09% -> 21.24%.`
- 可加 CSI delta：`Outer-only -> Outer + Controller: TR 147.05% -> 237.77%.`

### Slide 17 扩充建议

- 可加实验目的：`Matched-count random switching controls for switch frequency.`
- 可加图例解释：`Orange = learned timing; gray = random same-count switches; dashed = fixed HRL backbone.`
- 页面 takeaway：`The learned Controller is valuable because of when it switches, not merely how often it switches.`

### Slide 18 扩充建议

- 可加 Nasdaq case callout：`p(switch)=0.63, return gap +2.43 pp, MDD reduction +3.31 pp.`
- 可加 CSI case callout：`p(switch)=0.52, return gap +20.54 pp, MDD reduction +5.48 pp.`
- 讲述补充：强调这是 same-start, same-horizon frozen counterfactual，不混入后续真实路径的其他 switch。

### Slide 19 扩充建议

- 可加图读法：`Top: future relative return; middle: inner tilt; bottom: executed weights and alignment.`
- 可加定义提示：`\Delta w_{i,t}^{\mathrm{inner}}=w_{i,t}-b_{i,t}` measures local deviation from the active base.
- 页面 takeaway：`Inner Actor provides local allocation resonance, not an independent switching signal.`

### Slide 20 扩充建议

- 可加三条贡献压缩版：`Explicit decision decomposition`、`Trainable base-revision controller`、`Risk-return evidence with ablation and cases`。
- 可加 limitation：`Evaluation currently focuses on long-only equity markets and proportional transaction costs.`
- 讲述补充：结尾要把主线收回到 `learning when to revise`，这比简单复述实验数字更有记忆点。

### Slide 21 扩充建议

- 可加备答问题：`Why normalized controller features?`、`Why does Inner Actor not always improve TR?`、`Why random matched-count comparison?`
- 可加一句收尾：`The controller makes portfolio revision an adaptive event decision.`
- 讲述补充：如果有人问公式变量，先区分“组合向量”和“Controller 内部归一化特征”。

---

## Slide 1｜标题页

### 中文可贴内容

`CMTFlow：控制器引导基础组合修订与日度细化的层次化投资组合管理`

`汇报人：Wenxuan Tong`

`合作者：Zihan Feng, Junjie Wu`

`July 4th, 2026`

`核心观点：在投资组合管理中，学习“何时修订组合”与学习“持有什么组合”同样重要。`

### English Copy

`CMTFlow: Hierarchical Portfolio Management with Controller-Guided Base Revision and Daily Refinement`

`Presenter: Wenxuan Tong`

`Joint work with Zihan Feng, Junjie Wu`

`July 4th, 2026`

`Key message: In portfolio management, learning when to revise is as important as learning what to hold.`

---

## Slide 2｜研究背景：投资组合管理的基本目标

### 中文可贴内容

`非平稳市场中的投资组合管理`

`Portfolio Management (PM) 的目标是在不确定市场中跨资产配置资金，使长期收益最大化，同时控制风险、回撤和交易成本。`

`在量化交易中，大量研究希望从历史价格、成交量和资产关系中挖掘潜在规律，从而构造能够长期获利的投资组合。`

`然而，真实金融市场是非平稳的：收益分布会变化，资产相关性会变化，市场风格也会发生切换。`

`因此，一个过去表现良好的组合，可能在未来逐渐失效。`

`这意味着策略不仅要回答“当前买什么”，还要回答“当前组合是否还值得继续持有”。`

`核心问题：如何在收益、风险、成本和市场适应性之间取得平衡？`

### English Copy

`Portfolio Management in Non-Stationary Markets`

`Portfolio Management (PM) aims to allocate capital across assets under uncertainty, maximizing long-horizon returns while controlling risk, drawdown, and transaction cost.`

`In quantitative trading, many studies try to discover exploitable patterns from historical prices, volumes, and asset relationships, and then construct profitable portfolios.`

`However, real financial markets are non-stationary: return distributions shift, asset correlations evolve, and market regimes change over time.`

`As a result, a portfolio that works well in the past may gradually become ineffective in the future.`

`Therefore, a strategy should not only answer what to buy, but also whether the current portfolio is still worth holding.`

`Core question: how can we balance return, risk, cost, and market adaptability?`

---

## Slide 3｜问题定义：带漂移和交易成本的动态组合

### 中文可贴内容

`带漂移和交易成本的动态组合问题`

`给定 M 个可交易资产，策略在每个交易日 t 决定组合权重：`

`\mathbf{w}_t=(w_{1,t},...,w_{M,t})^\top,\quad \mathbf{1}^\top\mathbf{w}_t=1,\quad \mathbf{w}_t\ge0`

`资产从 t 到 t+1 的 close-to-close gross return 为：`

`\mathbf{y}_t=(c_{1,t+1}/c_{1,t},...,c_{M,t+1}/c_{M,t})^\top`

`即使不主动交易，资产价格变化也会导致昨日组合发生被动漂移：`

`\tilde{\mathbf{w}}_t=\frac{\mathbf{y}_{t-1}\odot\mathbf{w}_{t-1}}{\mathbf{y}_{t-1}^{\top}\mathbf{w}_{t-1}}`

`如果从漂移组合 \tilde{\mathbf{w}}_t 调整到新组合 \mathbf{w}_t，就会产生交易成本：`

`\mu_t=1-\rho\|\mathbf{w}_t-\tilde{\mathbf{w}}_t\|_1`

`组合财富演化为：`

`V_{t+1}=V_t\mu_t\mathbf{y}_t^\top\mathbf{w}_t`

`最终目标是在整个投资期内最大化终端财富：`

`\max_{\{\mathbf{w}_t\}_{t=0}^{T-1}}V_T`

`直观理解：组合管理不是单期优化，而是一个带状态漂移、交易成本和长期目标的序贯决策问题。`

### English Copy

`Dynamic Portfolio Problem with Drift and Transaction Cost`

`Given M tradable assets, the strategy determines the portfolio weight vector at each trading day t:`

`\mathbf{w}_t=(w_{1,t},...,w_{M,t})^\top,\quad \mathbf{1}^\top\mathbf{w}_t=1,\quad \mathbf{w}_t\ge0`

`The close-to-close gross return vector from day t to day t+1 is:`

`\mathbf{y}_t=(c_{1,t+1}/c_{1,t},...,c_{M,t+1}/c_{M,t})^\top`

`Even without active trading, yesterday's portfolio passively drifts due to price movements:`

`\tilde{\mathbf{w}}_t=\frac{\mathbf{y}_{t-1}\odot\mathbf{w}_{t-1}}{\mathbf{y}_{t-1}^{\top}\mathbf{w}_{t-1}}`

`Rebalancing from the drifted portfolio \tilde{\mathbf{w}}_t to the new portfolio \mathbf{w}_t induces transaction cost:`

`\mu_t=1-\rho\|\mathbf{w}_t-\tilde{\mathbf{w}}_t\|_1`

`The portfolio value evolves as:`

`V_{t+1}=V_t\mu_t\mathbf{y}_t^\top\mathbf{w}_t`

`The objective is to maximize terminal wealth over the full horizon:`

`\max_{\{\mathbf{w}_t\}_{t=0}^{T-1}}V_T`

`Intuition: portfolio management is not a single-period optimization problem, but a sequential decision problem with state drift, transaction cost, and long-term objective.`

---

## Slide 4｜相关工作：从静态优化到深度强化学习

### 中文可贴内容

`相关工作：不同方法如何求解组合权重 \mathbf{w}_t`

`Classical Portfolio Optimization`

`传统方法通常先估计预期收益 \boldsymbol{\mu}、风险协方差 \boldsymbol{\Sigma}，再在单期或静态假设下求解组合权重 \mathbf{w}_t。`

`典型公式：\max_{\mathbf{w}\in\Delta^M}\boldsymbol{\mu}^{\top}\mathbf{w}-\lambda\mathbf{w}^{\top}\boldsymbol{\Sigma}\mathbf{w}，其中 \Delta^M 表示权重和为 1 的可行组合空间。`

`代表方法：mean-variance optimization、CAPM、dynamic asset allocation 和 opportunistic rebalancing（Markowitz, 1952; Sharpe, 1964; Perold and Sharpe, 1988; Sullivan, 2008）。`

`局限：优化目标清晰、可解释性强，但依赖收益和协方差估计；再平衡时机通常由固定周期或阈值规则给出，难以适应快速变化的市场状态。`

`Deep RL Portfolio Management`

`深度强化学习方法将 PM 建模为序贯决策问题，直接学习从市场状态到组合权重 \mathbf{w}_t 的策略。`

`典型公式：\max_{\pi_\theta}\mathbb{E}[\sum_{t=0}^{T-1}\gamma^t r_t]，其中 \mathbf{w}_t=\pi_\theta(s_t)，s_t 是价格、技术指标和历史窗口等市场状态。`

`代表方法：PGPortfolio、FinRL、DeepTrader、AlphaStock 和 Investor-Imitator（Jiang et al., 2017; Liu et al., 2022; Wang et al., 2021; Wang et al., 2019; Ding et al., 2018）。`

`局限：很多方法把所有决策压缩成一个 daily allocation policy，每天直接输出完整组合权重，容易混淆中期组合构造和短期执行细化。`

`Hierarchical / Adaptive Portfolio Learning`

`层次化或自适应方法尝试提高策略灵活性，例如自适应再平衡间隔、层次化选股或多模块协同决策。`

`典型公式：\max_{\pi_H,\pi_L}\mathbb{E}[\sum_t\gamma^t r_t]，其中 z_t=\pi_H(s_t) 表示高层决策，\mathbf{w}_t=\pi_L(s_t,z_t) 表示低层组合输出。`

`代表方法：HADAPS、DeepAries 等自适应或层次化投资策略（Kim et al., 2023; Kim et al., 2025）。`

`局限：这些方法提高了策略表达能力，但通常没有显式维护 active base portfolio memory，也没有持续比较当前 active base \mathbf{b}_t 与候选 base \mathbf{w}_t^{\mathrm{cand}} 是否应该替换。`

`Research Gap`

`现有方法尚未统一处理三个问题：when to revise, what base to hold, and how to refine.`

`这也对应 CMTFlow 的核心设计：用 Controller 判断是否修订，用 Outer Actor 生成中期 base portfolio，用 Inner Actor 在 base 附近进行日度细化。`

### English Copy

`Related Work: How Existing Methods Solve Portfolio Weights \mathbf{w}_t`

`Classical Portfolio Optimization`

`Classical methods usually estimate expected return \boldsymbol{\mu} and covariance \boldsymbol{\Sigma}, and then solve portfolio weights \mathbf{w}_t under a static or single-period assumption.`

`Typical objective: \max_{\mathbf{w}\in\Delta^M}\boldsymbol{\mu}^{\top}\mathbf{w}-\lambda\mathbf{w}^{\top}\boldsymbol{\Sigma}\mathbf{w}, where \Delta^M denotes the feasible portfolio simplex.`

`Representative methods include mean-variance optimization, CAPM, dynamic asset allocation, and opportunistic rebalancing (Markowitz, 1952; Sharpe, 1964; Perold and Sharpe, 1988; Sullivan, 2008).`

`Limitations: the objective is clear and interpretable, but the method depends heavily on return and covariance estimation; revision timing is usually controlled by fixed calendar or threshold rules.`

`Deep RL Portfolio Management`

`Deep reinforcement learning methods formulate PM as a sequential decision problem and directly learn a policy from market states to portfolio weights \mathbf{w}_t.`

`Typical objective: \max_{\pi_\theta}\mathbb{E}[\sum_{t=0}^{T-1}\gamma^t r_t], where \mathbf{w}_t=\pi_\theta(s_t), and s_t contains prices, technical indicators, and historical windows.`

`Representative methods include PGPortfolio, FinRL, DeepTrader, AlphaStock, and Investor-Imitator (Jiang et al., 2017; Liu et al., 2022; Wang et al., 2021; Wang et al., 2019; Ding et al., 2018).`

`Limitations: many methods compress all decisions into one daily allocation policy, which directly outputs full portfolio weights and entangles medium-horizon base construction with short-horizon execution refinement.`

`Hierarchical / Adaptive Portfolio Learning`

`Hierarchical or adaptive methods improve policy flexibility through adaptive rebalancing intervals, hierarchical asset selection, or multi-module decision making.`

`Typical objective: \max_{\pi_H,\pi_L}\mathbb{E}[\sum_t\gamma^t r_t], where z_t=\pi_H(s_t) is a high-level decision and \mathbf{w}_t=\pi_L(s_t,z_t) is the lower-level portfolio output.`

`Representative methods include HADAPS and DeepAries for adaptive or hierarchical portfolio learning (Kim et al., 2023; Kim et al., 2025).`

`Limitations: they improve policy expressiveness, but usually do not explicitly maintain an active base portfolio memory, nor continuously compare the current active base \mathbf{b}_t with a candidate base \mathbf{w}_t^{\mathrm{cand}}.`

`Research Gap`

`Existing methods do not jointly address three questions: when to revise, what base to hold, and how to refine.`

`This motivates CMTFlow: the Controller decides whether to revise, the Outer Actor generates a medium-horizon base portfolio, and the Inner Actor performs daily refinement around the base.`

---

## Slide 5｜来源于生活的想法：真实量化团队是多尺度协同

### 中文可贴内容

`来源于实际投资团队的直观想法`

`在真实量化团队中，投资组合通常不是由一个单一策略每天重新决定所有资产权重，而是由多个角色在不同时间尺度上协同完成。`

`Portfolio Manager`

`先判断未来一段时间应该重点持有哪些资产，并给出一个中期基础配置框架。`

`对应 CMTFlow 中的 Outer Actor：回答 WHAT base to hold.`

`Trader`

`根据每天新到来的价格、成交量和短期波动，在基础组合附近调整实际执行权重。`

`对应 CMTFlow 中的 Inner Actor：回答 HOW to refine daily weights.`

`Risk Controller`

`持续观察当前组合的收益、回撤、风险暴露和候选替代组合，当当前组合存在未来风险时提醒或触发修订。`

`对应 CMTFlow 中的 Controller：回答 WHEN to revise.`

`核心直觉：组合管理天然是多尺度序贯决策，而不是单一的日度权重输出问题。`

### English Copy

`Motivation from Real Quantitative Investment Teams`

`In a real quantitative investment team, portfolio management is usually not handled by a single policy that re-decides all asset weights every day. Instead, multiple roles cooperate at different time scales.`

`Portfolio Manager`

`First determines which assets should be the core holdings over the next period and provides a medium-horizon base allocation framework.`

`Corresponding CMTFlow module: Outer Actor, which answers WHAT base to hold.`

`Trader`

`Adjusts actual execution weights around the base portfolio according to newly arriving prices, volumes, and short-term fluctuations.`

`Corresponding CMTFlow module: Inner Actor, which answers HOW to refine daily weights.`

`Risk Controller`

`Continuously monitors return, drawdown, risk exposure, and candidate replacement portfolios, and triggers revision when the current portfolio may become risky.`

`Corresponding CMTFlow module: Controller, which answers WHEN to revise.`

`Core intuition: portfolio management is naturally a multi-scale sequential decision problem, not a single daily weight-output problem.`

---

## Slide 6｜挑战：现有方法与实际分工之间的 Gap

### 中文可贴内容

`挑战：现有方法与实际多角色决策之间存在差距`

`Challenge 1: Daily policy is noisy`

`许多 RL 方法每天直接调整完整组合权重，相当于让一个策略同时完成 manager 和 trader 的工作。`

`问题：日度市场信号噪声很大，策略容易对短期波动过度反应，导致不稳定换手。`

`Challenge 2: Fixed holding is inflexible`

`有些方法预先设定固定持仓期，在持仓期内保持组合不变，或者只按照固定周期进行再平衡。`

`问题：市场规律的变化速度并不固定。有些阶段市场结构快速切换，需要更早修订；有些阶段趋势较稳定，频繁换仓反而会带来成本和噪声。固定持仓期缺乏灵活性，难以适应不同市场环境。`

`Challenge 3: Lack of real-time risk awareness`

`当前很多策略通常是在每次持仓开始前确定组合或持仓计划，而不是在持仓过程中实时监控 active portfolio 的状态再做决策。`

`问题：策略缺少对当前持仓收益退化、回撤扩大、风险暴露变化和候选组合优势的持续感知，因此难以及时判断当前 base 是否已经失效，以及是否应该切换到新的 candidate base。`

`本文目标：显式学习 whether and when to revise, what to hold, and how to refine.`

### English Copy

`Challenges: The Gap between Existing Methods and Practical Multi-Role Decisions`

`Challenge 1: Daily policy is noisy`

`Many RL methods directly adjust the full portfolio weights every day, which means one policy performs both the manager's and the trader's roles.`

`Problem: daily market signals are noisy, so the policy may overreact to short-term fluctuations and produce unstable turnover.`

`Challenge 2: Fixed holding is inflexible`

`Some methods predefine a fixed holding period, keep the portfolio unchanged within that period, or rebalance only at fixed calendar intervals.`

`Problem: market patterns do not evolve at a constant speed. Some regimes change quickly and require earlier revision, while stable regimes may not need frequent rebalancing. A fixed holding horizon is therefore too rigid to adapt to different market environments.`

`Challenge 3: Lack of real-time risk awareness`

`Many strategies determine the portfolio or holding plan before each holding period, rather than monitoring the active portfolio state in real time during the holding process.`

`Problem: they lack continuous awareness of return deterioration, drawdown expansion, risk exposure changes, and candidate portfolio advantages. As a result, they may fail to detect when the current base has become stale and when switching to a candidate base is necessary.`

`Goal: explicitly learn whether and when to revise, what to hold, and how to refine.`

---

## Slide 7｜方案总览：CMTFlow 的统一分层结构

### 中文可贴内容

`CMTFlow 总体框架`

`为了对应真实投资团队中的多角色分工，CMTFlow 构建了三个协同强化学习模块。`

`Controller`

`比较漂移后的当前持仓和 Outer Actor 生成的候选基础组合，学习当前 base 是否需要被修订。`

`对应问题：WHEN to revise?`

`Outer Actor`

`根据长期市场窗口生成稀疏的 segment-level candidate base portfolio。`

`对应问题：WHAT base to hold?`

`Inner Actor`

`在 active base portfolio 的资产支撑集内部进行日度局部权重细化。`

`对应问题：HOW to refine?`

`整体流程：Controller 决定保留还是切换 base，Outer 提供候选 base，Inner 将 active base 转化为最终执行组合。`

### English Copy

`Overview of CMTFlow`

`To match the multi-role structure in real investment teams, CMTFlow builds three coordinated reinforcement learning modules.`

`Controller`

`Compares the drifted current holding with the candidate base generated by the Outer Actor, and learns whether the current base should be revised.`

`Question answered: WHEN to revise?`

`Outer Actor`

`Generates a sparse segment-level candidate base portfolio from long-horizon market windows.`

`Question answered: WHAT base to hold?`

`Inner Actor`

`Performs local daily weight refinement inside the asset support of the active base portfolio.`

`Question answered: HOW to refine?`

`Overall flow: the Controller decides whether to keep or switch the base, the Outer Actor provides candidate bases, and the Inner Actor converts the active base into the final executed portfolio.`

---

## Slide 8｜方法总览：CMTFlow 的整体架构

### 中文可贴内容

`The architecture of proposed framework: CMTFlow`

`这页先给出完整架构，不急着讲公式。核心是把投资组合决策拆成三个相互协作的模块：Controller、Outer Actor 和 Inner Actor。`

`Outer Actor 负责生成 candidate base portfolio，回答 WHAT base to hold。`

`Controller 比较当前持仓状态和 candidate base，回答 WHETHER / WHEN to revise。`

`Inner Actor 在 active base 的支撑集内做日度局部细化，回答 HOW to refine。`

`变量主线`

`\tilde{\mathbf{w}}_t: 价格漂移后的当前执行持仓 exposure。`

`\mathbf{w}_t^{\mathrm{cand}}: Outer Actor 生成的候选基础组合。`

`\tilde{\mathbf{b}}_t: 漂移后的 active base portfolio。`

`g_t: Controller 的 hold/switch 动作。`

`\mathbf{b}_t: Controller 决定后的 active base portfolio。`

`\mathbf{w}_t^{\mathrm{in}}: Inner Actor 生成的支撑集约束细化组合。`

`\mathbf{w}_t: 最终执行组合。`

`一句话：CMTFlow 不是每天直接重算一个完整组合，而是先维护一个中期 base，再学习何时替换它，并在其内部做日度细化。`

### English Copy

`The architecture of the proposed framework: CMTFlow`

`This slide gives the full architecture before going into equations. The key idea is to decompose portfolio decisions into three coordinated modules: Controller, Outer Actor, and Inner Actor.`

`The Outer Actor generates a candidate base portfolio and answers WHAT base to hold.`

`The Controller compares the current holding state with the candidate base and answers WHETHER / WHEN to revise.`

`The Inner Actor performs daily local refinement within the support of the active base and answers HOW to refine.`

`Notation thread`

`\tilde{\mathbf{w}}_t: drifted current executed holding exposure after price movement.`

`\mathbf{w}_t^{\mathrm{cand}}: candidate base portfolio generated by the Outer Actor.`

`\tilde{\mathbf{b}}_t: drifted active base portfolio.`

`g_t: hold/switch action produced by the Controller.`

`\mathbf{b}_t: active base portfolio after the Controller decision.`

`\mathbf{w}_t^{\mathrm{in}}: support-constrained refinement portfolio produced by the Inner Actor.`

`\mathbf{w}_t: final executed portfolio.`

`In one sentence: CMTFlow does not directly recompute a full portfolio every day; it maintains a medium-horizon base, learns when to replace it, and refines it locally for daily execution.`

---

## Slide 9｜方法细节：Outer Actor 生成中期基础组合

### 中文可贴内容

`Outer Actor: Medium-Horizon Base Portfolio Construction`

`角色：Outer Actor 不是最终每日交易器，而是中期 base generator。它回答：下一段应该持有什么基础组合？`

`State`

`s_t^{\mathrm{out}}=\{\mathbf{X}_t^{\mathrm{out}},\tilde{\mathbf{w}}_t\}`

`\mathbf{X}_t^{\mathrm{out}}: long-horizon market window，用来捕捉中期趋势和跨资产关系。`

`\tilde{\mathbf{w}}_t: 当前漂移后的执行持仓，为 Outer Actor 提供当前组合背景。`

`Network`

`LSTM-HA: long-horizon asset-wise temporal modeling。`

`CAAN: cross-asset dependency modeling。`

`MLP head: asset-wise score generation。`

`Action / Output`

`网络输出资产得分 \mathbf{q}_t^{\mathrm{out}}，保留 Top-K 资产并归一化，得到 candidate base portfolio \mathbf{w}_t^{\mathrm{cand}}。`

`注意：这里推荐统一用 \mathbf{w}_t^{\mathrm{cand}}，不要再把输出写成 \mathbf{w}_t^{\mathrm{out}}，因为后面 Controller 比较的候选组合就是 \mathbf{w}_t^{\mathrm{cand}}。`

`直观作用：Outer Actor 提供稳定的中期战略锚点，避免最终组合完全被日度噪声牵引。`

### English Copy

`Outer Actor: Medium-Horizon Base Portfolio Construction`

`Role: the Outer Actor is not the final daily trader. It is a medium-horizon base generator that answers what base portfolio should anchor the next segment.`

`State`

`s_t^{\mathrm{out}}=\{\mathbf{X}_t^{\mathrm{out}},\tilde{\mathbf{w}}_t\}`

`\mathbf{X}_t^{\mathrm{out}}: long-horizon market window for capturing medium-term trends and cross-asset relationships.`

`\tilde{\mathbf{w}}_t: current drifted executed holding, which provides portfolio context to the Outer Actor.`

`Network`

`LSTM-HA: long-horizon asset-wise temporal modeling.`

`CAAN: cross-asset dependency modeling.`

`MLP head: asset-wise score generation.`

`Action / Output`

`The network outputs asset scores \mathbf{q}_t^{\mathrm{out}}, keeps the Top-K assets, and normalizes them into the candidate base portfolio \mathbf{w}_t^{\mathrm{cand}}.`

`For notation consistency, it is better to call the output \mathbf{w}_t^{\mathrm{cand}}, because this is exactly the candidate compared by the Controller later.`

`Intuition: the Outer Actor provides a stable medium-horizon anchor, preventing the final portfolio from being fully driven by daily noise.`

---

## Slide 10｜方法细节：Inner Actor 做支撑集约束的日度细化

### 中文可贴内容

`Inner Actor: Support-Constrained Daily Refinement`

`角色：Inner Actor 不重新选股票池，也不负责切仓。它只在当前 active base portfolio 内调整权重，回答 HOW to refine。`

`State`

`s_t^{\mathrm{in}}=\{\mathbf{X}_t^{\mathrm{in}},\tilde{\mathbf{w}}_t,\mathbf{b}_t\}`

`\mathbf{X}_t^{\mathrm{in}}: short-horizon market window，用来捕捉日度局部信号。`

`\tilde{\mathbf{w}}_t: 当前漂移后的执行持仓。`

`\mathbf{b}_t: Controller 决定后的 active base portfolio。`

`Support constraint`

`m_{i,t}=\mathbb{I}(b_{i,t}>0)`

`Inner Actor 只能在 \mathbf{b}_t 选中的资产内部重新分配权重，不能任意引入 base 外的新资产。`

`Executed portfolio`

`\mathbf{w}_t=(1-\alpha)\mathbf{b}_t+\alpha\mathbf{w}_t^{\mathrm{in}}`

`解释：最终执行组合仍然锚定在 active base 上，只允许 Inner Actor 做局部 tilt。`

`直观作用：Inner Actor 提供日频灵活性，但不会推翻 Outer/Controller 给出的中期结构。`

### English Copy

`Inner Actor: Support-Constrained Daily Refinement`

`Role: the Inner Actor does not re-select the asset universe, and it is not the switching controller. It only adjusts weights inside the current active base portfolio and answers how to refine.`

`State`

`s_t^{\mathrm{in}}=\{\mathbf{X}_t^{\mathrm{in}},\tilde{\mathbf{w}}_t,\mathbf{b}_t\}`

`\mathbf{X}_t^{\mathrm{in}}: short-horizon market window for capturing local daily signals.`

`\tilde{\mathbf{w}}_t: current drifted executed holding.`

`\mathbf{b}_t: active base portfolio after the Controller decision.`

`Support constraint`

`m_{i,t}=\mathbb{I}(b_{i,t}>0)`

`The Inner Actor can only redistribute weights among assets selected by \mathbf{b}_t and cannot introduce arbitrary new assets outside the base support.`

`Executed portfolio`

`\mathbf{w}_t=(1-\alpha)\mathbf{b}_t+\alpha\mathbf{w}_t^{\mathrm{in}}`

`The final executed portfolio is still anchored to the active base, with the Inner Actor applying only local tilts.`

`Intuition: the Inner Actor provides daily flexibility without overriding the medium-horizon structure determined by the Outer Actor and Controller.`

---

## Slide 11｜方法细节：Controller 的事件驱动基础组合修订

### 中文可贴内容

`Controller: Event-Driven Base Portfolio Revision`

`角色：Controller 不直接分配最终资金，而是判断当前 active base 是否应该继续持有。它回答 WHETHER / WHEN to revise。`

`Decision objects`

`\tilde{\mathbf{b}}_t=\mathcal{D}(\mathbf{b}_{t-1},\mathbf{y}_{t-1}): 漂移后的 active base。`

`\mathbf{w}_t^{\mathrm{cand}}: Outer Actor 在当前时刻给出的 candidate base。`

`Action`

`g_t\in\{0,1\}`

`g_t=0: hold the drifted active base。`

`g_t=1: switch to the candidate base。`

`Base update`

`\mathbf{b}_t=g_t\mathbf{w}_t^{\mathrm{cand}}+(1-g_t)\tilde{\mathbf{b}}_t`

`Daily update`

`Controller 每个交易日检查一次。若 \pi_t^{\mathrm{exit}}\ge0.5，则切换到 candidate base；否则继续使用当前 base。`

`如果持仓年龄达到 H_{\mathrm{ref}}=30 天，则触发 max-hold cap 强制修订。`

`Training objective`

`Controller 不使用外部 switch labels，而是通过 controlled path 与 fixed-segment reference path 的反事实投资结果学习。`

`直观作用：Controller 将固定日历再平衡变成可学习的 hold/switch event policy。`

### English Copy

`Controller: Event-Driven Base Portfolio Revision`

`Role: the Controller does not directly allocate final capital. It decides whether the current active base should continue to be held and answers whether / when to revise.`

`Decision objects`

`\tilde{\mathbf{b}}_t=\mathcal{D}(\mathbf{b}_{t-1},\mathbf{y}_{t-1}): drifted active base.`

`\mathbf{w}_t^{\mathrm{cand}}: candidate base proposed by the Outer Actor at the current day.`

`Action`

`g_t\in\{0,1\}`

`g_t=0: hold the drifted active base.`

`g_t=1: switch to the candidate base.`

`Base update`

`\mathbf{b}_t=g_t\mathbf{w}_t^{\mathrm{cand}}+(1-g_t)\tilde{\mathbf{b}}_t`

`Daily update`

`The Controller is checked every trading day. If \pi_t^{\mathrm{exit}}\ge0.5, it switches to the candidate base; otherwise it keeps the current base.`

`If the holding age reaches H_{\mathrm{ref}}=30 days, the maximum-hold cap forces a revision.`

`Training objective`

`The Controller does not use external switch labels. It learns from counterfactual investment consequences between the controlled path and the fixed-segment reference path.`

`Intuition: the Controller turns calendar-based rebalancing into a learnable hold/switch event policy.`

---

## Slide 12｜方法细节：Controller 状态与退出概率

### 中文可贴内容

`Controller State and Exit Probability`

`这页最重要的是把变量讲统一：前面真正用于组合更新的是 \tilde{\mathbf{b}}_t、\mathbf{w}_t^{\mathrm{cand}} 和 \mathbf{b}_t；这里的 \bar{\mathbf{w}}_t 与 \bar{\mathbf{w}}_t^{\mathrm{cand}} 是 Controller 内部用于比较的归一化特征。`

`Notation alignment`

`\tilde{\mathbf{w}}_t: price drift 后的当前执行持仓 exposure。`

`\mathbf{w}_t^{\mathrm{cand}}: Outer Actor 生成的 candidate base。`

`\bar{\mathbf{w}}_t=\mathrm{norm}(\tilde{\mathbf{w}}_t): Controller 内部使用的 normalized current holding feature。`

`\bar{\mathbf{w}}_t^{\mathrm{cand}}=\mathrm{norm}(\mathbf{w}_t^{\mathrm{cand}}): Controller 内部使用的 normalized candidate feature。`

`因此，bar 不是新的 portfolio decision，只是为了让 Controller 可比较当前持仓和候选组合。`

`State`

`s_t^{\mathrm{ctrl}}=\{\mathbf{X}_t^{\mathrm{ctrl}},\bar{\mathbf{w}}_t,\bar{\mathbf{w}}_t^{\mathrm{cand}},\mathbf{u}_t,\mathbf{a}_t^{\mathrm{ctrl}}\}`

`\mathbf{X}_t^{\mathrm{ctrl}}: recent market tensor。`

`\mathbf{u}_t: holding-state vector，包括 holding age、segment return、segment drawdown、current concentration。`

`\mathbf{a}_t^{\mathrm{ctrl}}: action-comparison features，包括 turnover、candidate concentration、support overlap。`

`Auxiliary estimates`

`holding return, holding risk, switch advantage。`

`Exit probability`

`e_t=e_t^{\mathrm{base}}+\eta\tanh(\hat{A}_t^{\mathrm{sw}}/c_A)`

`\pi_t^{\mathrm{exit}}=\sigma(e_t)`

`解释：exit probability 是非线性的 policy signal，不应该被理解成单一的未来收益预测因子。`

### English Copy

`Controller State and Exit Probability`

`The key point of this slide is notation alignment. The portfolio update uses \tilde{\mathbf{b}}_t, \mathbf{w}_t^{\mathrm{cand}}, and \mathbf{b}_t, while \bar{\mathbf{w}}_t and \bar{\mathbf{w}}_t^{\mathrm{cand}} are normalized internal features used by the Controller for comparison.`

`Notation alignment`

`\tilde{\mathbf{w}}_t: current executed holding exposure after price drift.`

`\mathbf{w}_t^{\mathrm{cand}}: candidate base generated by the Outer Actor.`

`\bar{\mathbf{w}}_t=\mathrm{norm}(\tilde{\mathbf{w}}_t): normalized current holding feature used inside the Controller.`

`\bar{\mathbf{w}}_t^{\mathrm{cand}}=\mathrm{norm}(\mathbf{w}_t^{\mathrm{cand}}): normalized candidate feature used inside the Controller.`

`Therefore, the bar notation does not introduce a new portfolio decision. It only makes the current holding and candidate portfolio comparable for the Controller.`

`State`

`s_t^{\mathrm{ctrl}}=\{\mathbf{X}_t^{\mathrm{ctrl}},\bar{\mathbf{w}}_t,\bar{\mathbf{w}}_t^{\mathrm{cand}},\mathbf{u}_t,\mathbf{a}_t^{\mathrm{ctrl}}\}`

`\mathbf{X}_t^{\mathrm{ctrl}}: recent market tensor.`

`\mathbf{u}_t: holding-state vector, including holding age, segment return, segment drawdown, and current concentration.`

`\mathbf{a}_t^{\mathrm{ctrl}}: action-comparison features, including turnover, candidate concentration, and support overlap.`

`Auxiliary estimates`

`holding return, holding risk, switch advantage.`

`Exit probability`

`e_t=e_t^{\mathrm{base}}+\eta\tanh(\hat{A}_t^{\mathrm{sw}}/c_A)`

`\pi_t^{\mathrm{exit}}=\sigma(e_t)`

`Interpretation: the exit probability is a nonlinear policy signal, not a one-dimensional future-return predictor.`

---

## Slide 13｜模型训练：分阶段训练与伪代码

### 中文可贴内容

`Pseudocode: End-to-End Staged Training of CMTFlow`

`为什么分阶段训练？三个模块的时间尺度和 reward signal 不同，如果从零开始完全联合训练，优化会不稳定。`

`Stage I: Outer base-portfolio warmup`

`固定 30-day reference segments，单独训练 Outer Actor，学习稳定的 segment-level candidate base allocator。`

`Stage II: Inner daily refinement warmup`

`使用稳定 Outer checkpoint 生成 active base sequence，训练 Inner Actor 围绕 base 做日度 refinement。`

`Stage III: Fixed-HRL joint stabilization`

`Controller inactive；Outer + Inner 在固定 30-day reference segments 下联合稳定。`

`Stage IV: Controller policy-gradient optimization`

`冻结或固定 HRL backbone，通过 controlled path vs fixed reference path 的反事实 rollout 训练 Controller。`

`Stage V: Controller-active end-to-end finetuning`

`加载最好的 Controller checkpoint，让 Controller、Outer Actor 和 Inner Actor 在 daily controller-active protocol 下共同微调。`

`结论：最终模型的训练逻辑和测试逻辑一致，都是 daily controller-active decision protocol。`

### English Copy

`Pseudocode: End-to-End Staged Training of CMTFlow`

`Why staged training? The three modules operate at different temporal scales and receive different reward signals, so fully joint training from scratch would be unstable.`

`Stage I: Outer base-portfolio warmup`

`Train the Outer Actor alone under fixed 30-day reference segments to learn a stable segment-level candidate-base allocator.`

`Stage II: Inner daily refinement warmup`

`Use the stable Outer checkpoint to generate active base sequences and train the Inner Actor to refine daily weights around the base.`

`Stage III: Fixed-HRL joint stabilization`

`Keep the Controller inactive and jointly stabilize Outer + Inner under fixed 30-day reference segments.`

`Stage IV: Controller policy-gradient optimization`

`Train the Controller using counterfactual rollouts between the controlled path and the fixed reference path.`

`Stage V: Controller-active end-to-end finetuning`

`Load the best Controller checkpoint and jointly finetune the Controller, Outer Actor, and Inner Actor under the daily controller-active protocol.`

`Conclusion: the final training logic is aligned with the evaluation logic: both use the daily controller-active decision protocol.`

---

## Slide 14｜实验设置：数据集与指标

### 中文可贴内容

`Experimental Setup`

`Datasets`

`Nasdaq-100: large-cap U.S. technology-oriented equity market。`

`CSI-300: China A-share large-cap equity market。`

`Data split`

`Train: 2000-04 到 2017-12。`

`Validation: 2018-01 到 2019-12。`

`Test: 2020-01 到 2025-09 左右，具体以表格为准。`

`Evaluation protocol`

`Controller is checked daily。`

`No minimum-hold lock。`

`30-day maximum-hold cap。`

`Evaluation threshold: \pi_t^{\mathrm{exit}}\ge0.5。`

`Evaluated metrics`

`TR: total return, higher is better。`

`Sharpe: risk-adjusted return, higher is better。`

`MDD: maximum drawdown, lower is better。`

`CR: annualized return divided by MDD, higher is better。`

`讲述重点：后面所有结果都不要只看 TR，要同时看 Sharpe、MDD 和 CR。`

### English Copy

`Experimental Setup`

`Datasets`

`Nasdaq-100: large-cap U.S. technology-oriented equity market.`

`CSI-300: China A-share large-cap equity market.`

`Data split`

`Train: 2000-04 to 2017-12.`

`Validation: 2018-01 to 2019-12.`

`Test: roughly 2020-01 to 2025-09; use the exact table dates when needed.`

`Evaluation protocol`

`The Controller is checked daily.`

`No minimum-hold lock is imposed.`

`A 30-day maximum-hold cap is used.`

`Evaluation threshold: \pi_t^{\mathrm{exit}}\ge0.5.`

`Evaluated metrics`

`TR: total return, higher is better.`

`Sharpe: risk-adjusted return, higher is better.`

`MDD: maximum drawdown, lower is better.`

`CR: annualized return divided by MDD, higher is better.`

`Speaking point: do not judge the method only by TR; read Sharpe, MDD, and CR together.`

---

## Slide 15｜数值结果：主实验性能对比

### 中文可贴内容

`Performance Comparison: Baseline Comparison`

`这页主要说明：CMTFlow 不是每个单项指标都第一，但整体上兼顾了收益和风险。`

`Baselines`

`Traditional / Online: Buy&Hold, Markowitz, UCRP, Anticor, OLMAR, WMAMR。`

`Deep RL: AlphaStock, DeepAries, DeepTrader。`

`Nasdaq-100`

`CMTFlow 获得 matched methods 中最高累计收益：TR = 265.53%。`

`同时 MDD = 18.62%，显著低于高收益传统基线 WMAMR 的 33.88% 和 Anticor 的 44.59%。`

`DeepAries 的 Sharpe、MDD 和 CR 更好，但 TR 只有 162.96%，明显低于 CMTFlow。`

`CSI-300`

`CMTFlow 的 TR = 204.99%，略低于 DeepTrader 的 212.81%。`

`但 CMTFlow 将 Sharpe 从 1.03 提升到 1.14，并将 MDD 从 31.86% 降到 22.78%。`

`CR = 1.09，仅次于 DeepAries。`

`结论：CMTFlow 的优势不是单纯最大化终端财富，而是改善收益路径质量和风险收益权衡。`

### English Copy

`Performance Comparison: Baseline Comparison`

`This slide shows that CMTFlow is not the best on every single metric, but it achieves a strong overall balance between return and risk.`

`Baselines`

`Traditional / Online: Buy&Hold, Markowitz, UCRP, Anticor, OLMAR, WMAMR.`

`Deep RL: AlphaStock, DeepAries, DeepTrader.`

`Nasdaq-100`

`CMTFlow obtains the highest matched total return: TR = 265.53%.`

`Meanwhile, its MDD is 18.62%, much lower than high-return traditional baselines such as WMAMR at 33.88% and Anticor at 44.59%.`

`DeepAries has better Sharpe, MDD, and CR, but its TR is only 162.96%, substantially lower than CMTFlow.`

`CSI-300`

`CMTFlow obtains TR = 204.99%, slightly below DeepTrader's 212.81%.`

`However, CMTFlow improves Sharpe from 1.03 to 1.14 and reduces MDD from 31.86% to 22.78%.`

`Its CR is 1.09, second only to DeepAries.`

`Conclusion: CMTFlow does not simply maximize terminal wealth; it improves return-path quality and the risk-return trade-off.`

---

## Slide 16｜数值结果：Ablation 表格与模块贡献

### 中文可贴内容

`Ablation Study`

`这页的表格同时包含两类对比：组件消融和固定窗口 controller。核心问题是：收益来自哪个模块，是否只是固定频率换仓造成的？`

`Component ablation`

`Nasdaq-100: Outer-only TR = 220.42%, MDD = 32.09%, CR = 0.74。`

`Nasdaq-100: Outer + Controller TR = 237.50%, MDD = 21.24%, CR = 1.18。`

`Nasdaq-100: full CMTFlow TR = 265.53%, MDD = 18.62%, CR = 1.42。`

`解释：加入 Controller 后，收益提高、回撤明显下降，完整模型进一步提升。`

`CSI-300: Outer-only TR = 147.05%, Sharpe = 0.94, CR = 0.99。`

`CSI-300: Outer + Controller TR = 237.77%, Sharpe = 1.22, CR = 1.16。`

`CSI-300: full CMTFlow TR = 204.99%, Sharpe = 1.14, MDD = 22.78%, CR = 1.09。`

`解释：CSI-300 中 Controller 是主要收益改善来源；Inner Actor 的收益贡献更市场依赖，但 full model 的 MDD 略低于 Outer + Controller。`

`Fixed-window rows`

`固定 5/10/20/30/60 天切仓无法稳定复现 CMTFlow 的收益风险权衡。`

`结论：Controller 是主要动态修订机制；Inner Actor 更适合理解为局部权重细化，而不是独立的 universal alpha source。`

### English Copy

`Ablation Study`

`This table contains two comparisons: component ablations and fixed-window controllers. The key question is which module contributes the gains and whether the improvement is merely caused by fixed-frequency rebalancing.`

`Component ablation`

`Nasdaq-100: Outer-only TR = 220.42%, MDD = 32.09%, CR = 0.74.`

`Nasdaq-100: Outer + Controller TR = 237.50%, MDD = 21.24%, CR = 1.18.`

`Nasdaq-100: full CMTFlow TR = 265.53%, MDD = 18.62%, CR = 1.42.`

`Interpretation: adding the Controller improves return and clearly reduces drawdown; the full model further improves the result.`

`CSI-300: Outer-only TR = 147.05%, Sharpe = 0.94, CR = 0.99.`

`CSI-300: Outer + Controller TR = 237.77%, Sharpe = 1.22, CR = 1.16.`

`CSI-300: full CMTFlow TR = 204.99%, Sharpe = 1.14, MDD = 22.78%, CR = 1.09.`

`Interpretation: in CSI-300, the Controller is the main source of return improvement; the Inner Actor's return contribution is market-dependent, while the full model has slightly lower MDD than Outer + Controller.`

`Fixed-window rows`

`Fixed 5/10/20/30/60-day switching cannot reliably reproduce the risk-return trade-off of CMTFlow.`

`Conclusion: the Controller is the dominant dynamic revision mechanism; the Inner Actor is better understood as local weight refinement rather than a standalone universal alpha source.`

---

## Slide 17｜数值结果：Random Switch Matched-Count Comparison

### 中文可贴内容

`Ablation Analysis: Random Switch Matched-Count Comparison`

`这页不是固定窗口实验，而是 random switch matched-count comparison。它回答的问题是：CMTFlow 是否只是因为 switch 次数更多才表现好？`

`图中灰色路径`

`随机 switch 策略，控制 switch count 与 learned controller 匹配，用来构造同等换仓次数下的随机时机基线。`

`图中虚线`

`Fixed HRL，即没有 learned Controller 的固定分段 backbone。`

`图中橙色路径`

`Full controller，即 CMTFlow 学到的状态依赖 switch timing。`

`Nasdaq-100`

`Full controller 在最终路径上明显高于 Fixed HRL，并且不是随机同次数 switch 可以稳定复现的结果。`

`CSI-300 / SH market`

`Full controller 同样显著高于 Fixed HRL，说明 Controller 的价值主要来自学到何时 switch，而不是单纯增加换仓。`

`讲述重点：这页不要讲成固定 5/10/20/30/60 天窗口；PPT 图的目标是控制 switch count 后验证 learned timing 的价值。`

`结论：Controller 的优势来自 state-dependent timing，而不是 random timing 或 switch count 本身。`

### English Copy

`Ablation Analysis: Random Switch Matched-Count Comparison`

`This slide is not the fixed-window experiment. It is a random switch matched-count comparison, asking whether CMTFlow works simply because it switches more often.`

`Gray trajectories`

`Random switch policies with switch counts matched to the learned controller, providing random-timing baselines under comparable switching frequency.`

`Dashed line`

`Fixed HRL, the fixed-segment backbone without the learned Controller.`

`Orange line`

`Full controller, the state-dependent switch timing learned by CMTFlow.`

`Nasdaq-100`

`The full controller clearly improves over Fixed HRL, and its realized path is not something that random same-count switching can reliably reproduce.`

`CSI-300 / SH market`

`The full controller also stays well above Fixed HRL, indicating that the Controller's value mainly comes from learning when to switch rather than merely increasing the number of switches.`

`Speaking point: do not explain this slide as the fixed 5/10/20/30/60-day window comparison. The goal of this PPT figure is to control switch count and validate learned timing.`

`Conclusion: the Controller's advantage comes from state-dependent timing, not from random timing or switch count alone.`

---

## Slide 18｜案例研究：Controller 的可解释 switch 行为

### 中文可贴内容

`Case Study: Controller Switch Cases`

`这页用两个 30-trading-day frozen counterfactual case 解释 Controller 的经济含义：同一个 switch day，从同一个起点比较继续旧 base 和切到新 base。`

`Nasdaq-100 case`

`Key decision: 2021/04/19。`

`Switch probability: 0.63，threshold = 0.50。`

`30-day future return: keep -1.91% → switch +0.52%。`

`Return gap: +2.43 pp。`

`Future MDD: keep 8.82% → switch 5.51%。`

`30-day MDD reduction: +3.31 pp。`

`解释：如果继续旧 base，后续收益更弱且回撤更深；Controller 在持仓状态恶化前选择退出。`

`CSI-300 case`

`Key decision: 2021/07/07。`

`Switch probability: 0.52，threshold = 0.50。`

`30-day future return: keep -10.29% → switch +10.25%。`

`Return gap: +20.54 pp。`

`Future MDD: keep 13.71% → switch 8.23%。`

`30-day MDD reduction: +5.48 pp。`

`解释：Controller-selected new base 避免了旧 base continuation 的后续收益损失和更大回撤。`

`结论：Controller 的可解释性来自同起点、同 horizon 的反事实比较，不是事后看完整真实路径。`

### English Copy

`Case Study: Controller Switch Cases`

`This slide explains the economic meaning of the Controller using two 30-trading-day frozen counterfactual cases: from the same switch day, compare keeping the old base with switching to the new base.`

`Nasdaq-100 case`

`Key decision: 2021/04/19.`

`Switch probability: 0.63, threshold = 0.50.`

`30-day future return: keep -1.91% → switch +0.52%.`

`Return gap: +2.43 pp.`

`Future MDD: keep 8.82% → switch 5.51%.`

`30-day MDD reduction: +3.31 pp.`

`Interpretation: if the old base were kept, the subsequent path would be weaker and suffer deeper drawdown; the Controller exits a deteriorating holding state in advance.`

`CSI-300 case`

`Key decision: 2021/07/07.`

`Switch probability: 0.52, threshold = 0.50.`

`30-day future return: keep -10.29% → switch +10.25%.`

`Return gap: +20.54 pp.`

`Future MDD: keep 13.71% → switch 8.23%.`

`30-day MDD reduction: +5.48 pp.`

`Interpretation: the controller-selected new base avoids the subsequent return loss and larger drawdown of the old-base continuation.`

`Conclusion: the interpretability of the Controller comes from same-start, same-horizon counterfactual comparison, not from retrospectively reading the full realized path.`

---

## Slide 19｜案例研究：Inner Actor 的局部调权行为

### 中文可贴内容

`Case Study: Inner Actor Base Adjustment vs Future Return`

`这页说明 Inner Actor 如何在 active base 内做局部 tilt，而不是决定何时切仓。`

`解释变量`

`\Delta w_{i,t}^{\mathrm{inner}}=w_{i,t}-b_{i,t}`

`其中 b_{i,t} 是 active base weight，w_{i,t} 是最终 executed weight；这里是向量 \mathbf{b}_t 和 \mathbf{w}_t 的第 i 个分量，所以分量符号不加粗。`

`Positive tilt: Inner Actor 相对 base 增加该资产权重。`

`Negative tilt: Inner Actor 相对 base 降低该资产权重。`

`图的三层含义`

`第一行：future 5-day relative return，绿色表示未来相对更强。`

`第二行：inner tilt，青色表示 overweight，棕色表示 underweight。`

`第三行：inner adjustment 后的 executed portfolio weights。`

`底部：把 tilt-return alignment 聚合到资产层面，正条表示调权方向与未来相对收益方向一致。`

`Nasdaq-100 selected window`

`Mean corr(tilt, future relative return) = 0.46。`

`Positive alignment days = 73%。`

`CSI-300 selected window`

`Mean corr = 0.33。`

`Positive alignment days = 70%。`

`谨慎结论：Inner Actor 在代表性窗口中体现出局部调权与未来相对收益的共振，但它不是主要 switch 模块，也不应被说成稳定独立 alpha source。`

### English Copy

`Case Study: Inner Actor Base Adjustment vs Future Return`

`This slide explains how the Inner Actor applies local tilts inside the active base, rather than deciding when to switch.`

`Explanation variable`

`\Delta w_{i,t}^{\mathrm{inner}}=w_{i,t}-b_{i,t}`

`Here b_{i,t} is the active base weight and w_{i,t} is the final executed weight; they are scalar components of the vectors \mathbf{b}_t and \mathbf{w}_t, so the component symbols are not bolded.`

`Positive tilt: the Inner Actor increases the asset weight relative to the base.`

`Negative tilt: the Inner Actor decreases the asset weight relative to the base.`

`How to read the figure`

`Top row: future 5-day relative return; green means future relative winner.`

`Second row: inner tilt; teal means overweight and brown means underweight.`

`Third row: executed portfolio weights after inner adjustment.`

`Bottom panel: asset-level tilt-return alignment; positive bars mean the tilt direction agrees with future relative return.`

`Nasdaq-100 selected window`

`Mean corr(tilt, future relative return) = 0.46.`

`Positive alignment days = 73%.`

`CSI-300 selected window`

`Mean corr = 0.33.`

`Positive alignment days = 70%.`

`Careful conclusion: the Inner Actor shows local resonance between weight tilts and future relative returns in representative windows, but it is not the main switching module and should not be claimed as a stable standalone alpha source.`

---

## Slide 20｜讨论与总结

### 中文可贴内容

`Discussion & Conclusion`

`Main Contributions`

`Framework innovation`

`CMTFlow 将投资组合强化学习中常被混在一起的三个问题显式解耦：when to revise、what base portfolio to hold、how to refine daily weights。`

`Trainable revision controller`

`Controller 将固定周期再平衡规则变成可学习的 hold/switch event policy，使模型能够根据当前持仓状态和候选组合自适应修订 base portfolio。`

`Empirical verification`

`在 Nasdaq-100 和 CSI-300 上，CMTFlow 取得了较好的高收益和低回撤权衡。消融和解释性实验表明，Controller 是主要的性能改善来源，Inner Actor 提供补充性的局部调权能力。`

`Limitations`

`Limited evaluation scope: 当前实验集中在两个 equity markets，且采用 long-only 和 proportional transaction cost 假设。`

`Room for extension: 后续可以扩展到更多资产类别、更真实的 market-impact models，以及显式风险偏好的 Controller 设计。`

`Takeaway`

`Learning when to revise is as important as learning what to hold.`

### English Copy

`Discussion & Conclusion`

`Main Contributions`

`Framework innovation`

`CMTFlow explicitly decouples three decisions that are often entangled in portfolio RL: when to revise, what base portfolio to hold, and how to refine daily weights.`

`Trainable revision controller`

`The Controller turns fixed-period rebalancing into a learnable hold/switch event policy, allowing the model to adaptively revise the base portfolio according to the current holding state and candidate replacement.`

`Empirical verification`

`On Nasdaq-100 and CSI-300, CMTFlow achieves a favorable high-return and controlled-drawdown trade-off. Ablation and interpretability studies confirm the Controller as the dominant source of performance gains and validate the complementary role of the Inner Actor in local within-segment weight refinement.`

`Limitations`

`Limited evaluation scope: current experiments are restricted to two equity markets under long-only and proportional transaction cost assumptions.`

`Room for extension: future work may extend the framework to broader asset classes, more realistic market-impact models, and Controller designs with explicit risk preferences.`

`Takeaway`

`Learning when to revise is as important as learning what to hold.`

---

## Slide 21｜Q&A

### 中文可贴内容

`Thank you for listening!`

`Any Questions?`

`CMTFlow: Controller-Guided Base Revision + Daily Refinement`

`可以准备回答三个问题：`

`1. 为什么 Slide 12 使用 \bar{\mathbf{w}}_t，而前面讲的是 \tilde{\mathbf{w}}_t 和 \mathbf{b}_t？`

`答：\bar{\mathbf{w}}_t 是 Controller 内部归一化后的 current holding feature，来源于 drifted executed holding \tilde{\mathbf{w}}_t，用来和 normalized candidate feature \bar{\mathbf{w}}_t^{\mathrm{cand}} 做比较；它不是新的最终组合决策。`

`2. 为什么 CSI-300 中 Outer + Controller 的收益高于 full CMTFlow？`

`答：这说明 Inner Actor 的收益贡献具有市场依赖性；论文主张 Controller 是主要动态修订机制，Inner Actor 是局部细化模块。`

`3. Slide 17 为什么要用 random switch matched-count comparison？`

`答：它控制 switch 次数后比较随机时机与 learned timing，用来说明优势不是来自 switch count 本身。`

### English Copy

`Thank you for listening!`

`Any Questions?`

`CMTFlow: Controller-Guided Base Revision + Daily Refinement`

`Possible questions to prepare:`

`1. Why does Slide 12 use \bar{\mathbf{w}}_t while previous slides use \tilde{\mathbf{w}}_t and \mathbf{b}_t?`

`Answer: \bar{\mathbf{w}}_t is the normalized current-holding feature used inside the Controller. It is derived from the drifted executed holding \tilde{\mathbf{w}}_t and compared with the normalized candidate feature \bar{\mathbf{w}}_t^{\mathrm{cand}}; it is not a new final portfolio decision.`

`2. Why does Outer + Controller obtain higher return than full CMTFlow on CSI-300?`

`Answer: this shows that the Inner Actor's return contribution is market-dependent. The paper's main claim is that the Controller is the dominant dynamic revision mechanism, while the Inner Actor is a local refinement module.`

`3. Why use random switch matched-count comparison on Slide 17?`

`Answer: it controls the number of switches and compares random timing with learned timing, showing that the advantage does not come from switch count alone.`
