# CMTFlow 20 分钟中文汇报 PPT 详细提纲

本文档用于指导制作一份约 20 分钟的中文学术汇报 PPT。内容分配参考 `SWAIB_TransG_slides.pptx`：先铺背景和问题，再讲相关工作与挑战，然后进入方法总览、建模、模块细节、训练流程，最后用实验结果、消融和解释性案例支撑结论。

设计目标：

- 讲述节奏接近模板：背景和问题先讲清楚，方法页连续展开，实验页作为证据链收束。
- 页面内容足够具体：每页列出标题、页面目的、版面元素、可直接放入 PPT 的文字、图表/素材、讲述重点。
- 适合手工制作或用可编辑模板版生成：每页内容都能拆成 PPT 原生文本框、卡片、箭头、图表和图片对象。
- 避免过度堆字：页面上只放关键短句，详细解释放到讲稿或备注。

---

## 总体结构与时间分配

- Slide 1：标题页，0.5 分钟。
- Slide 2-3：研究背景，约 1.7 分钟。
- Slide 4-5：问题定义，约 1.8 分钟。
- Slide 6-7：相关工作与挑战，约 1.6 分钟。
- Slide 8-14：方法与训练，约 7.4 分钟。
- Slide 15-18：实验设置、主结果、消融、解释性案例，约 5.2 分钟。
- Slide 19：讨论与总结，约 1.0 分钟。
- Slide 20：Q&A，机动。

总讲述时间约 19 分钟，留 1 分钟给过渡和临场解释。

---

## Slide 1: 标题页

- 时间：0.5 分钟
- 页面角色：封面
- 对应模板：标题幻灯片
- 页面目的：
  - 让听众第一眼知道研究对象、方法关键词和报告主题。
  - 建立主线：组合管理不是单纯每日权重预测，而是多时间尺度决策。

### 页面上具体放什么

- 主标题：
  - `CMTFlow：控制器引导的分层投资组合管理框架`
- 英文副标题：
  - `Hierarchical Portfolio Management with Controller-Guided Base Revision and Daily Refinement`
- 作者信息：
  - `作者 / 单位 / 日期`
- 一句主线提示：
  - `显式协调：何时换仓、换成什么、如何每日微调`

### 建议版式

- 白色背景，深蓝色边框和左上/右下斜角块，贴近模板封面。
- `CMTFlow` 用深红色强调，其余标题用黑色或深蓝色。
- 标题居中偏上，副标题位于标题下方，作者信息在页面中下部。
- 底部可放一条细线或短句，不放复杂图。

### 视觉元素

- 可选轻量装饰：浅灰色金融曲线、抽象分层箭头、淡色网格。
- 不建议放真实实验图，封面保持干净。

### 讲述重点

- 开场只讲主线，不展开技术细节。
- 强调报告围绕三个问题展开：`when / what / how`。

---

## Slide 2: 研究背景：投资组合管理的基本目标

- 时间：0.8 分钟
- 页面角色：背景
- 对应模板：Background 第 1 页
- 页面目的：
  - 解释投资组合管理的基本目标。
  - 说明为什么这个任务天然是长期序列决策。

### 页面上具体放什么

- 标题：
  - `研究背景：投资组合管理的基本目标`
- 左侧短句或 bullet：
  - `多资产资本配置：在不确定市场中分配资金`
  - `目标不是单纯收益最大化，而是收益、风险、成本的长期权衡`
  - `强化学习适合建模序列决策，但金融市场持续非平稳`
- 右侧三角结构：
  - 顶点 1：`收益`
    - 小字：`Total Return / Annual Return`
  - 顶点 2：`风险`
    - 小字：`Volatility / Max Drawdown`
  - 顶点 3：`成本`
    - 小字：`Turnover / Transaction Cost`
- 底部 takeaway：
  - `好的策略应改善整条投资路径，而不只是提高某一天的预测准确率`

### 建议版式

- 左侧 45% 放文字解释。
- 右侧 45% 放收益-风险-成本三角图。
- 底部横条放 takeaway。

### 视觉元素

- 三角图使用深蓝、深红、灰色。
- 可加一条简化累计财富曲线作为背景，但不要抢主内容。

### 讲述重点

- 强调金融策略不能只看终点收益，还要看回撤、波动和交易成本。
- 为后面讲 `reward = log return - cost` 做铺垫。

---

## Slide 3: 研究背景：组合决策不是单一日频动作

- 时间：0.9 分钟
- 页面角色：背景延展 / 决策流程
- 对应模板：Background 第 2 页
- 页面目的：
  - 引出多时间尺度决策。
  - 区分中期持仓、每日微调、异常退出。

### 页面上具体放什么

- 标题：
  - `研究背景：组合决策不是单一日频动作`
- 中间三阶段循环：
  - `中期持仓`
    - `形成稳定 base portfolio`
  - `每日修正`
    - `围绕基准组合做局部微调`
  - `异常退出`
    - `市场状态恶化时触发切换`
- 左下对比：
  - `固定周期再平衡：稳定但反应慢`
- 右下对比：
  - `纯日频调仓：灵活但易受噪声驱动`
- 底部 takeaway：
  - `需要把“持仓段”和“每日微调”分开建模`

### 建议版式

- 中间用环形箭头或横向流程展示三个阶段。
- `异常退出 / hold-switch` 用红色强调。
- 页面不要放公式，重在讲直觉。

### 视觉元素

- 三个卡片或三个圆角节点。
- 节点之间用深蓝箭头连接。
- 在 `异常退出` 旁放一个问号或警示标识，暗示这是关键难点。

### 讲述重点

- 固定再平衡和日频调仓各有问题。
- 本文的目标不是替代所有日频策略，而是把不同时间尺度分工清楚。

---

## Slide 4: 问题定义：带漂移和交易成本的动态组合

- 时间：0.9 分钟
- 页面角色：问题定义
- 对应模板：Problem Definition 第 1 页
- 页面目的：
  - 给出动态组合管理的基本执行链条。
  - 让听众理解 drifted portfolio 和 transaction cost。

### 页面上具体放什么

- 标题：
  - `问题定义：带漂移和交易成本的动态组合`
- 横向流程：
  - `w_{t-1}`
    - `昨日执行组合`
  - `Drift`
    - `价格变化导致权重漂移`
  - `Rebalance`
    - `主动调整产生换手`
  - `w_t`
    - `当日最终执行权重`
  - `Reward`
    - `log return - transaction cost`
- 右侧或底部小公式：
  - `turnover_t = ||w_t - w_t^drift||_1`
  - `reward_t = log return_t - cost_t`
- 评价指标：
  - `TR / AR / Vol / Sharpe / MDD / CR`

### 建议版式

- 中间放横向流程图，占 60%-70% 页面宽度。
- 底部放公式条和指标标签。
- 每个流程节点用可编辑卡片，不要用长段文字。

### 视觉元素

- `Drift` 节点可用浅蓝色。
- `Rebalance / cost` 用红色强调。
- `Reward` 用深蓝色。

### 讲述重点

- 即使不主动交易，组合权重也会自然漂移。
- 交易成本让“频繁调整”不一定好。

---

## Slide 5: 问题定义：本文关注的三个核心决策

- 时间：0.9 分钟
- 页面角色：概念解释
- 对应模板：Problem Definition 第 2 页
- 页面目的：
  - 把全文问题拆成 `When / What / How`。
  - 建立三个模块和三个问题的一一对应关系。

### 页面上具体放什么

- 标题：
  - `问题定义：本文关注的三个核心决策`
- 三张并列卡片：
  - 卡片 1：
    - 大字：`When`
    - 问题：`当前 base portfolio 是否已经失效？`
    - 模块：`Controller`
    - 输出：`hold / switch`
  - 卡片 2：
    - 大字：`What`
    - 问题：`如果切换，新的中期基准组合是什么？`
    - 模块：`Outer Actor`
    - 输出：`candidate base portfolio`
  - 卡片 3：
    - 大字：`How`
    - 问题：`在基准组合内部，如何每日权重微调？`
    - 模块：`Inner Actor`
    - 输出：`executed weights`
- 底部 takeaway：
  - `不同时间尺度的决策，不应压缩进单一每日动作`

### 建议版式

- 三卡片均分页面宽度。
- `When / Controller` 用红色，`What / Outer` 用蓝色，`How / Inner` 用紫色或绿色。
- 卡片下方用一条细线连接到下一页方法总览。

### 视觉元素

- 每张卡片可以加一个小图标：
  - `When`：时钟或开关。
  - `What`：资产篮子或柱状图。
  - `How`：调节滑杆或微调箭头。

### 讲述重点

- 这页是全报告的概念支点，需要讲清楚。
- 后面所有模块都围绕这三个问题展开。

---

## Slide 6: 相关工作：从静态优化到深度强化学习

- 时间：0.8 分钟
- 页面角色：相关工作 / 对比
- 对应模板：Related Work
- 页面目的：
  - 简洁交代已有方法脉络。
  - 说明本文的切入点不是“又一个 RL 投资组合模型”，而是 base revision timing。

### 页面上具体放什么

- 标题：
  - `相关工作：从静态优化到深度强化学习`
- 三列对比：
  - `Classical Optimization`
    - `均值方差、CAPM、规则再平衡`
    - 优点：`解释性强`
    - 局限：`静态假设 / 单期目标`
  - `Deep RL Portfolio`
    - `PGPortfolio、FinRL、DeepTrader 等`
    - 优点：`可直接优化长期收益`
    - 局限：`常把中期配置与每日执行压缩为一个动作`
  - `Adaptive / HRL`
    - `HADAPS、DeepAries 等层级 / 自适应再平衡方法`
    - 优点：`增强策略灵活性`
    - 局限：`缺少显式可持有、漂移、替换的 base portfolio memory`
- 底部本文定位：
  - `本文关注：基准组合何时被修正`

### 建议版式

- 三列卡片，每列上方是方法类别，下方是 `优点 / 局限`。
- 底部红色强调条写本文切入点。

### 视觉元素

- 用表格或三栏卡片都可以。
- 不需要放引用列表，避免页面过满。

### 讲述重点

- 不要展开过多文献细节。
- 重点是让听众理解本文和已有工作的差异。

---

## Slide 7: 挑战：直接应用 RL 仍存在非平凡限制

- 时间：0.8 分钟
- 页面角色：挑战页
- 对应模板：When applying RL directly...
- 页面目的：
  - 从相关工作自然过渡到 CMTFlow 的必要性。
  - 强调换仓时机是独立难题。

### 页面上具体放什么

- 标题：
  - `挑战：直接应用 RL 仍存在非平凡限制`
- 左侧三个挑战：
  - `1. 日频动作容易被短期噪声驱动`
  - `2. 固定再平衡无法识别持仓状态是否恶化`
  - `3. 换仓需要同时比较旧组合、候选组合、成本和持仓年龄`
- 右侧关键问题：
  - `Should we hold or switch today?`
  - 下方两个选项：
    - `Hold: keep drifted base`
    - `Switch: call Outer for new base`
- 底部 takeaway：
  - `需要显式 Controller 学习 hold/switch，而不是机械按日历换仓`

### 建议版式

- 左侧 55% 放挑战列表。
- 右侧 35% 放 hold/switch 决策示意。
- `Switch` 或 `Controller` 用红色突出。

### 视觉元素

- 可放问号、分叉箭头、开关按钮。
- 右侧可以画旧 base 和新 candidate 两个小组合卡片。

### 讲述重点

- Controller 的必要性来自“何时换仓”这个独立难题。
- 它不是替代选股模块，而是决定是否结束当前持仓段。

---

## Slide 8: 方案总览：CMTFlow 的统一分层结构

- 时间：1.2 分钟
- 页面角色：方案总览 / 架构
- 对应模板：Overview of Our Solution
- 页面目的：
  - 一页展示 CMTFlow 的全局结构。
  - 让听众知道后面几页会分别解释三个模块。

### 页面上具体放什么

- 标题：
  - `方案总览：CMTFlow 的统一分层结构`
- 主体图：
  - 使用整体架构图。
  - 图要占页面 60%-70% 宽度。
- 右侧 callout：
  - `Outer：生成候选 base portfolio`
  - `Controller：判断 hold / switch`
  - `Inner：围绕当前 base 每日微调`
- 底部 takeaway：
  - `核心：将组合管理拆成 when / what / how 三个协同决策`

### 建议版式

- 左侧或中间放大图，右侧放三条 callout。
- 图外加细边框，避免和白底混在一起。
- callout 不要遮挡图内标签。

### 必需图片

- 架构图；严格输入资产；保留模块层次、箭头和标签。

  ![CMTFlow Architecture](figures/cmtflow_architecture_vector.png)

### 讲述重点

- 先讲整体输入和环境反馈，再讲三个策略模块。
- 强调三个模块不是独立策略，而是同一决策流中的分工。

---

## Slide 9: 强化学习建模：状态、动作与奖励

- 时间：0.8 分钟
- 页面角色：形式化建模
- 对应模板：MARL Formulation
- 页面目的：
  - 将 Slide 8 的直观结构转成 RL formulation。
  - 让方法页有形式化支撑，但不陷入公式细节。

### 页面上具体放什么

- 标题：
  - `强化学习建模：状态、动作与奖励`
- 四个模块框：
  - `State`
    - `market features`
    - `recent market tensor`
    - `portfolio state`
    - `candidate base`
  - `Action`
    - `candidate base portfolio`
    - `hold / switch`
    - `executed weights`
  - `Environment`
    - `price movement`
    - `transaction cost`
    - `portfolio value`
  - `Reward`
    - `log return - cost`
- 底部评价：
    - `Evaluation: TR / AR / Vol / Sharpe / MDD / CR`

### 建议版式

- 横向闭环图：`State -> Action -> Environment -> Reward -> State`。
- 每个框只放 3-4 个短词。
- 奖励框用红色强调交易成本。

### 视觉元素

- 箭头闭环。
- 状态和动作使用蓝色，奖励使用红色。

### 讲述重点

- 动作不是单一权重，而是由三层动作构成。
- 奖励关注完整投资路径，不是单日预测。

---

## Slide 10: 方法架构：从候选组合到最终执行权重

- 时间：1.2 分钟
- 页面角色：方法主流程
- 对应模板：Methodology - architecture
- 页面目的：
  - 展开每日决策链条。
  - 讲清 hold/switch 如何决定 active base。

### 页面上具体放什么

- 标题：
  - `方法架构：从候选组合到最终执行权重`
- 主体图：
  - 决策流程图。
- 右侧或底部公式：
  - `b_t = w_t^{cand}, if switch`
  - `b_t = \tilde{b}_t, if hold`
  - `w_t = Inner(b_t, state_t)`
- 3 条解释：
  - `Outer 生成候选组合 w_t^{cand}`
  - `Controller 判断是否替换当前 base`
  - `Inner 始终输出最终执行权重 w_t`

### 建议版式

- 左侧大图，右侧公式和解释。
- `Controller` 分叉点用红框或箭头强调。
- 公式条放在图下方也可以。

### 必需图片

- 决策流程图；严格输入资产；保留分支逻辑和公式语义。

  ![CMTFlow Decision Flow](figures/cmtflow_decision_flow_imagegen.png)

### 讲述重点

- Controller 不是单独预测收益，而是在旧 base 和新 candidate 之间做行动选择。
- base portfolio 可持有、可漂移、可替换，这是本文区别于普通日频策略的关键。

---

## Slide 11: 方法细节：Outer Actor 生成中期基准组合

- 时间：0.8 分钟
- 页面角色：方法模块
- 对应模板：Centralized Graph Generator / module detail
- 页面目的：
  - 解释 Outer 的职责边界。
  - 强调它是中期资产选择器，而非每日执行器。

### 页面上具体放什么

- 标题：
  - `方法细节：Outer Actor 生成中期基准组合`
- 主图：
  - 三模块图，突出 Outer 区域。
- 左侧或右侧结构框：
  - `Input`
    - `long lookback market window`
    - `weights_drift`
  - `Encoder`
    - `asset-level temporal encoding`
    - `cross-asset representation`
  - `Output`
    - `top-K candidate base portfolio`
- 底部一句话：
  - `Outer 回答 What to hold next，而不是 How to trade today`

### 建议版式

- 主图占 55%-60% 页面。
- 旁边放 `Input -> Encoder -> Output` 三段卡片。
- `top-K`、`base portfolio` 用红色强调。

### 必需图片

- 三模块图；严格输入资产；保留模块名和关系。

  ![CMTFlow Three Modules](figures/cmtflow_architecture_vector.png)

### 讲述重点

- Outer 形成的是下一持仓段的锚点。
- 它降低日频策略完全重构股票池带来的噪声。

---

## Slide 12: 方法细节：Controller 学习何时替换基准组合

- 时间：1.2 分钟
- 页面角色：方法模块 / 关键机制
- 对应模板：Methodology - key mechanism
- 页面目的：
  - 解释 Controller 为什么是核心贡献。
  - 讲清它如何把固定周期换仓变成可学习事件策略。

### 页面上具体放什么

- 标题：
  - `方法细节：Controller 学习何时替换基准组合`
- 主体：
  - 复用决策流程图，突出 Controller 分叉点。
- Controller 输入卡片：
  - `current holding state`
  - `recent market tensor`
  - `candidate base`
  - `holding-state vector`
  - `turnover / concentration / overlap`
- Controller 输出卡片：
  - `exit probability`
  - `hold / switch`
- 评价规则小字：
  - `daily decision stride`
  - `threshold = 0.5`
  - `H_max = 30`
- 底部 takeaway：
  - `不是固定 5/10/20/30/60 天切换，而是状态依赖的事件触发`

### 建议版式

- 左侧放图，右侧放输入/输出卡片。
- `hold` 用蓝色，`switch` 用红色。
- 底部用红色横条强调和固定窗口的区别。

### 必需图片

- 决策流程图；严格输入资产；保留 Controller 分支和标签。

  ![CMTFlow Decision Flow](figures/cmtflow_decision_flow_imagegen.png)

### 讲述重点

- Controller 是决定当前持仓段是否结束的模块。
- 它的价值需要在消融和随机切换实验中验证。

---

## Slide 13: 方法细节：Inner Actor 做每日局部微调

- 时间：0.8 分钟
- 页面角色：方法模块
- 对应模板：Coordinated Policy Generation
- 页面目的：
  - 解释 Inner 的职责边界。
  - 避免把 Inner 误解为另一个独立选股模块。

### 页面上具体放什么

- 标题：
  - `方法细节：Inner Actor 做每日局部微调`
- 主图：
  - 三模块图，突出 Inner 区域。
- 核心概念：
  - `base weights`
  - `inner tilt = executed weights - base weights`
  - `executed weights`
- 三条说明：
  - `不重新决定股票池`
  - `在 active base portfolio 内调整权重`
  - `提供日频灵活性，但不负责主要换仓控制`
- 底部 caution：
  - `Inner 是局部 refinement，不应表述为稳定独立 alpha 来源`

### 建议版式

- 主图一侧，另一侧画 `base weights -> tilt -> executed weights`。
- `tilt` 用紫色或绿色强调。
- caution 可放在底部浅灰提示框。

### 必需图片

- 三模块图；严格输入资产；保留模块名和关系。

  ![CMTFlow Three Modules](figures/cmtflow_architecture_vector.png)

### 讲述重点

- Inner 和 Controller 是互补关系。
- Controller 管“是否换仓”，Inner 管“当前持仓内怎么微调”。

---

## Slide 14: 模型训练：固定 HRL 预训练与 Controller 学习

- 时间：1.2 分钟
- 页面角色：训练流程
- 对应模板：Model Training + Pseudocode
- 页面目的：
  - 说明训练为什么分阶段。
  - 澄清 fixed 30-day 是训练参考，不是最终决策规则。

### 页面上具体放什么

- 标题：
  - `模型训练：固定 HRL 预训练与 Controller 学习`
- 主体图：
  - 训练流程图。
- 右侧三阶段：
  - `Stage 1: Fixed HRL warmup`
    - `train Outer / Inner under fixed segments`
  - `Stage 2: Controller learning`
    - `learn daily hold/switch policy`
  - `Stage 3: Final evaluation`
    - `daily event policy, no fixed calendar rule`
- 底部强调：
  - `fixed 30-day schedule 是训练参考路径，不是最终测试规则`

### 建议版式

- 左侧放训练流程图。
- 右侧放三阶段竖向时间线。
- `final evaluation` 用红色强调。

### 必需图片

- 训练流程图；严格输入资产；保留训练阶段和反馈关系。

  ![CMTFlow Training Flow](figures/cmtflow_training_flow_vector.png)

### 讲述重点

- 先让 Outer/Inner 学会构造和微调组合，再训练 Controller 学会什么时候结束持仓段。
- 这个流程降低三层策略同时训练的不稳定性。

---

## Slide 15: 实验设置

- 时间：0.8 分钟
- 页面角色：实验设置
- 对应模板：Experimental Setup
- 页面目的：
  - 交代数据集、划分、baseline 和指标。
  - 证明实验协议是时间顺序、无未来信息泄漏的。

### 页面上具体放什么

- 标题：
  - `实验设置`
  - 数据集表格：
  - `Nasdaq-100`
    - `#Stocks: 39`
    - `Train: 2000/04/07--2017/12/29`
    - `Valid: 2018/01/02--2020/04/22`
    - `Test: 2020/04/23--2025/10/03`
  - `CSI-300`
    - `#Stocks: 53`
    - `Train: 2000/04/07--2017/12/28`
    - `Valid: 2018/01/02--2019/12/31`
    - `Test: 2020/01/02--2025/02/28`
- Baseline 卡片：
  - `Traditional strategies`
  - `Deep RL baselines`
  - `Fixed 5/10/20/30/60-day controllers`
- Metric 卡片：
  - `TR / AR / Vol / Sharpe / MDD / CR`
- Setting 小字：
  - `Transaction cost rate = 5e-5`
  - `Outer window = 60`
  - `Inner window = 10`
  - `Controller window = 30`

### 建议版式

- 左侧 65% 放数据集表格。
- 右侧 30% 放 Baseline 和 Metrics 两个卡片。
- 底部放主要超参数，不要太醒目。

### 视觉元素

- 表格使用深蓝表头、浅灰交替行。
- 市场名称可用红色或粗体标出。

### 讲述重点

- 重点讲 chronological split 和两个市场。
- baseline 不需要逐个展开。

---

## Slide 16: 数值结果：主实验性能对比

- 时间：1.6 分钟
- 页面角色：核心结果 / 数据证据
- 对应模板：Performance Comparison / Numerical Results
- 页面目的：
  - 展示主实验的风险收益权衡。
  - 明确不是所有指标第一，而是更均衡。

### 页面上具体放什么

- 标题：
  - `数值结果：主实验性能对比`
- 左侧图：
  - `Main Equity Curves`
- 右侧图：
  - `Main Metric Bars`
- 图旁关键数字：
  - `Nasdaq-100: TR = 265.53%, MDD = 18.62%`
  - `WMAMR MDD = 33.88%, Anticor MDD = 44.59%`
  - `CSI-300: TR = 204.99%, Sharpe = 1.14, MDD = 22.78%`
  - `DeepTrader: TR = 212.81%, Sharpe = 1.03, MDD = 31.86%`
- 底部 takeaway：
  - `CMTFlow 改善风险收益权衡，而不是追求所有单项指标绝对最优`

### 建议版式

- 左右双图布局。
- 底部横条写结论。
- 关键数字用小 callout，不要遮挡图。

### 必需图片

- 累计收益曲线；严格输入资产；保留坐标轴、图例、曲线和数值关系。

  ![Main Equity Curves](figures/main_equity_curves.png)

- 主指标柱状图；严格输入资产；保留指标、颜色和图例含义。

  ![Main Metric Bars](figures/main_metric_bars.png)

### 讲述重点

- Nasdaq 上，CMTFlow 收益最高或接近最高，同时显著降低高收益传统策略的回撤。
- CSI-300 上，收益略低于 DeepTrader，但 Sharpe 更高、MDD 更低。
- 避免说“所有指标均最优”。

---

## Slide 17: 数值结果：消融实验与机制验证

- 时间：1.3 分钟
- 页面角色：消融结果 / 机制验证
- 对应模板：Ablation Study / Sensitivity Analysis
- 页面目的：
  - 证明 Controller 是主要有效组件。
  - 证明优势不是固定频率切换带来的。

### 页面上具体放什么

- 标题：
  - `数值结果：消融实验与机制验证`
- 主图：
  - `Ablation Metric Bars`
- 右侧或底部数字 callout：
  - `Nasdaq: Outer-only TR 220.42%, MDD 32.09%`
  - `Nasdaq: Outer + Controller TR 237.50%, MDD 21.24%`
  - `Full CMTFlow: TR 265.53%, MDD 18.62%`
  - `CSI-300: Outer-only TR 147.05%, Sharpe 0.94`
  - `CSI-300: Outer + Controller TR 237.77%, Sharpe 1.22`
- 底部 takeaway：
  - `Learned controller 不能被固定 5/10/20/30/60 天切仓规则替代`

### 建议版式

- 左侧放消融图，占 65%-70%。
- 右侧放 `Controller contribution` 卡片和 `Fixed-window check` 卡片。
- `Controller` 和 `fixed-window` 的对比用红/蓝色区分。

### 必需图片

- 消融指标图；严格输入资产；保留模型变体、指标、坐标轴和颜色。

  ![Ablation Metric Bars](figures/ablation_metric_bars.png)

### 讲述重点

- Controller 是主要动态适应来源。
- Inner Actor 的贡献更市场依赖，应解释为局部调权模块。
- CSI-300 中 Outer + Controller 收益高于 full model，因此不要夸大 Inner。

---

## Slide 18: 案例研究：Controller 与 Inner 的可解释行为

- 时间：1.5 分钟
- 页面角色：案例研究 / 可解释性
- 对应模板：Case Study
- 页面目的：
  - 把主结果和消融结论转化为具体经济直觉。
  - 展示 Controller 不是固定持仓窗口可以替代，Inner 有局部调权行为。

### 页面上具体放什么

- 标题：
  - `案例研究：Controller 与 Inner 的可解释行为`
- 三个图块：
  - `Controller switch cases`
  - `Fixed-window comparison`
  - `Inner actor base adjustment`
- 图旁短 callout：
  - `Controller case: switch 可改善 30 日冻结反事实收益并降低未来回撤`
  - `Fixed windows: learned controller 优于固定 5/10/20/30/60 日持仓窗口`
  - `Inner tilt: executed weight - base weight 与未来相对收益存在局部共振`
- 可放关键 case 数字：
  - `Nasdaq case: hold -1.91%, switch 0.52%; MDD 8.82% -> 5.51%`
  - `CSI-300 case: hold -10.29%, switch 10.25%; MDD 13.71% -> 8.23%`
  - `Fixed windows: Ours has higher TR, Sharpe, and CR than all fixed-window variants on CSI-300`
  - `Inner tilt correlation: Nasdaq 0.46, CSI-300 0.33`
- 底部 takeaway：
  - `Controller 的价值主要体现在关键风险窗口；Inner 提供局部持仓修正`

### 建议版式

- 三图并排，图上方或下方放短标题。
- 不要试图在这一页解释所有图例，只讲每张图证明什么。
- 若图太密，可把三图变成两行：上方 controller case，下方 fixed-window comparison + inner adjustment。

### 必需图片

- Controller 切换案例；严格输入资产。

  ![Controller Switch Cases](figures/explainability/controller_switch_cases.png)

- 固定持仓窗口对比；严格输入资产。

  ![Fixed Window Comparison](figures/explainability/fixed_window_comparison.png)

- Inner Actor 局部调权；严格输入资产。

  ![Inner Actor Base Adjustment](figures/explainability/inner_actor_base_adjustment.png)

### 讲述重点

- Case 图说明关键风险窗口中 switch 有经济含义。
- Fixed-window comparison 说明不是选择某个固定持仓周期即可复现。
- Inner 图说明它是局部 refinement，而不是主要风险规避模块。

---

## Slide 19: 讨论与总结

- 时间：1.0 分钟
- 页面角色：结论
- 对应模板：Discussion & Conclusion
- 页面目的：
  - 收束全文贡献。
  - 明确结论边界和未来工作。

### 页面上具体放什么

- 标题：
  - `讨论与总结`
- 三个贡献卡片：
  - `1. 问题重构`
    - `base revision`
    - `base construction`
    - `daily refinement`
  - `2. 核心机制`
    - `Controller 将固定周期换仓变成可学习事件策略`
  - `3. 实验结论`
    - `更稳健的风险收益权衡`
    - `Controller 是主要自适应来源`
- 未来工作条：
  - `更多市场与资产类别`
  - `更复杂交易约束`
  - `更稳健风险控制`
- 底部 takeaway：
  - `Learning when to revise is as important as learning what to hold`

### 建议版式

- 三个贡献卡片横向排列。
- 未来工作放底部横条。
- 英文 takeaway 可以用斜体或深蓝色小字。

### 视觉元素

- `Controller`、`when to revise` 用红色强调。
- 贡献卡片背景用浅灰或浅蓝。

### 讲述重点

- 回到 `when / what / how` 主线。
- 不夸大结论：说明 Controller 是主要机制，Inner 是补充。

---

## Slide 20: Q&A

- 时间：机动
- 页面角色：结束页
- 对应模板：Thank you / Any Questions
- 页面目的：
  - 简洁收尾，引导提问。

### 页面上具体放什么

- 大标题：
  - `谢谢，欢迎提问`
- 副标题：
  - `Learning when to revise is as important as learning what to hold.`
- 可选底部小字：
  - `CMTFlow`

### 建议版式

- 仿模板结束页：白底、深蓝边框、居中大字。
- 不放复杂图表。
- 可用左上/右下深蓝斜角保持风格统一。

### 讲述重点

- 只保留一句核心 takeaway。
- 准备应对三个问题：
  - `为什么 Inner 在 CSI-300 中没有进一步提高收益？`
  - `Controller 是否会增加换手和交易成本？`
  - `固定窗口对比如何保证公平？`

---

## 附：制作 PPT 时的统一规范

### 页面风格

- 参考模板：白底、深蓝边框、左上/右下深蓝斜角、红色强调词。
- 标题一般放左上，字号要明显大于正文。
- 每页最多 1 个主视觉中心，不要同时塞太多图。
- 英文模块名可以保留，如 `Outer Actor`、`Controller`、`Inner Actor`、`base portfolio`。

### 文字密度

- 页面正文尽量用短句，不放完整讲稿。
- 每页最多 3-5 个主要 bullet。
- 结果页允许放关键数字，但不要把整张表搬上去。

### 图表使用

- 实验图必须保留原始坐标轴、图例、颜色和数值关系。
- 架构图可以裁剪或放大，但不要重绘导致模块关系变形。
- 如果图中小字太密，PPT 页面旁边用 callout 摘出结论，而不是强迫听众读全图。

### 讲述节奏

- Slide 2-7：讲为什么需要这个问题。
- Slide 8-14：讲 CMTFlow 怎么解决。
- Slide 15-18：讲证据链是否支持。
- Slide 19-20：收束贡献和问答。
