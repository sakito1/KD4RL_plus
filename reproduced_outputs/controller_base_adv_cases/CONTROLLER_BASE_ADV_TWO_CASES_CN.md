# Controller Base–Adv 两市场可解释案例

## Material Passport

- Origin Skill：academic-research-suite / experiment-agent
- Origin Mode：validate
- Origin Date：2026-07-29
- Verification Status：VERIFIED
- Cases：NASDAQ-100 2025-05-08；CSI-300 2021-07-07
- Probe：保持市场与当前组合状态不变，仅令candidate等于当前漂移权重

## 1. 需要先修正的解释

代码结构中，Base head和Adv head共享同一个264维表征，且该表征同时包含当前组合、
candidate组合及二者差异。因此不能把Base严格定义为“只观察当前组合的状态”。

更准确的表述是：

- **Base**：在完整决策状态上给出的基础切换倾向；当前checkpoint中表现为稳定的
  保守Hold先验；
- **Adv**：对candidate相对当前组合吸引力的显式辅助预测，并通过有界logit修正
  调节Base；
- **最终动作**：比较 `Base + Adv correction` 是否越过0。

## 2. 两个案例

| Market     | Date       | Action   |   Segment return (%) |   Drawdown (%) |   Base logit |   Candidate ΔBase |   Adv raw |   Adv correction |   Candidate ΔAdv correction |   Final p |   Neutral-candidate p |   Future Hold (%) |   Future Candidate (%) |
|:-----------|:-----------|:---------|---------------------:|---------------:|-------------:|------------------:|----------:|-----------------:|----------------------------:|----------:|----------------------:|------------------:|-----------------------:|
| NASDAQ-100 | 2025-05-08 | Hold     |              16.5763 |         0      |      -1.0184 |            0.0057 |   -0.0099 |          -0.8674 |                     -2.3584 |    0.1317 |                0.6146 |           12.8869 |                 6.714  |
| CSI-300    | 2021-07-07 | Switch   |              -2.6495 |         2.6495 |      -0.9901 |            0.0071 |    0.0123 |           1.0424 |                      1.2226 |    0.5131 |                0.2355 |          -10.1752 |                 4.4299 |

## 3. NASDAQ-100：2025-05-08（为什么不切）

当前组合已经持有23日，区间收益
+16.58%，当前回撤0.00%：
这是一个状态良好的当前组合。Base logit为-1.018，
提供保守Hold先验；Outer candidate相对当前组合产生
-0.867的Adv修正，使最终Switch概率降至
0.132，Controller选择Hold。

未来20日冻结反事实中，继续Hold收益为+12.89%；
采用candidate收益为+6.71%，差值为
-6.17个百分点。
因此这个案例支持的是：当前组合状态好，且candidate相对更差，Adv进一步强化Hold。

## 4. CSI-300：2021-07-07（为什么切）

当前组合已经持有8日，区间收益
-2.65%，当前回撤2.65%。
Base logit为-0.990，仍未单独支持切换；candidate产生
+1.042的Adv修正，把最终概率推至
0.513并触发Switch。

未来20日中，继续Hold收益为-10.18%，采用candidate为
+4.43%，改善
+14.61个百分点。

## 5. Candidate消融如何解释

把candidate替换成当前组合后，市场状态、持仓时间、区间收益和回撤全部保持不变。
因此：

- `Candidate ΔBase` 衡量Base head受到candidate输入影响的程度；
- `Candidate ΔAdv correction` 衡量Adv通道对candidate差异的响应；
- 如果Adv变化明显大于Base变化，可以说该checkpoint在行为上形成了“稳定Base +
  candidate-sensitive Adv”的近似分工；
- 即使消融支持这种近似分工，也不能声称两个head在架构上完全解耦。

## 6. 解释边界

两个case用于解释计算闭环，不是总体有效性的统计证明。对NASDAQ-100全部231个
自由Switch决策做candidate=current消融后，没有一次出现“中性candidate为Hold、
真实candidate将其翻转为Switch”；只有1次Adv修正略微增加（+0.0018）。
因此不能为NASDAQ挑选一个并不存在的“candidate通过Adv推动Switch”的案例。

这里改用一个可核验的Hold案例，与CSI-300的Switch案例组成互补解释：

- NASDAQ-100：当前组合好、candidate较差，Adv强化Hold；
- CSI-300：当前组合弱、candidate较好，Adv克服Base先验并推动Switch。

总体统计还表明，CSI-300的Adv与训练对齐反事实优势存在弱正相关，而
NASDAQ-100没有显著全局关系。因此两例只能证明行为链条在具体决策上可以解释，
不能据此声称两个市场都有稳定、普遍的Adv预测能力。
