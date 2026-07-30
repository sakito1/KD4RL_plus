# Controller Base–Adv 互补案例图表报告

## Material Passport

- Origin：完整测试集Controller trace与冻结反事实轨迹
- Verification Status：VERIFIED
- Markets：NASDAQ-100 seed 49；CSI-300 seed 90
- Cases：NASDAQ-100 2023-01-17；CSI-300 2021-07-07
- Claim level：一个Hold优势案例 + 一个Switch优势案例

## 1. 图表要证明什么

本图验证的核心机制是：

`Base + Adv correction = Final logit`。

NASDAQ案例说明：弱正Adv不足以克服Base，最终Hold，事后当前组合更好。
CSI案例说明：candidate的正Adv克服Base，最终Switch，事后candidate更好。
两例共同展示Controller既不会对任意正Adv机械切换，也能在candidate优势足够强时
执行切换。

## 2. 案例数值

| Market     | Date       | Action   |   Held days |   Segment return (%) |   Drawdown (%) |   Base logit |   Raw Adv |   Adv correction |   Adv-only p (%) |   Final p (%) |   Candidate−Hold 20d (pp) |   Candidate−Hold 30d (pp) |
|:-----------|:-----------|:---------|------------:|---------------------:|---------------:|-------------:|----------:|-----------------:|-----------------:|--------------:|--------------------------:|--------------------------:|
| NASDAQ-100 | 2023-01-17 | Hold     |          17 |               7.0996 |         0      |      -1.0212 |    0.0109 |           0.9471 |          72.0533 |       48.1492 |                   -3.2806 |                   -1.164  |
| CSI-300    | 2021-07-07 | Switch   |           8 |              -2.6495 |         2.6495 |      -0.9901 |    0.0123 |           1.0424 |          73.9313 |       51.3077 |                   14.6051 |                   20.5611 |

### NASDAQ-100：Hold更好（2023-01-17）

当前组合持有17日，区间收益
+7.10%，回撤
0.00%。raw Adv为
0.01095，方向为正但低于约0.012的Base门槛。
Adv-only概率为72.05%，加入Base后降为
48.15%并Hold。candidate未来20日和30日分别落后
3.28和
1.16个百分点。

### CSI-300：Switch更好（2021-07-07）

当前组合持有8日，区间收益
-2.65%，回撤
2.65%。Base-only概率仅为
27.09%，单独倾向Hold；raw Adv为
0.01233，超过当前Base要求的门槛。
Adv correction把最终Switch概率推至
51.31%并触发Switch。candidate未来20日和30日分别领先
14.61和
20.56个百分点。
受控消融中，把candidate替换为当前组合后，Switch概率降为
23.55%并变为
Hold；说明该次动作翻转来自candidate相关的Adv通道，
而不是Base自身变化。

## 3. 可用于论文的案例结论

> The two cases illustrate complementary Controller behaviors. In NASDAQ-100,
> the Base hurdle filters a weak positive Adv signal and preserves the
> better-performing current portfolio. In CSI-300, a sufficiently strong
> candidate-relative Adv signal overcomes the conservative Base and triggers
> a profitable switch.

## 4. 案例解释边界

- 两个case用于展示机制闭环，不用于估计总体显著性；
- 未来收益只用于事后验证，没有进入当天Controller输入；
- 固定20/30日反事实冻结两套组合权重，比较的是同日起点下的配置差异；
- Base在当前checkpoint中接近固定负偏置，案例能说明其门槛作用，
  但不能证明学习型Base head不可替代；证明必要性仍需no-Base或fixed-threshold消融。
