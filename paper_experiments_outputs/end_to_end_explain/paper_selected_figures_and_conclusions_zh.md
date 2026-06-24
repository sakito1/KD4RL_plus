# End-to-End HRL Controller 解释实验：双市场展示图与结论

实验输出目录：`paper_experiments_outputs/end_to_end_explain`

PDF 说明：默认 `.pdf` 是由高分辨率 PNG 展平生成的字体安全版，适合论文插图和避免字体读取乱码；同目录下的 `_editable.pdf` 保留 matplotlib 矢量文字，适合后期编辑。

这版报告按 SH 与 NAS 两个市场并列组织。新增的 `Controller+Outer` 消融表示：controller 可以决定是否 switch，并使用 outer 组合动作；inner 动作被置零，用于分离“controller/outer 切换”与“inner 权重微调”的贡献。

## 推荐主文图

1. 双市场消融柱状图
   - SH：`figures/02_inference_ablation/fig03c_inference_ablation_bar_sh_seed90.pdf`
   - NAS：`figures/02_inference_ablation/fig03c_inference_ablation_bar_nas_seed49.pdf`
   - 用途：直接展示 No-Inner Fixed、Fixed HRL、Controller+Outer、Full Controller 四种推理设置的收益、Sharpe、最大回撤和切换次数。

2. 双市场 controller case window
   - SH：`figures/07_case_windows/fig10_case_window_sh_2021_07_large_avoidance.pdf`
   - NAS：`figures/07_case_windows/fig09_case_window_nas_2021_04_negative_hold.pdf`
   - 用途：回答“controller 在什么情况下 switch”。图中标出窗口内 switch 点、exit probability、窗口收益/回撤，以及关键 switch 后的 20 日继续持仓反事实与切换反事实。

3. NAS switch cluster 补充主图
   - `figures/07_case_windows/fig08_case_window_nas_2020_07_switch_cluster.pdf`
   - 用途：NAS 切换频繁，适合用完整 30 日持仓窗口说明多个 switch 共同压低回撤，而不是强行解释单个 switch。

4. 双市场随机切换对照
   - SH：`figures/06_random_switch/fig07_random_switch_comparison_sh_seed90.pdf`
   - NAS：`figures/06_random_switch/fig07_random_switch_comparison_nas_seed49.pdf`
   - 用途：说明 controller 的行为不是简单“多切几次”。

## Controller+Outer 消融结论

**NAS。** Fixed HRL 的测试期总收益为 228.19%，最大回撤为 31.73%；加入 Controller+Outer 后，总收益升至 238.28%，最大回撤降至 21.24%。Full Controller 进一步把总收益提高到 266.37%，最大回撤降至 18.62%。因此 NAS 上 controller/outer 已经贡献了主要的回撤控制，inner 在 controller-active setting 下继续带来收益提升和额外回撤降低。

**SH。** Fixed HRL 的测试期总收益为 155.37%，Sharpe 为 0.98；Controller+Outer 将总收益提高到 233.05%，Sharpe 提高到 1.21，是 SH 中收益和 Sharpe 最强的设置。Full Controller 的总收益为 200.73%，Sharpe 为 1.12，最大回撤 22.78%，略低于 Controller+Outer 的 23.29%。这说明 SH 上 controller/outer 本身已经很强，inner 加入后更偏向轻微降低回撤，但会牺牲一部分收益。

## Case 解释结论

**SH case。** 2021-06-11 至 2021-07-23 的 30 个交易日窗口中，Full Controller 收益为 3.28%，Fixed HRL 为 1.24%；窗口最大回撤为 3.19%，Fixed HRL 为 4.11%。关键 switch 发生在 2021-07-07：如果继续持有原组合，未来 20 日反事实收益为 -10.74%；切换后的反事实收益为 4.31%，避免损失 15.05 个百分点。这是最清楚的“switch 避免大幅下跌”案例。

**NAS case B。** 2021-03-25 至 2021-05-06 的 30 个交易日窗口中，Full Controller 收益为 8.18%，Fixed HRL 为 6.07%；窗口最大回撤为 2.66%，Fixed HRL 为 3.48%。关键 switch 发生在 2021-04-19：继续持仓反事实 20 日收益为 -6.05%，切换反事实为 -2.63%，避免损失 3.42 个百分点。

**NAS case A。** 2020-07-08 至 2020-08-18 的 30 个交易日窗口中，controller 触发 18 次 free switch。Full Controller 收益 7.89%，Fixed HRL 为 5.25%；窗口最大回撤 2.45%，Fixed HRL 为 4.64%，回撤降低 2.19 个百分点。该图适合解释 NAS 中“多次切换共同控制回撤”的行为。

## 论文可用表述

**Controller 有效性。** 两个市场的消融结果表明，controller 的主要作用来自主动切换和 outer 组合选择。在 NAS 上，Controller+Outer 已经将最大回撤从 31.73% 降至 21.24%，Full Controller 进一步降至 18.62%；在 SH 上，Controller+Outer 将收益和 Sharpe 大幅提升，说明 controller/outer 对市场状态切换具有直接贡献。

**Inner 与 Controller 的分工。** 在固定 HRL 推理中，inner module 对两个市场均有正收益贡献：NAS 上 Fixed HRL 相比 No-Inner 提高 7.03 个百分点，SH 上提高 11.77 个百分点。加入 controller 后，inner 的效果表现为市场依赖：NAS 上 full 比 Controller+Outer 进一步提升收益并降低回撤；SH 上 Controller+Outer 的收益最高，full 则略微降低回撤但收益回落。因此论文中不宜把 inner 简化为“总是提高最终收益”，更准确的表述是：inner 在固定持仓框架中提升组合内部配置能力，在 controller-active 框架中与切换频率、交易路径共同作用。

**Controller 行为解释。** Case window 图显示，controller 的 switch 往往出现在继续持有反事实开始变差的阶段。SH 的关键 case 中，继续持仓未来 20 日会产生 -10.74% 的收益，而切换反事实为 4.31%；NAS 的关键 case 中，继续持仓未来 20 日为 -6.05%，切换后为 -2.63%。这些局部反事实与全局消融共同说明，controller 不是只增加交易次数，而是在部分高风险窗口中改变风险暴露，从而避免后续收益劣化或降低回撤。

**随机切换对照。** 在 50 组 matched-count random switch 中，SH 的 Full Controller 收益和 Sharpe 均高于所有随机对照；NAS 的 Full Controller 收益/Sharpe 约处于 54 分位，但最大回撤 18.62% 优于所有随机对照，随机对照平均最大回撤为 23.53%。因此 SH 更适合强调收益与 Sharpe，NAS 更适合强调回撤控制。

## 不建议强放主文

1. `figures/04_switch_alignment/fig05_exit_prob_calibration_nas_seed49.pdf`
   - NAS 上 exit probability 与 switch advantage 的线性相关较弱，适合放附录或不放。

2. `figures/05_switch_events/fig06_switch_event_study_nas_seed49.pdf`
   - NAS 的全事件均值接近 0，不能单独支撑“所有 switch 平均都避免损失”。NAS 更适合用 case-window、消融和随机对照讲。

## 数据索引

- case 数字：`metrics/selected_case_windows.csv`
- 全部指标：`metrics/all_metrics.csv`
- 推理消融：`metrics/inference_ablation.csv`
- controller 对齐：`metrics/switch_alignment_summary.csv`
- switch events：`metrics/switch_events.csv`
- random matched-count：`metrics/random_switch_comparison.csv`
