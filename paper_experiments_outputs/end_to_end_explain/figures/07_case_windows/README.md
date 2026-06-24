# Controller Case Windows 实验说明

## 这个实验是做什么的

该实验选取完整 30 个交易日持仓窗口，展示 controller 在什么市场状态下触发 switch，以及 switch 如何改变后续收益或回撤。它是最适合论文主文展示的可解释性实验。

每个 case 都包含一个关键 switch，并给出“继续持有原组合”和“执行切换”的 20 日反事实路径。

## 图怎么看

每张 case 图从上到下包含四个部分：

1. 窗口收益路径：红线是 Full Controller，黑色虚线是 Fixed HRL，橙色三角表示 controller switch。
2. 窗口回撤路径：比较 Full Controller 和 Fixed HRL 在该 30 日窗口内的 drawdown。
3. Exit probability：观察 switch 是否发生在退出概率抬升或超过 0.5 附近。
4. 关键 switch 后 20 日反事实：灰色虚线是继续持有，黄色线是切换反事实，红线是实际 controller 路径。

读图时应先定位橙色 switch 点，再看 switch 后的反事实曲线。如果继续持有曲线明显走弱，而切换曲线更高或回撤更小，就可以解释为 controller 避免了后续收益劣化。

## 主要结论

SH case 是最清楚的单点 switch 解释案例。2021-06-11 至 2021-07-23 窗口中，关键 switch 发生在 2021-07-07。继续持有反事实未来 20 日收益为 -10.74%，切换反事实为 4.31%，避免损失 15.05 个百分点。该 case 可以直接支持“controller 在下跌风险出现时切换，避免大幅收益劣化”的叙述。

NAS case B 展示一个负向持仓路径被缓解的案例。2021-03-25 至 2021-05-06 窗口中，关键 switch 发生在 2021-04-19。继续持有反事实未来 20 日收益为 -6.05%，切换反事实为 -2.63%，避免损失 3.42 个百分点。

NAS case A 展示多次 switch 共同控制回撤。2020-07-08 至 2020-08-18 窗口中，controller 触发 18 次 free switch；Full Controller 收益为 7.89%，Fixed HRL 为 5.25%；Full Controller 最大回撤为 2.45%，Fixed HRL 为 4.64%。这个 case 更适合解释 NAS 中频繁切换如何压低回撤，而不是强行解释单个 switch。

## 论文中怎么用

建议主文优先放 SH case 和 NAS case B；如果需要说明 NAS 切换频繁但回撤很小，再补 NAS case A。图题和 caption 应强调市场、时间窗口和关键结论，不要把文件名中的 seed 写入论文标题。
