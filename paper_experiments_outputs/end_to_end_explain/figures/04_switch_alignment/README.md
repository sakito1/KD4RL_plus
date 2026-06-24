# Switch Alignment 实验说明

## 这个实验是做什么的

该实验检查 controller 输出的 exit probability 是否与 switch 的潜在收益方向一致。它用于回答：controller 的 switch 决策是否只是随机触发，还是至少在概率层面区分了“更倾向切换”和“更倾向继续持有”的状态。

## 图怎么看

- `fig05_exit_prob_calibration_*.pdf`：横轴是 exit probability 分箱，纵轴是平均 switch advantage。若高概率区间的平均 advantage 更高，说明 controller 的概率输出具有一定校准意义。
- `fig05b_switch_advantage_switched_vs_held_*.pdf`：比较实际 held 与 switched 两类 free decision 的 switch advantage 分布。Switched 组整体更高时，说明 controller 倾向在更有利的时点切换。

这里的 switch advantage 是局部反事实指标，反映“切换相对继续持有”的短期优势。它会有噪声，不应被解读为每一次 switch 都必然赚钱。

## 主要结论

controller 的概率输出能区分 switch 与 hold：NAS 中 switch 日的平均 exit probability 为 0.589，hold 日为 0.249；SH 中 switch 日为 0.531，hold 日为 0.325。这说明 controller 的动作不是无信息触发。

但全局线性相关较弱：NAS 的 exit probability 与 switch advantage 相关性约为 0.016，SH 约为 0.066。因此这组图更适合说明“controller 学到了 switch/hold 的概率分离”，不适合单独支撑“exit probability 完全校准到未来收益优势”。

## 论文中怎么用

建议作为附录或辅助图。主文中若使用，应配合 case window 图说明具体 switch 时点的反事实收益。
