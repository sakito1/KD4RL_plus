# Controller Interpretability

这个目录解释 controller 到底在什么情况下 switch。`controller_case_*.png` 是自动筛选出的关键 free switch：第一行固定切点后 30 个交易日，比较“继续旧基础组合（无 controller）”和“切到新基础组合（controller）”的反事实收益；第二行比较同一冻结窗口下的未来回撤；第三行展示切点前后的 exit probability 与反事实切仓优势。这样可以避免真实路径后续多次切仓污染单个 switch 的比较。

`switch_counterfactual_distribution_*.png` 比较所有实际 free switch 点之后 20 日的 switch/hold 反事实收益分布；`switch_remaining_horizon_counterfactual_distribution_*.png` 进一步比较每个实际 switch 在“切仓前组合原本剩余持仓期”内的 switch/hold 冻结反事实收益分布，避免真实路径后续多次切仓污染比较。`fixed_window_comparison_*.png` 比较 learned controller 与 5/10/20/30/60 日固定持仓窗口；`controller_probability_resonance_*.png` 展示 exit probability 是否和未来切仓优势同向变化。

结论文字可概括为：controller 的 switch 在多个 case 中对应即将恶化的持仓；切仓后的冻结反事实路径通常降低回撤或改善未来收益；从所有 switch 的剩余持仓期分布看，switch 候选可以和继续持有旧组合进行同 horizon 比较；相比多个固定持仓窗口，learned controller 在 TR、Sharpe 和 CR 上更稳定，说明收益改善不是来自某个手工固定周期。
