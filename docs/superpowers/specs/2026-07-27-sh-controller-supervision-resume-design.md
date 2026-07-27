# SH Controller 强监督续训脚本设计

## 目标

在 SH seed 44、46 已训练完成的 Outer+Inner checkpoint 上，仅续训
Controller。Outer 与 Inner 全程冻结。Controller 训练及监督目标保持30日，
仅测试时把强制切换上限放宽到60日。PG 阶段的 switch BCE 系数从
0.01 提高到 0.10。

## 执行方式

- seed 44、46 默认分别使用 GPU 0、1，并行执行且每张 GPU 仅运行一个任务。
- 每个 seed 使用独立的源 checkpoint、输出目录和日志。
- 默认加载 `hrl_fixed_best.pth`，不从包含旧 Controller 的
  `best_model.pth` 继续。
- Controller 的预训练系数保持 1.0，Risk 与 Advantage 的 PG 辅助系数
  保持 0.01；只调整 `controller_sup_coef=0.10`。
- Switch 标签采用
  \((Risk\ge10\%\land Advantage>2\%)\lor Advantage\ge10\%\)。
- Controller 训练、状态期限和监督目标保持30日，测试强制期限为60日。
- 固定 30 日 Outer+Inner 指标沿用源实验，不在这次 60 日 Controller
  训练环境中重新定义。

## 可操作性

源 checkpoint 路径、GPU、训练轮数和输出目录均允许通过环境变量覆盖。
脚本启动前检查 Python 和两个 checkpoint；任一缺失即停止，不产生半套
实验。`DRY_RUN=1` 只打印两条完整命令。

## 验证

测试检查两个任务的 seed、checkpoint、冻结模式、60 日期限、
`controller_sup_coef=0.10`、GPU 分配和最终测试开关。
