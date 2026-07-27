# Controller 60 日强制切仓批量测试设计

## 目标

对 `results/full_cmtflow_seed_sweep_4gpu` 下所有已完成的 CMTFlow
种子执行纯测试，将 Controller 在测试阶段的强制切仓上限从 30
个交易日改为 60 个交易日，不重新训练且不覆盖原始结果。

## 方案

新增 `train_sh/test_full_cmtflow_controller_maxhold60.sh`。脚本自动扫描
训练时保存的 `seed_*_command.json`，并仅处理同时存在对应
`best_model.pth` 的运行。它复用命令文件中的完整模型配置，保留全局
`max_hold=30`，仅覆盖 `controller_eval_max_hold=60`，再通过
`test_only_checkpoint` 进入纯测试路径。

所有任务默认在 GPU 0--3 上运行四个并行队列，每张 GPU 的队列内部
串行执行，保证每张卡同一时间只有一个测试任务。测试输出写入独立目录
`results/full_cmtflow_test_controller_maxhold60`。缺失 checkpoint 的运行
会被跳过；单个测试失败不会阻止其余种子继续测试，脚本最终生成汇总并以
非零状态报告是否存在失败。

## 可配置项

- `SOURCE_ROOT`：训练结果根目录。
- `OUTPUT_ROOT`：60 日测试结果根目录。
- `GPU_IDS`：测试使用的 GPU 列表，默认 `0 1 2 3`。
- `GPU_ID`：兼容单 GPU 调用；仅在未设置 `GPU_IDS` 时生效。
- `PYTHON_BIN`：Python 解释器，默认取当前环境。
- `DRY_RUN=1`：只打印将执行的命令。

## 验证

自动化测试使用临时的结果目录、命令 JSON 和 checkpoint，验证：

1. 只识别已完成运行；
2. 使用 `test_only_checkpoint`；
3. 保留 `max_hold=30`；
4. 将 `controller_eval_max_hold` 改为 `60`；
5. 输出路径与原训练目录隔离。
