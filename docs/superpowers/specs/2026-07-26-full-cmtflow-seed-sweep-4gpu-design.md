# 完整 CMTFlow 多种子四 GPU 训练设计

## 目标

从随机初始化开始，对以下市场—种子完成完整训练和测试：

- NAS：44、45、47、50、56、57、58；
- SH：44、46、49、54。

使用四张 GPU，每张 GPU 同时运行两个独立种子进程，总并发上限为八。

## 单任务训练流程

每个市场—种子由同一个 `run_hrl_training.py` 进程连续执行：

1. Outer Actor：4 轮；
2. Inner Actor：2 轮；
3. Outer+Inner joint：1 轮；
4. Controller 监督预训练：3 轮，每批轨迹 replay 30 次；
5. Controller counterfactual PG：5 轮；
6. 加载验证集最优最终模型并执行测试集评估。

除阶段轮数改为 4/2/1 外，Outer/Inner 的其余参数沿用当前
`outer_inner_4_3_2_k5` 配置。Controller 参数沿用当前
`train_controller_from_outer_inner.sh` 配置，包括 300 日窗口、12 个固定训练
窗口、Risk/Advantage target scale 20，以及 PG 阶段 Risk、Advantage、Label、
Rate 系数均为 0.01。

Controller 阶段冻结已训练好的 Outer 和 Inner；不执行 Controller 之后的全模块
联合微调。

## Checkpoint

每个种子目录保留：

```text
temp_warmup_outer.pth
temp_warmup_inner.pth
hrl_fixed_best.pth
best_model.pth
last_model.pth
```

其中 `hrl_fixed_best.pth` 是完成 Outer+Inner joint 后、开始 Controller 前的模型，
可直接用于消融或单独重训 Controller；`best_model.pth` 是验证集选出的最终模型。

## 四 GPU 调度

调度器建立八条 lane：

- GPU 0：lane 0、lane 1；
- GPU 1：lane 0、lane 1；
- GPU 2：lane 0、lane 1；
- GPU 3：lane 0、lane 1。

11 个任务轮询分配到八条 lane。八条 lane 并行，每条 lane 内任务串行，因此每张
GPU 同时最多两个进程。市场种子、GPU 编号和每卡并发数均允许通过环境变量覆盖。

## 输出与失败处理

每个任务具有独立运行目录和 scheduler log。任务完成后，调度器从日志提取
Scenario 1--3 的 Total Ret、Ann Ret、Ann Vol、Sharpe、Max DD、Switches 以及
Controller 概率统计，写入统一 `test_results_summary.txt`。

单任务失败不会阻止同 lane 后续任务；全部 lane 结束后，只要存在失败任务，
调度器返回非零状态。`DRY_RUN=1` 只打印阶段参数、任务分配和命令，不启动训练。

## 验证要求

自动测试需要证明：

1. 单任务命令从零开始，不加载 frozen checkpoint；
2. 阶段轮数严格为 4/2/1/3/5；
3. Controller 使用当前监督预训练和 PG 参数；
4. 不包含 `--no_train_controller`、`--skip_test` 或 Controller 后联合微调；
5. 默认生成指定 11 个任务及四张 GPU 的八条 lane；
6. shell 语法和完整 dry-run 通过。
