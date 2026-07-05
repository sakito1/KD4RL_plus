# Manifest

`baseline_sources.csv` 是本目录的索引表。核心字段：

- `curve_status`: `available` 表示可以用于收益曲线，其他状态只用于指标或溯源。
- `table_*`: 论文表格中的指标。
- `recomputed_*`: 从曲线、日志或指标文件重算/读取的对应值。
- `source_path` / `log_snippet_path`: 原始结果和关键日志位置。
