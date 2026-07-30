#!/usr/bin/env python3
import csv
from pathlib import Path

root = Path(__file__).resolve().parents[1]
output = root / "outputs"
output.mkdir(exist_ok=True)
for table_name in ("table1", "table2"):
    source = root / "expected" / f"{table_name}.csv"
    rows = list(csv.reader(source.open(encoding="utf-8")))
    header, body = rows[0], rows[1:]
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join("---" for _ in header) + "|",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in body)
    (output / f"{table_name}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(output / f"{table_name}.md")
