# Paper Experiments Root Entry Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use $superpower-subagents (recommended) or $superpower-executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking via update_plan.

**Goal:** Make the repository-root `run_paper_experiments_final.py` delegate to the canonical package implementation so all default paths resolve inside the repository.

**Architecture:** The root file becomes a thin command-line adapter. All parsing, model loading, experiment execution, and plotting remain in `paper_experiments.run_paper_experiments_final`, eliminating duplicated implementation and path drift.

**Tech Stack:** Python 3.10, pytest, `importlib`

---

### Task 1: Replace the duplicated root implementation with a tested wrapper

**Files:**
- Create: `paper_experiments/tests/test_root_paper_experiments_entry.py`
- Modify: `run_paper_experiments_final.py`

- [ ] **Step 1: Write the failing delegation test**

```python
import importlib.util
from pathlib import Path

from paper_experiments import run_paper_experiments_final as canonical


def test_root_entry_delegates_to_canonical_main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    entry_path = repo_root / "run_paper_experiments_final.py"
    spec = importlib.util.spec_from_file_location("root_paper_experiments_entry", entry_path)
    assert spec is not None and spec.loader is not None
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)

    assert entry.main is canonical.main
```

- [ ] **Step 2: Run the test and verify the duplicated entry fails**

Run:

```bash
/home/tongwenxuan/conda/envs/xuangu/bin/python -m pytest \
  paper_experiments/tests/test_root_paper_experiments_entry.py -q
```

Expected: one failed assertion because the root file currently defines a separate `main`.

- [ ] **Step 3: Replace the root file with the minimal wrapper**

```python
#!/usr/bin/env python3
"""Run the canonical final paper experiment entry point."""

from paper_experiments.run_paper_experiments_final import main


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run focused and existing paper experiment tests**

Run:

```bash
/home/tongwenxuan/conda/envs/xuangu/bin/python -m pytest \
  paper_experiments/tests/test_root_paper_experiments_entry.py \
  paper_experiments/tests/test_final_formatting.py \
  paper_experiments/tests/test_switch_endpoint_distribution.py -q
```

Expected: `3 passed`.

- [ ] **Step 5: Verify the root command with validated cached inputs**

Run:

```bash
/home/tongwenxuan/conda/envs/xuangu/bin/python \
  run_paper_experiments_final.py \
  --markets sh nas \
  --seeds sh:90 nas:49 \
  --device cuda \
  --skip_fixed_eval
```

Expected: exit code 0 and summaries reporting 20 main metric rows, 8 ablation metric rows, 4 controller cases, and 2 inner summary rows.

- [ ] **Step 6: Commit only the wrapper and its regression test**

```bash
git add run_paper_experiments_final.py \
  paper_experiments/tests/test_root_paper_experiments_entry.py
git commit -m "fix: delegate paper experiment root entry"
```

## Verification

- The regression test proves the root module exposes the canonical `main`.
- Existing formatting and controller distribution tests remain green.
- The real root command completes against SH90/NAS49 cached inputs.
