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
