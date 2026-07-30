import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from paper_experiments.render_aaai27_figure3 import build_case_inputs


def test_build_case_inputs_uses_manifest_dates_and_selected_case(
    tmp_path: Path,
) -> None:
    selected = pd.DataFrame(
        {
            "date": ["2021-04-19", "2020-06-01"],
            "step": [248, 26],
            "hold_curve_30": [json.dumps([1.0, 0.9, 0.95]), "[]"],
            "switch_curve_30": [json.dumps([1.0, 1.0, 1.1]), "[]"],
            "is_switch": [1, 1],
            "is_free_switch": [1, 1],
        }
    )
    selected_path = tmp_path / "selected.csv"
    selected.to_csv(selected_path, index=False)
    case_spec = {
        "case_id": 1,
        "decision_date": "2021-04-19",
        "plot_start_date": "2021-04-20",
        "plot_end_date": "2021-04-21",
    }

    case, portfolio, actions = build_case_inputs(selected_path, case_spec)

    assert case["date"] == "2021-04-19"
    assert portfolio["date"].tolist() == [
        "2021-04-19",
        "2021-04-20",
        "2021-04-21",
    ]
    assert portfolio["step"].tolist() == [248, 249, 250]
    assert len(actions) == 2


def test_figure3_script_can_run_as_a_standalone_file() -> None:
    project_root = Path(__file__).resolve().parents[1]

    result = subprocess.run(
        [
            sys.executable,
            str(project_root / "paper_experiments/render_aaai27_figure3.py"),
            "--help",
        ],
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "--case_manifest" in result.stdout
