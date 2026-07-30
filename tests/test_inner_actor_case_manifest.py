import json
from argparse import Namespace
from pathlib import Path

import pandas as pd
import pytest

from paper_experiments.plot_inner_actor_base_adjustment import (
    ensure_trace,
    load_case_manifest,
    load_prices,
    resolve_case_window,
)


def heatmap_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.to_datetime(
        ["2024-05-13", "2024-05-14", "2024-05-15", "2024-05-16"]
    )
    tilt = pd.DataFrame(
        {
            "IDXX.O": [0.01, 0.02, 0.01, -0.01],
            "ADSK.O": [-0.01, 0.00, 0.02, 0.01],
            "OTHER": [0.00, 0.00, 0.00, 0.00],
        },
        index=dates,
    )
    future = tilt * 2.0
    return tilt, future


def test_resolve_case_window_preserves_exact_dates_and_asset_order() -> None:
    tilt, future = heatmap_frames()
    case = {
        "start_date": "2024-05-14",
        "end_date": "2024-05-16",
        "assets": ["ADSK.O", "IDXX.O"],
    }

    window = resolve_case_window(tilt, future, case)

    assert window["idx"].tolist() == pd.to_datetime(
        ["2024-05-14", "2024-05-15", "2024-05-16"]
    ).tolist()
    assert window["assets"] == ["ADSK.O", "IDXX.O"]
    assert window["start"] == 1
    assert window["end"] == 3


@pytest.mark.parametrize(
    ("case", "message"),
    [
        (
            {
                "start_date": "2024-05-10",
                "end_date": "2024-05-16",
                "assets": ["IDXX.O"],
            },
            "start_date",
        ),
        (
            {
                "start_date": "2024-05-14",
                "end_date": "2024-05-16",
                "assets": ["MISSING"],
            },
            "assets",
        ),
    ],
)
def test_resolve_case_window_rejects_unavailable_inputs(
    case: dict[str, object],
    message: str,
) -> None:
    tilt, future = heatmap_frames()

    with pytest.raises(ValueError, match=message):
        resolve_case_window(tilt, future, case)


def test_load_case_manifest_reads_market_mapping(tmp_path: Path) -> None:
    manifest_path = tmp_path / "cases.json"
    manifest_path.write_text(
        json.dumps(
            {
                "nas": {
                    "start_date": "2024-05-13",
                    "end_date": "2024-06-25",
                    "assets": ["IDXX.O"],
                }
            }
        ),
        encoding="utf-8",
    )

    loaded = load_case_manifest(Namespace(case_manifest=str(manifest_path)))

    assert loaded["nas"]["start_date"] == "2024-05-13"


def test_load_case_manifest_returns_empty_mapping_when_omitted() -> None:
    assert load_case_manifest(Namespace(case_manifest=None)) == {}


def test_ensure_trace_can_read_explicit_actions_root(tmp_path: Path) -> None:
    actions_root = tmp_path / "actions"
    actions_root.mkdir()
    trace = pd.DataFrame(
        {
            "date": ["2024-05-13"],
            "asset_names_json": ['["IDXX.O"]'],
            "base_weights_json": ["[1.0]"],
            "exec_weights_json": ["[1.0]"],
            "inner_tilt_json": ["[0.0]"],
        }
    )
    trace.to_csv(
        actions_root / "nas_seed49_full_controller_inner_base_actions.csv",
        index=False,
    )

    loaded = ensure_trace(
        Namespace(
            actions_root=str(actions_root),
            force_eval=False,
        ),
        tmp_path / "fresh-output",
        "nas",
        49,
    )

    assert loaded["date"].tolist() == ["2024-05-13"]


def test_load_prices_accepts_explicit_data_root(tmp_path: Path) -> None:
    market_dir = tmp_path / "nas"
    market_dir.mkdir()
    pd.DataFrame(
        {
            "date": ["2024-05-13", "2024-05-13"],
            "tic": ["IDXX.O", "OTHER"],
            "adjclose": [100.0, 50.0],
        }
    ).to_csv(market_dir / "nas_data.csv", index=False)

    prices = load_prices("nas", ["IDXX.O"], prices_root=tmp_path)

    assert prices.columns.tolist() == ["IDXX.O"]
    assert prices.iloc[0, 0] == pytest.approx(100.0)
