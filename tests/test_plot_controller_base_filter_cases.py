from pathlib import Path

import numpy as np

from scripts.plot_controller_base_filter_cases import (
    build_outputs,
    load_case_data,
)


REPO = Path(__file__).resolve().parents[1]


def test_case_decomposition_and_outcomes_match_saved_traces():
    cases = load_case_data(REPO).set_index("market")

    nas = cases.loc["NASDAQ-100"]
    assert nas["date"] == "2023-01-17"
    assert np.isclose(nas["base_logit"], -1.0211745500564575)
    assert np.isclose(nas["adv_raw"], 0.01094556599855423)
    assert np.isclose(nas["final_probability"], 0.48149167121842945)
    assert nas["action"] == "Hold"
    assert nas["candidate_minus_hold_20"] < 0
    assert nas["candidate_minus_hold_30"] < 0

    csi = cases.loc["CSI-300"]
    assert csi["date"] == "2021-07-07"
    assert np.isclose(csi["base_logit"], -0.990081250667572)
    assert np.isclose(csi["adv_raw"], 0.012328431010246277)
    assert np.isclose(csi["final_probability"], 0.5130767108364654)
    assert csi["action"] == "Switch"
    assert np.isclose(csi["neutral_candidate_probability"], 0.23552373)
    assert csi["neutral_candidate_action"] == "Hold"
    assert csi["candidate_minus_hold_20"] > 0
    assert csi["candidate_minus_hold_30"] > 0

    assert (cases["adv_raw"] > 0).all()
    assert (cases["adv_only_probability"] > 0.5).all()
    assert (cases["formula_error"] < 5e-7).all()


def test_build_outputs_writes_figure_table_and_report(tmp_path):
    (tmp_path / "controller_base_filter_population.csv").write_text(
        "legacy population output", encoding="utf-8"
    )
    build_outputs(REPO, tmp_path)

    expected = [
        "controller_base_filter_cases.png",
        "controller_base_filter_cases.pdf",
        "controller_base_filter_cases.csv",
        "CONTROLLER_BASE_FILTER_CASES_CN.md",
    ]
    for name in expected:
        path = tmp_path / name
        assert path.is_file()
        assert path.stat().st_size > 0

    assert not (tmp_path / "controller_base_filter_population.csv").exists()
    report = (tmp_path / "CONTROLLER_BASE_FILTER_CASES_CN.md").read_text(
        encoding="utf-8"
    )
    assert "全测试集机制证据" not in report
    assert "HAC" not in report
    assert "bootstrap" not in report
    assert "NASDAQ-100：Hold更好" in report
    assert "CSI-300：Switch更好" in report
    assert "不能证明学习型Base head不可替代" in report
