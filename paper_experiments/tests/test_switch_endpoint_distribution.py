import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper_experiments.run_paper_experiments_final import compute_switch_remaining_horizon_distribution


def test_switch_remaining_horizon_distribution_freezes_both_decision_candidates():
    actions = pd.DataFrame(
        {
            "step": [10, 11],
            "date": ["2020-01-10", "2020-01-11"],
            "decision_type": ["free_decision", "free_decision"],
            "duration_before_decision": [25, 1],
            "is_switch": [1, 1],
            "is_free_switch": [1, 1],
            "exit_prob": [0.7, 0.8],
            "hold_curve_30": ["[1.0, 0.99, 0.98, 0.97, 0.96, 0.95]", "[1.0, 1.01]"],
            "switch_curve_30": ["[1.0, 1.01, 1.02, 1.03, 1.04, 1.05]", "[1.0, 0.99]"],
        }
    )

    out = compute_switch_remaining_horizon_distribution(actions, max_hold=30, horizon=30)

    first = out[out["step"] == 10].iloc[0]
    assert round(first["remaining_holding_days"], 6) == 5
    assert round(first["counterfactual_hold_return_to_original_end"], 6) == -0.05
    assert round(first["switch_return_to_original_end"], 6) == 0.05
    assert round(first["switch_minus_counterfactual_hold"], 6) == 0.10


if __name__ == "__main__":
    test_switch_remaining_horizon_distribution_freezes_both_decision_candidates()
