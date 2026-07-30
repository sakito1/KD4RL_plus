import csv
import json
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


APPENDIX_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = APPENDIX_ROOT.parent


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


class AppendixPackageTest(unittest.TestCase):
    def test_required_public_files_are_present(self):
        required = [
            "README.md",
            "ARCHITECTURE_AND_TRAINING.md",
            "CLAIM_BOUNDARIES.md",
            "MODEL_VERSION.json",
            "configs/controller_cases.json",
            "code/analyze_fixed_window_sensitivity.py",
            "code/analyze_transaction_cost.py",
            "code/analyze_controller_statistics.py",
            "code/analyze_trader_statistics.py",
            "code/plot_fixed_window_sensitivity.py",
            "code/plot_controller_cases.py",
            "code/render_statistical_tables.py",
            "code/run_appendix.py",
            "inputs/fixed_window/fixed_window_metrics.csv",
            "inputs/fixed_window/daily_replay_nasdaq100.csv",
            "inputs/fixed_window/daily_replay_csi300.csv",
            "expected/tables/fixed_window_sensitivity.csv",
            "expected/tables/fixed_window_summary.csv",
            "expected/tables/fixed_window_wealth_nasdaq100.csv",
            "expected/tables/fixed_window_wealth_csi300.csv",
            "expected/tables/transaction_cost_sensitivity.csv",
            "expected/tables/controller_decision_validation.csv",
            "expected/tables/trader_refinement_validation.csv",
            "expected/figures/fixed_window_sensitivity_nasdaq100.pdf",
            "expected/figures/fixed_window_sensitivity_nasdaq100.png",
            "expected/figures/fixed_window_sensitivity_csi300.pdf",
            "expected/figures/fixed_window_sensitivity_csi300.png",
            "expected/figures/controller_cases_nas.pdf",
            "expected/figures/controller_cases_sh.pdf",
        ]
        missing = [name for name in required if not (APPENDIX_ROOT / name).is_file()]
        self.assertEqual(missing, [])

    def test_model_version_locks_markets_seeds_fees_and_five_stages(self):
        metadata = json.loads(
            (APPENDIX_ROOT / "MODEL_VERSION.json").read_text(encoding="utf-8")
        )
        self.assertEqual(metadata["markets"]["nas"]["dataset"], "NASDAQ-100")
        self.assertEqual(metadata["markets"]["nas"]["seed"], 49)
        self.assertEqual(metadata["markets"]["sh"]["dataset"], "CSI-300")
        self.assertEqual(metadata["markets"]["sh"]["seed"], 90)
        self.assertEqual(metadata["training"]["transaction_cost_rate"], 0.00005)
        self.assertEqual(metadata["evaluation"]["paper_cost_rate"], 0.0001)
        self.assertEqual(len(metadata["training"]["progressive_stages"]), 5)
        self.assertNotIn("CSI-240", json.dumps(metadata))

    def test_appendix_packages_and_loads_no_neural_network_model(self):
        self.assertEqual(list(APPENDIX_ROOT.rglob("*.pth")), [])
        for script in (APPENDIX_ROOT / "code").glob("*.py"):
            source = script.read_text(encoding="utf-8")
            self.assertNotIn("import torch", source, script.name)
            self.assertNotIn("torch.load", source, script.name)

    def test_parent_package_distributes_only_two_final_models(self):
        models = sorted(
            path.relative_to(PACKAGE_ROOT).as_posix()
            for path in PACKAGE_ROOT.rglob("*.pth")
        )
        self.assertEqual(
            models,
            [
                "checkpoints/csi300/checkpoints/best_model.pth",
                "checkpoints/nasdaq100/checkpoints/best_model.pth",
            ],
        )

    def test_model_dead_code_is_removed_without_dropping_paper_heads(self):
        model_source = (
            PACKAGE_ROOT / "src/Components/PPO_model.py"
        ).read_text(encoding="utf-8")
        for dead_symbol in [
            "class CausalConv1dBlock",
            "last_temporal_attn1",
            "last_temporal_attn2",
            "def pred_return(",
            "def pred_next_return(",
            "tcn_kernel_size",
            "inner_critic_args",
        ]:
            self.assertNotIn(dead_symbol, model_source)
        for required_symbol in [
            "self.pred_head",
            "self.return_head",
            "self.risk_head",
            "self.switch_adv_head",
            "self.fallback_projection",
        ]:
            self.assertIn(required_symbol, model_source)
        training_source = (
            PACKAGE_ROOT / "src/run_hrl_training.py"
        ).read_text(encoding="utf-8")
        self.assertIn("--controller_policy_temperature", training_source)

    def test_public_paths_do_not_expose_seed_numbers(self):
        seed_pattern = re.compile(r"(?:seed[_-]?\d+|nas49|sh90)", re.IGNORECASE)
        offending = [
            path.relative_to(PACKAGE_ROOT).as_posix()
            for path in PACKAGE_ROOT.rglob("*")
            if seed_pattern.search(path.name)
            and "__pycache__" not in path.parts
        ]
        self.assertEqual(offending, [])

    def test_controller_manifest_has_two_hold_and_two_switch_cases(self):
        manifest = json.loads(
            (APPENDIX_ROOT / "configs/controller_cases.json").read_text(
                encoding="utf-8"
            )
        )
        cases = manifest["cases"]
        self.assertEqual(
            {(case["market_key"], case["date"]) for case in cases},
            {
                ("nas", "2025-05-05"),
                ("nas", "2020-07-06"),
                ("sh", "2020-11-25"),
                ("sh", "2021-07-07"),
            },
        )
        self.assertEqual(
            [case["decision"] for case in cases].count("HOLD"),
            2,
        )
        self.assertEqual(
            [case["decision"] for case in cases].count("SWITCH"),
            2,
        )
        expected_probabilities = {
            ("nas", "2025-05-05"): 0.14612801373004913,
            ("nas", "2020-07-06"): 0.5239332914352417,
            ("sh", "2020-11-25"): 0.2319350689649582,
            ("sh", "2021-07-07"): 0.5130767226219177,
        }
        for case in cases:
            key = (case["market_key"], case["date"])
            self.assertAlmostEqual(case["switch_probability"], expected_probabilities[key])

    def test_transaction_cost_headline_values_match_locked_replay(self):
        rows = read_csv(
            APPENDIX_ROOT / "expected/tables/transaction_cost_sensitivity.csv"
        )
        indexed = {
            (row["market"], float(row["cost_pct"])): row
            for row in rows
        }
        self.assertAlmostEqual(
            float(indexed[("NASDAQ-100", 0.01)]["total_return_pct"]),
            262.49,
            places=2,
        )
        self.assertAlmostEqual(
            float(indexed[("CSI-300", 0.01)]["total_return_pct"]),
            237.01,
            places=2,
        )
        self.assertAlmostEqual(
            float(indexed[("NASDAQ-100", 0.05)]["delta_tr_pp"]),
            -26.46,
            places=2,
        )
        self.assertAlmostEqual(
            float(indexed[("CSI-300", 0.05)]["delta_tr_pp"]),
            -27.03,
            places=2,
        )

    def test_fixed_window_table_has_dense_two_market_coverage(self):
        rows = read_csv(
            APPENDIX_ROOT / "expected/tables/fixed_window_sensitivity.csv"
        )
        self.assertEqual(len(rows), 120)
        for market in ("NASDAQ-100", "CSI-300"):
            market_rows = [row for row in rows if row["market"] == market]
            self.assertEqual(
                sorted(int(row["fixed_window_days"]) for row in market_rows),
                list(range(1, 61)),
            )
            self.assertTrue(
                all(
                    float(row["evaluation_cost_pct"]) == 0.01
                    for row in market_rows
                )
            )

    def test_controller_and_trader_headline_tables_match_locked_statistics(self):
        controller = {
            row["market"]: row
            for row in read_csv(
                APPENDIX_ROOT / "expected/tables/controller_decision_validation.csv"
            )
        }
        self.assertEqual(int(controller["NASDAQ-100"]["decisions"]), 1334)
        self.assertAlmostEqual(
            float(controller["NASDAQ-100"]["switch_rate_pct"]), 17.32, places=2
        )
        self.assertAlmostEqual(
            float(controller["CSI-300"]["return_value_bp_day"]), 5.831, places=3
        )
        self.assertAlmostEqual(
            float(controller["CSI-300"]["mdd_value_pp"]), 0.371, places=3
        )

        trader = {
            row["market"]: row
            for row in read_csv(
                APPENDIX_ROOT / "expected/tables/trader_refinement_validation.csv"
            )
        }
        self.assertAlmostEqual(
            float(trader["NASDAQ-100"]["mean_active_share_pct"]), 1.071, places=3
        )
        self.assertAlmostEqual(
            float(trader["CSI-300"]["active_share_gt_1pct_pct"]), 74.3, places=1
        )
        self.assertAlmostEqual(
            float(trader["NASDAQ-100"]["observed_delta_sigma_pp"]),
            -0.0267,
            places=4,
        )
        self.assertAlmostEqual(
            float(trader["CSI-300"]["random_delta_sigma_pp"]),
            0.0054,
            places=4,
        )
        self.assertAlmostEqual(float(trader["CSI-300"]["p_value"]), 0.0002, places=4)

    def test_one_command_renderer_writes_all_public_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            completed = subprocess.run(
                [
                    sys.executable,
                    str(APPENDIX_ROOT / "code/run_appendix.py"),
                    "--output-dir",
                    directory,
                ],
                cwd=PACKAGE_ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            output = Path(directory)
            for relative in [
                "tables/transaction_cost_sensitivity.csv",
                "tables/controller_decision_validation.csv",
                "tables/trader_refinement_validation.csv",
                "tables/fixed_window_sensitivity.csv",
                "tables/fixed_window_summary.csv",
                "tables/fixed_window_wealth_nasdaq100.csv",
                "tables/fixed_window_wealth_csi300.csv",
                "tables/appendix_tables.md",
                "tables/appendix_tables.tex",
                "figures/fixed_window_sensitivity_nasdaq100.pdf",
                "figures/fixed_window_sensitivity_nasdaq100.png",
                "figures/fixed_window_sensitivity_csi300.pdf",
                "figures/fixed_window_sensitivity_csi300.png",
                "figures/controller_cases_nas.pdf",
                "figures/controller_cases_nas.png",
                "figures/controller_cases_sh.pdf",
                "figures/controller_cases_sh.png",
            ]:
                self.assertTrue((output / relative).is_file(), relative)
            for filename in [
                "transaction_cost_sensitivity.csv",
                "controller_decision_validation.csv",
                "trader_refinement_validation.csv",
                "fixed_window_sensitivity.csv",
                "fixed_window_summary.csv",
                "fixed_window_wealth_nasdaq100.csv",
                "fixed_window_wealth_csi300.csv",
            ]:
                self.assertEqual(
                    (output / "tables" / filename).read_bytes(),
                    (APPENDIX_ROOT / "expected/tables" / filename).read_bytes(),
                    filename,
                )
            for filename in [
                "controller_cases_nas.pdf",
                "controller_cases_nas.png",
                "controller_cases_sh.pdf",
                "controller_cases_sh.png",
                "fixed_window_sensitivity_nasdaq100.pdf",
                "fixed_window_sensitivity_nasdaq100.png",
                "fixed_window_sensitivity_csi300.pdf",
                "fixed_window_sensitivity_csi300.png",
            ]:
                self.assertEqual(
                    (output / "figures" / filename).read_bytes(),
                    (APPENDIX_ROOT / "expected/figures" / filename).read_bytes(),
                    filename,
                )

    def test_root_integrity_verifier_accepts_complete_package(self):
        completed = subprocess.run(
            [sys.executable, str(PACKAGE_ROOT / "scripts/verify_package.py")],
            cwd=PACKAGE_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("verified", completed.stdout)


if __name__ == "__main__":
    unittest.main()
