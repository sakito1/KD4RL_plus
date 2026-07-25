import inspect
import subprocess
import sys

import pytest

import utils.config as config
import utils.config_Nas as config_nas
import utils.config_SH as config_sh
from AlphaStock.Train import (
    alphastock_main,
    apply_transaction_costs,
    rebalance_turnovers,
)
from Train import deep_baseline
from env.PPO_env import PPO_Env


EXPECTED_RATE = 1e-4


def test_all_market_configs_use_one_basis_point():
    assert config.TRANSACTION_COST_RATE == pytest.approx(EXPECTED_RATE)
    assert config_nas.TRANSACTION_COST_RATE == pytest.approx(EXPECTED_RATE)
    assert config_sh.TRANSACTION_COST_RATE == pytest.approx(EXPECTED_RATE)


def test_cmtflow_environment_defaults_to_configured_rate():
    assert inspect.signature(PPO_Env).parameters["transaction_cost_pct"].default is None
    assert "config.TRANSACTION_COST_RATE" in inspect.getsource(PPO_Env.__init__)


def test_deep_baseline_adapters_forward_the_canonical_rate():
    deeparies_source = inspect.getsource(deep_baseline._run_deeparies)
    deeptrader_source = inspect.getsource(deep_baseline._run_deeptrader)
    assert "config.TRANSACTION_COST_RATE" in deeparies_source
    assert 'settings["fee"] = float(config.TRANSACTION_COST_RATE)' in deeptrader_source


def test_alphastock_turnover_cost_is_deducted_from_returns():
    gross_returns = [0.02, -0.01]
    turnovers = [1.0, 0.5]
    net_returns = apply_transaction_costs(
        gross_returns,
        turnovers,
        transaction_cost_rate=EXPECTED_RATE,
    )
    assert net_returns.tolist() == pytest.approx([0.0199, -0.01005])
    assert "apply_transaction_costs" in inspect.getsource(alphastock_main.validate)
    assert "apply_transaction_costs" in inspect.getsource(alphastock_main.generate_estimate)


def test_alphastock_turnover_compares_target_with_drifted_holding():
    weights = [[0.5, 0.5], [0.6, 0.4]]
    prices = [[100.0, 110.0], [100.0, 90.0]]
    turnovers = rebalance_turnovers(weights, prices, decision_offsets=[0, 1])
    assert turnovers.tolist() == pytest.approx([1.0, 0.1])


def test_deeptrader_passes_configured_fee_to_environment():
    from pathlib import Path

    for runner in (
        Path("DeepTrader/src/run.py"),
        Path("DeepTrader/DeepTrader/src/run.py"),
    ):
        assert "fee=func_args.fee" in runner.read_text()


def test_deeparies_runner_help_is_renderable():
    result = subprocess.run(
        [sys.executable, "run_deeparies_baseline.py", "--help"],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--fee_rate" in result.stdout
