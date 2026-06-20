import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Train.PPO_train import HRL_Trainer


class _FakeLogger:
    def __init__(self):
        self.messages = []

    def info(self, message, *args):
        if args:
            message = message % args
        self.messages.append(str(message))


class ControllerReportLoggingTests(unittest.TestCase):
    def test_print_report_includes_switch_breakdown(self):
        trainer = object.__new__(HRL_Trainer)
        trainer.logger = _FakeLogger()

        trainer._print_report(
            "diagnostic",
            {
                "switch_count": 7,
                "switch_free_count": 2,
                "forced_hold_count": 11,
                "forced_switch_count": 5,
                "total_steps": 30,
            },
            {
                "total_ret": 0.12,
                "ann_ret": 0.08,
                "ann_vol": 0.10,
                "sharpe": 0.8,
                "max_dd": 0.05,
            },
        )

        report = "\n".join(trainer.logger.messages)
        self.assertIn("Switches   : 7", report)
        self.assertIn("Switch detail: free=2, forced_h=11, forced_s=5", report)


if __name__ == "__main__":
    unittest.main()
