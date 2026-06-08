import os
import sys
import unittest
from types import SimpleNamespace

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from Train.PPO_train import HRL_Networks


class InnerActorArchitectureTests(unittest.TestCase):
    def test_inner_actor_does_not_enable_feature_gate(self):
        cfg = SimpleNamespace(
            dataset={"features_name": ["adjopen", "adjhigh", "adjlow", "adjclose", "amount", "amp", "body"]},
            trade_num=3,
            min_hold=10,
            max_hold=40,
            inner_max_boundary=0.6,
            inner_feature_gate=True,
        )

        networks = HRL_Networks(ssm_dim=16, num_stocks=5, cfg=cfg)

        self.assertFalse(hasattr(networks.inner, "feature_gate"))
        self.assertFalse(any("feature_gate" in name for name in networks.inner.state_dict()))


if __name__ == "__main__":
    unittest.main()
