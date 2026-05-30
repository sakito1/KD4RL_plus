import torch
import torch.nn as nn
from torch.distributions import Categorical


class PGControllerNet(nn.Module):
    """Policy-only controller for hold/switch decisions.

    The controller compares the live, hold, and switch portfolio embeddings
    plus portfolio-state and cost features. It intentionally has no value head:
    training uses episode-level policy gradient on realized Sharpe.
    """

    def __init__(self, z_dim=16, h_dim=16, hidden_dim=32, fusion_hidden=64,
                 port_state_dim=6, use_risk_gate=True, risk_gate_floor=0.05,
                 risk_gate_prior_scale=1.0, embedding_mode="full",
                 initial_hold_bias=0.0, zero_policy_output=False):
        super().__init__()
        self.use_risk_gate = bool(use_risk_gate)
        self.risk_gate_floor = float(risk_gate_floor)
        self.risk_gate_prior_scale = float(risk_gate_prior_scale)
        self.embedding_mode = str(embedding_mode)
        emb_multipliers = {
            "full": 4,
            "hold_delta": 2,
            "live_delta": 2,
            "delta": 1,
        }
        if self.embedding_mode not in emb_multipliers:
            raise ValueError(f"Unknown embedding_mode: {self.embedding_mode}")
        # Old SSM exports use z/h as 16+16 dims, while RiskTPSM-Lite can
        # expose 32+32 through the same compatibility fields. LazyLinear keeps
        # the controller input-compatible with both without changing the pooled
        # controller state size.
        self.asset_projection = nn.Sequential(
            nn.LazyLinear(hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        if self.use_risk_gate:
            self.risk_gate_net = nn.Sequential(
                nn.Linear(hidden_dim + 8, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1),
            )
            nn.init.zeros_(self.risk_gate_net[-1].weight)
            nn.init.zeros_(self.risk_gate_net[-1].bias)
        controller_dim = hidden_dim * emb_multipliers[self.embedding_mode] + port_state_dim + 1 + 3
        self.policy = nn.Sequential(
            nn.Linear(controller_dim, fusion_hidden),
            nn.LayerNorm(fusion_hidden),
            nn.GELU(),
            nn.Linear(fusion_hidden, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2),
        )
        if zero_policy_output:
            nn.init.zeros_(self.policy[-1].weight)
        if initial_hold_bias:
            with torch.no_grad():
                self.policy[-1].bias.copy_(
                    torch.tensor(
                        [float(initial_hold_bias), -float(initial_hold_bias)],
                        dtype=self.policy[-1].bias.dtype,
                    )
                )

    @staticmethod
    def _optional_asset_feature(value, like):
        if value is None:
            return torch.zeros_like(like)
        return value.reshape(like.shape)

    def _risk_gate(self, asset_emb, weights, base_weights, switch_weights,
                   ssm_p=None, q_bear=None, q_bull=None):
        if not self.use_risk_gate:
            return torch.ones_like(weights)
        ssm_p = self._optional_asset_feature(ssm_p, weights)
        q_bear = self._optional_asset_feature(q_bear, weights)
        q_bull = self._optional_asset_feature(q_bull, weights)
        risk_features = torch.stack([
            weights,
            base_weights,
            switch_weights,
            torch.abs(switch_weights - weights),
            switch_weights - base_weights,
            ssm_p,
            q_bear,
            q_bull,
        ], dim=-1)
        learned_logits = self.risk_gate_net(
            torch.cat([asset_emb, risk_features], dim=-1)
        ).squeeze(-1)
        risk_prior = q_bear - q_bull + 0.5 * torch.abs(switch_weights - weights)
        return torch.sigmoid(learned_logits + self.risk_gate_prior_scale * risk_prior)

    def _gated_portfolio_weights(self, weights, gate):
        if not self.use_risk_gate:
            return weights
        adjusted = weights * (self.risk_gate_floor + gate)
        return adjusted / (adjusted.sum(dim=1, keepdim=True) + 1e-8)

    def encode_state(self, z, h, weights, base_weights, switch_weights, port_state,
                     held_p, candidate_costs, ssm_p=None, q_bear=None, q_bull=None):
        batch_size = z.shape[0]
        asset_emb = self.asset_projection(torch.cat([z, h], dim=-1))
        gate = self._risk_gate(
            asset_emb, weights, base_weights, switch_weights,
            ssm_p=ssm_p, q_bear=q_bear, q_bull=q_bull,
        )
        live_weights = self._gated_portfolio_weights(weights, gate)
        hold_weights = self._gated_portfolio_weights(base_weights, gate)
        switch_weights_gated = self._gated_portfolio_weights(switch_weights, gate)
        emb_live = torch.sum(asset_emb * live_weights.unsqueeze(-1), dim=1)
        emb_hold = torch.sum(asset_emb * hold_weights.unsqueeze(-1), dim=1)
        emb_switch = torch.sum(asset_emb * switch_weights_gated.unsqueeze(-1), dim=1)
        delta_emb = emb_switch - emb_hold
        if self.embedding_mode == "full":
            emb_features = [emb_live, emb_hold, emb_switch, delta_emb]
        elif self.embedding_mode == "hold_delta":
            emb_features = [emb_hold, delta_emb]
        elif self.embedding_mode == "live_delta":
            emb_features = [emb_live, delta_emb]
        else:
            emb_features = [delta_emb]
        return torch.cat([
            *emb_features,
            port_state.reshape(batch_size, -1),
            held_p.reshape(batch_size, -1),
            candidate_costs.reshape(batch_size, -1),
        ], dim=1)

    def forward(self, obs):
        state = self.encode_state(
            obs["ssm"]["z"],
            obs["ssm"]["h"],
            obs["weights_drift"],
            obs["base_drift"],
            obs["candidate_switch_base"],
            obs["port_state"],
            obs["held_p"],
            obs["candidate_costs"],
            ssm_p=obs["ssm"].get("p"),
            q_bear=obs["ssm"].get("q_bear"),
            q_bull=obs["ssm"].get("q_bull"),
        )
        return self.policy(state)

    def gate_diagnostics(self, obs):
        weights = obs["weights_drift"]
        if not self.use_risk_gate:
            zeros = torch.zeros(weights.shape[0], dtype=weights.dtype, device=weights.device)
            return {
                "gate_mean": zeros,
                "gate_std": zeros,
                "gate_max": zeros,
                "gate_live": zeros,
                "gate_switch_minus_hold": zeros,
                "gate_top5_mass": zeros,
            }
        z, h = obs["ssm"]["z"], obs["ssm"]["h"]
        asset_emb = self.asset_projection(torch.cat([z, h], dim=-1))
        gate = self._risk_gate(
            asset_emb,
            weights,
            obs["base_drift"],
            obs["candidate_switch_base"],
            ssm_p=obs["ssm"].get("p"),
            q_bear=obs["ssm"].get("q_bear"),
            q_bull=obs["ssm"].get("q_bull"),
        )
        top_k = min(5, gate.shape[1])
        gate_sum = gate.sum(dim=1) + 1e-8
        return {
            "gate_mean": gate.mean(dim=1),
            "gate_std": gate.std(dim=1, unbiased=False),
            "gate_max": gate.max(dim=1).values,
            "gate_live": torch.sum(weights * gate, dim=1),
            "gate_switch_minus_hold": torch.sum(
                (obs["candidate_switch_base"] - obs["base_drift"]) * gate, dim=1
            ),
            "gate_top5_mass": torch.topk(gate, k=top_k, dim=1).values.sum(dim=1) / gate_sum,
        }

    def act(self, obs, deterministic=False):
        logits = self.forward(obs)
        dist = Categorical(logits=logits)
        action = torch.argmax(logits, dim=-1) if deterministic else dist.sample()
        return action, dist.log_prob(action), dist.entropy()


class RunningObjectiveBaseline:
    """Scalar moving average baseline, not a learned value function."""

    def __init__(self, momentum=0.9):
        self.momentum = float(momentum)
        self.value = None

    def advantage(self, objective):
        objective = float(objective)
        baseline = objective if self.value is None else self.value
        advantage = objective - baseline
        if self.value is None:
            self.value = objective
        else:
            self.value = self.momentum * self.value + (1.0 - self.momentum) * objective
        return advantage
