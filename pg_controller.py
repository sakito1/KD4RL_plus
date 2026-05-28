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
                 port_state_dim=6):
        super().__init__()
        self.asset_projection = nn.Sequential(
            nn.Linear(z_dim + h_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        controller_dim = hidden_dim * 4 + port_state_dim + 1 + 3
        self.policy = nn.Sequential(
            nn.Linear(controller_dim, fusion_hidden),
            nn.LayerNorm(fusion_hidden),
            nn.GELU(),
            nn.Linear(fusion_hidden, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2),
        )

    def encode_state(self, z, h, weights, base_weights, switch_weights, port_state,
                     held_p, candidate_costs):
        batch_size = z.shape[0]
        asset_emb = self.asset_projection(torch.cat([z, h], dim=-1))
        emb_live = torch.sum(asset_emb * weights.unsqueeze(-1), dim=1)
        emb_hold = torch.sum(asset_emb * base_weights.unsqueeze(-1), dim=1)
        emb_switch = torch.sum(asset_emb * switch_weights.unsqueeze(-1), dim=1)
        delta_emb = emb_switch - emb_hold
        return torch.cat([
            emb_live,
            emb_hold,
            emb_switch,
            delta_emb,
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
        )
        return self.policy(state)

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
