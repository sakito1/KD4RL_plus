import random
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F


class EmbMonitorQNet(nn.Module):
    """Q network for hold/switch decisions using both candidate portfolios."""

    def __init__(self, z_dim=16, h_dim=16, hidden_dim=32, fusion_hidden=64,
                 port_state_dim=6):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.asset_projection = nn.Sequential(
            nn.LazyLinear(hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        controller_dim = hidden_dim * 4 + port_state_dim + 1 + 3
        self.q_head = nn.Sequential(
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
            emb_live, emb_hold, emb_switch, delta_emb,
            port_state.reshape(batch_size, -1),
            held_p.reshape(batch_size, -1),
            candidate_costs.reshape(batch_size, -1),
        ], dim=1)

    def forward(self, z, h, weights, base_weights, switch_weights, port_state,
                held_p, candidate_costs):
        controller_state = self.encode_state(
            z, h, weights, base_weights, switch_weights, port_state,
            held_p, candidate_costs,
        )
        return self.q_head(controller_state)


class DQNReplayBuffer:
    def __init__(self, capacity):
        self.data = deque(maxlen=int(capacity))

    def __len__(self):
        return len(self.data)

    def store(self, obs, action, reward, next_obs, done):
        def replay_tensor(value):
            # Environment states may originate inside inference_mode while
            # frozen HRL policies execute. Clone outside it for DQN autograd.
            return value.squeeze(0).detach().clone().cpu()

        self.data.append((
            replay_tensor(obs["ssm"]["z"]),
            replay_tensor(obs["ssm"]["h"]),
            replay_tensor(obs["weights_drift"]),
            replay_tensor(obs["base_drift"]),
            replay_tensor(obs["candidate_switch_base"]),
            replay_tensor(obs["port_state"]),
            replay_tensor(obs["held_p"]),
            replay_tensor(obs["candidate_costs"]),
            int(action),
            float(reward),
            replay_tensor(next_obs["ssm"]["z"]),
            replay_tensor(next_obs["ssm"]["h"]),
            replay_tensor(next_obs["weights_drift"]),
            replay_tensor(next_obs["base_drift"]),
            replay_tensor(next_obs["candidate_switch_base"]),
            replay_tensor(next_obs["port_state"]),
            replay_tensor(next_obs["held_p"]),
            replay_tensor(next_obs["candidate_costs"]),
            float(done),
        ))

    def sample(self, batch_size, device):
        batch = random.sample(self.data, int(batch_size))
        fields = list(zip(*batch))
        return {
            "z": torch.stack(fields[0]).to(device),
            "h": torch.stack(fields[1]).to(device),
            "weights": torch.stack(fields[2]).to(device),
            "base_weights": torch.stack(fields[3]).to(device),
            "switch_weights": torch.stack(fields[4]).to(device),
            "port_state": torch.stack(fields[5]).to(device),
            "held_p": torch.stack(fields[6]).to(device),
            "candidate_costs": torch.stack(fields[7]).to(device),
            "actions": torch.tensor(fields[8], dtype=torch.long, device=device),
            "rewards": torch.tensor(fields[9], dtype=torch.float32, device=device),
            "next_z": torch.stack(fields[10]).to(device),
            "next_h": torch.stack(fields[11]).to(device),
            "next_weights": torch.stack(fields[12]).to(device),
            "next_base_weights": torch.stack(fields[13]).to(device),
            "next_switch_weights": torch.stack(fields[14]).to(device),
            "next_port_state": torch.stack(fields[15]).to(device),
            "next_held_p": torch.stack(fields[16]).to(device),
            "next_candidate_costs": torch.stack(fields[17]).to(device),
            "dones": torch.tensor(fields[18], dtype=torch.float32, device=device),
        }


class DQNMonitorAgent:
    def __init__(self, device, z_dim=16, h_dim=16, hidden_dim=32, fusion_hidden=64,
                 lr=3e-4, gamma=0.99, target_update=500, grad_clip=1.0):
        self.device = device
        self.q_net = EmbMonitorQNet(z_dim, h_dim, hidden_dim, fusion_hidden).to(device)
        self.target_net = EmbMonitorQNet(z_dim, h_dim, hidden_dim, fusion_hidden).to(device)
        self.target_net.eval()
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=float(lr))
        self.gamma = float(gamma)
        self.target_update = int(target_update)
        self.grad_clip = float(grad_clip)
        self.update_steps = 0
        self._target_initialized = False

    def _sync_target_once_initialized(self):
        if self._target_initialized:
            return
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()
        self._target_initialized = True

    def select_action(self, obs, epsilon=0.0):
        if random.random() < float(epsilon):
            return random.randrange(2)
        with torch.inference_mode():
            q = self.q_net(
                obs["ssm"]["z"], obs["ssm"]["h"], obs["weights_drift"],
                obs["base_drift"], obs["candidate_switch_base"], obs["port_state"],
                obs["held_p"], obs["candidate_costs"],
            )
            self._sync_target_once_initialized()
        return int(torch.argmax(q, dim=1).item())

    def update(self, replay, batch_size):
        if len(replay) < int(batch_size):
            return None
        batch = replay.sample(batch_size, self.device)
        q_selected = self.q_net(
            batch["z"], batch["h"], batch["weights"], batch["base_weights"],
            batch["switch_weights"], batch["port_state"], batch["held_p"],
            batch["candidate_costs"],
        ).gather(
            1, batch["actions"].unsqueeze(1)
        ).squeeze(1)
        self._sync_target_once_initialized()
        # Target participates in the differentiable loss expression; no_grad
        # avoids critic gradients without creating an inference-only tensor.
        with torch.no_grad():
            next_q = self.target_net(
                batch["next_z"], batch["next_h"], batch["next_weights"],
                batch["next_base_weights"], batch["next_switch_weights"],
                batch["next_port_state"], batch["next_held_p"],
                batch["next_candidate_costs"],
            ).max(dim=1).values
            target = batch["rewards"] + self.gamma * (1.0 - batch["dones"]) * next_q
        loss = F.smooth_l1_loss(q_selected, target)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), self.grad_clip)
        self.optimizer.step()
        self.update_steps += 1
        if self.update_steps % self.target_update == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())
        return float(loss.item())

    def save(self, path, metadata=None):
        torch.save({
            "q_net": self.q_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "update_steps": self.update_steps,
            "metadata": metadata or {},
        }, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(checkpoint["q_net"])
        self.target_net.load_state_dict(checkpoint.get("target_net", checkpoint["q_net"]))
        self.update_steps = int(checkpoint.get("update_steps", 0))
        self._target_initialized = True
        return checkpoint.get("metadata", {})


class ConditionalUtilityCritic(nn.Module):
    """Estimate the net one-day utilities of the hold and switch candidates."""

    def __init__(self, z_dim=16, h_dim=16, hidden_dim=32, fusion_hidden=64,
                 port_state_dim=6):
        super().__init__()
        self.encoder = EmbMonitorQNet(
            z_dim=z_dim, h_dim=h_dim, hidden_dim=hidden_dim,
            fusion_hidden=fusion_hidden, port_state_dim=port_state_dim,
        )
        controller_dim = hidden_dim * 4 + port_state_dim + 1 + 3
        self.utility_head = nn.Sequential(
            nn.Linear(controller_dim, fusion_hidden),
            nn.LayerNorm(fusion_hidden),
            nn.GELU(),
            nn.Linear(fusion_hidden, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, z, h, weights, base_weights, switch_weights, port_state,
                held_p, candidate_costs):
        controller_state = self.encoder.encode_state(
            z, h, weights, base_weights, switch_weights, port_state,
            held_p, candidate_costs,
        )
        return self.utility_head(controller_state)


class UtilityReplayBuffer:
    def __init__(self, capacity):
        self.data = deque(maxlen=int(capacity))

    def __len__(self):
        return len(self.data)

    def store(self, obs, targets):
        def stored(value):
            return value.squeeze(0).detach().clone().cpu()

        self.data.append((
            stored(obs["ssm"]["z"]),
            stored(obs["ssm"]["h"]),
            stored(obs["weights_drift"]),
            stored(obs["base_drift"]),
            stored(obs["candidate_switch_base"]),
            stored(obs["port_state"]),
            stored(obs["held_p"]),
            stored(obs["candidate_costs"]),
            stored(targets),
        ))

    def sample(self, batch_size, device):
        fields = list(zip(*random.sample(self.data, int(batch_size))))
        return [torch.stack(field).to(device) for field in fields]


class ConditionalUtilityAgent:
    def __init__(self, device, hidden_dim=32, fusion_hidden=64, lr=3e-4,
                 grad_clip=1.0):
        self.device = device
        self.net = ConditionalUtilityCritic(
            hidden_dim=hidden_dim, fusion_hidden=fusion_hidden
        ).to(device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=float(lr))
        self.grad_clip = float(grad_clip)

    def predict(self, obs):
        return self.net(
            obs["ssm"]["z"], obs["ssm"]["h"], obs["weights_drift"],
            obs["base_drift"], obs["candidate_switch_base"], obs["port_state"],
            obs["held_p"], obs["candidate_costs"],
        )

    def select_action(self, obs, epsilon=0.0):
        if random.random() < float(epsilon):
            return random.randrange(2)
        with torch.inference_mode():
            utilities = self.predict(obs)
        return int(torch.argmax(utilities, dim=1).item())

    def update(self, replay, batch_size):
        if len(replay) < int(batch_size):
            return None
        batch = replay.sample(batch_size, self.device)
        predicted = self.net(*batch[:8])
        loss = F.smooth_l1_loss(predicted, batch[8])
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.grad_clip)
        self.optimizer.step()
        return float(loss.item())

    def save(self, path, metadata=None):
        torch.save({
            "utility_net": self.net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "metadata": metadata or {},
        }, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.net.load_state_dict(checkpoint["utility_net"])
        return checkpoint.get("metadata", {})
