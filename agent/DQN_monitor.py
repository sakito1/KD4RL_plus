import random
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F


class EmbMonitorQNet(nn.Module):
    """Q network for daily hold/switch decisions using pretrained SSM latents."""

    def __init__(self, z_dim=16, h_dim=16, hidden_dim=32):
        super().__init__()
        self.asset_projection = nn.Sequential(
            nn.Linear(z_dim + h_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.q_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, z, h, weights):
        asset_emb = self.asset_projection(torch.cat([z, h], dim=-1))
        portfolio_emb = torch.sum(asset_emb * weights.unsqueeze(-1), dim=1)
        return self.q_head(portfolio_emb)


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
            int(action),
            float(reward),
            replay_tensor(next_obs["ssm"]["z"]),
            replay_tensor(next_obs["ssm"]["h"]),
            replay_tensor(next_obs["weights_drift"]),
            float(done),
        ))

    def sample(self, batch_size, device):
        batch = random.sample(self.data, int(batch_size))
        fields = list(zip(*batch))
        return {
            "z": torch.stack(fields[0]).to(device),
            "h": torch.stack(fields[1]).to(device),
            "weights": torch.stack(fields[2]).to(device),
            "actions": torch.tensor(fields[3], dtype=torch.long, device=device),
            "rewards": torch.tensor(fields[4], dtype=torch.float32, device=device),
            "next_z": torch.stack(fields[5]).to(device),
            "next_h": torch.stack(fields[6]).to(device),
            "next_weights": torch.stack(fields[7]).to(device),
            "dones": torch.tensor(fields[8], dtype=torch.float32, device=device),
        }


class DQNMonitorAgent:
    def __init__(self, device, z_dim=16, h_dim=16, hidden_dim=32, lr=3e-4,
                 gamma=0.99, target_update=500, grad_clip=1.0):
        self.device = device
        self.q_net = EmbMonitorQNet(z_dim, h_dim, hidden_dim).to(device)
        self.target_net = EmbMonitorQNet(z_dim, h_dim, hidden_dim).to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=float(lr))
        self.gamma = float(gamma)
        self.target_update = int(target_update)
        self.grad_clip = float(grad_clip)
        self.update_steps = 0

    def select_action(self, obs, epsilon=0.0):
        if random.random() < float(epsilon):
            return random.randrange(2)
        with torch.inference_mode():
            q = self.q_net(obs["ssm"]["z"], obs["ssm"]["h"], obs["weights_drift"])
        return int(torch.argmax(q, dim=1).item())

    def update(self, replay, batch_size):
        if len(replay) < int(batch_size):
            return None
        batch = replay.sample(batch_size, self.device)
        q_selected = self.q_net(batch["z"], batch["h"], batch["weights"]).gather(
            1, batch["actions"].unsqueeze(1)
        ).squeeze(1)
        # Target participates in the differentiable loss expression; no_grad
        # avoids critic gradients without creating an inference-only tensor.
        with torch.no_grad():
            next_q = self.target_net(
                batch["next_z"], batch["next_h"], batch["next_weights"]
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
        return checkpoint.get("metadata", {})
