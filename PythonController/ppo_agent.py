import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path

# Tile grid dimensions (must match controller.py)
TILE_H = 16
TILE_W = 32


class Policy(nn.Module):
    def __init__(self, scalar_dim: int, tile_dim: int, act_dim: int):
        super().__init__()
        self.scalar_dim = scalar_dim
        self.tile_dim = tile_dim

        tile_h = TILE_H
        tile_w = tile_dim // tile_h  # 32

        # Two strided convs each halve spatial dims: 16x32 -> 8x16 -> 4x8
        self.tile_cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        cnn_out_dim = 32 * (tile_h // 4) * (tile_w // 4)  # 32*4*8 = 1024

        self.trunk = nn.Sequential(
            nn.Linear(cnn_out_dim + scalar_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(256, act_dim)
        self.value_head = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor):
        single = x.dim() == 1
        if single:
            x = x.unsqueeze(0)

        scalars = x[:, :self.scalar_dim]
        tiles = x[:, self.scalar_dim:].reshape(-1, 1, TILE_H, self.tile_dim // TILE_H)

        cnn_feat = self.tile_cnn(tiles).flatten(1)
        features = self.trunk(torch.cat([scalars, cnn_feat], dim=1))

        logits = self.policy_head(features)
        value = self.value_head(features)

        if single:
            logits = logits.squeeze(0)
            value = value.squeeze(0)

        return logits, value


class PPOAgent:
    def __init__(self, scalar_dim: int, tile_dim: int, act_dim: int):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scalar_dim = scalar_dim
        self.tile_dim = tile_dim
        self.act_dim = act_dim

        self.policy = Policy(scalar_dim, tile_dim, act_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=5e-5)

        self.gamma = 0.99
        self.lam = 0.95
        self.clip = 0.2
        self.max_grad_norm = 0.5
        self.entropy_coef = 0.005
        self.target_kl = 0.01

        self.memory = []

    def save(self, path):
        checkpoint_path = Path(path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "scalar_dim": self.scalar_dim,
                "tile_dim": self.tile_dim,
                "act_dim": self.act_dim,
                "policy_state_dict": self.policy.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            checkpoint_path,
        )

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    def act(self, state):
        try:
            state = torch.tensor(state, dtype=torch.float32).to(self.device)
            logits, value = self.policy(state)
            logits = torch.nan_to_num(logits, nan=0.0, posinf=20.0, neginf=-20.0)
            value = torch.nan_to_num(value, nan=0.0, posinf=1e6, neginf=-1e6)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            return action.item(), dist.log_prob(action).item(), value.item()
        except Exception as exc:
            import traceback
            print(f"ERROR: act() failed — using random action. {exc}")
            traceback.print_exc()
            action = int(torch.randint(0, self.act_dim, (1,)).item())
            return action, 0.0, 0.0

    def store(self, transition):
        self.memory.append(transition)

    def compute_gae(self, next_value):
        states, actions, rewards, dones, log_probs, values = zip(*self.memory)

        advantages = []
        gae = 0

        old_values = list(values)
        values = list(values) + [next_value]

        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.lam * (1 - dones[t]) * gae
            advantages.insert(0, gae)

        returns = [a + v for a, v in zip(advantages, old_values)]

        return (
            torch.tensor(np.asarray(states), dtype=torch.float32),
            torch.tensor(actions),
            torch.tensor(log_probs),
            torch.tensor(returns, dtype=torch.float32),
            torch.tensor(advantages, dtype=torch.float32),
            torch.tensor(old_values, dtype=torch.float32),
        )

    def update(self, next_value):
        try:
            return self._update_impl(next_value)
        except Exception as exc:
            import traceback
            print(f"ERROR: PPO update failed — skipping this batch. {exc}")
            traceback.print_exc()
            self.memory.clear()
            return None

    def _update_impl(self, next_value):
        if len(self.memory) < 64:
            self.memory.clear()
            return None

        states, actions, old_log_probs, returns, advantages, old_values = self.compute_gae(next_value)

        states = states.to(self.device)
        actions = actions.to(self.device)
        old_log_probs = old_log_probs.to(self.device)
        returns = returns.to(self.device)
        advantages = advantages.to(self.device)
        old_values = old_values.to(self.device)

        advantages = torch.nan_to_num(advantages, nan=0.0, posinf=0.0, neginf=0.0)
        adv_mean = advantages.mean()
        adv_std = advantages.std(unbiased=False)
        if torch.isfinite(adv_std) and adv_std.item() > 1e-8:
            advantages = (advantages - adv_mean) / (adv_std + 1e-8)
        else:
            advantages = advantages - adv_mean

        if not (
            torch.isfinite(states).all()
            and torch.isfinite(old_log_probs).all()
            and torch.isfinite(returns).all()
            and torch.isfinite(advantages).all()
        ):
            self.memory.clear()
            return None

        n = states.shape[0]
        batch_size = min(256, n)

        loss_total = 0.0
        policy_loss_total = 0.0
        value_loss_total = 0.0
        entropy_total = 0.0
        steps = 0

        for _ in range(4):  # epochs
            perm = torch.randperm(n, device=self.device)
            early_stop = False
            for start in range(0, n, batch_size):
                idx = perm[start:start + batch_size]
                b_states = states[idx]
                b_actions = actions[idx]
                b_old_log_probs = old_log_probs[idx]
                b_returns = returns[idx]
                b_advantages = advantages[idx]
                b_old_values = old_values[idx]

                logits, values_pred = self.policy(b_states)
                logits = torch.nan_to_num(logits, nan=0.0, posinf=20.0, neginf=-20.0)
                values_pred = torch.nan_to_num(values_pred, nan=0.0, posinf=1e6, neginf=-1e6)
                dist = torch.distributions.Categorical(logits=logits)

                new_log_probs = dist.log_prob(b_actions)

                # KL early stopping
                with torch.no_grad():
                    kl = (b_old_log_probs - new_log_probs).mean()
                if kl > 1.5 * self.target_kl:
                    early_stop = True
                    break

                ratio = (new_log_probs - b_old_log_probs).exp()
                surr1 = ratio * b_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * b_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Clipped value loss
                v = values_pred.squeeze()
                v_clipped = b_old_values + torch.clamp(v - b_old_values, -self.clip, self.clip)
                value_loss = torch.max(
                    (b_returns - v).pow(2),
                    (b_returns - v_clipped).pow(2),
                ).mean()

                entropy = dist.entropy().mean()
                loss = policy_loss + 0.5 * value_loss - self.entropy_coef * entropy

                if not torch.isfinite(loss):
                    continue

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                loss_total += loss.item()
                policy_loss_total += policy_loss.item()
                value_loss_total += value_loss.item()
                entropy_total += entropy.item()
                steps += 1

            if early_stop:
                break

        self.memory.clear()
        if steps == 0:
            return None
        return {
            "loss": loss_total / steps,
            "policy_loss": policy_loss_total / steps,
            "value_loss": value_loss_total / steps,
            "entropy": entropy_total / steps,
            "batch_size": n,
        }
