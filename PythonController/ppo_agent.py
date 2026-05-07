import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path


class Policy(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(256, act_dim)
        self.value_head = nn.Linear(256, 1)

    def forward(self, x):
        x = self.net(x)
        logits = self.policy_head(x)
        value = self.value_head(x)
        return logits, value


class PPOAgent:
    def __init__(self, obs_dim, act_dim):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        self.policy = Policy(obs_dim, act_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=3e-4)

        self.gamma = 0.99
        self.lam = 0.95
        self.clip = 0.2
        self.max_grad_norm = 0.5

        self.memory = []

    def save(self, path):
        checkpoint_path = Path(path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "obs_dim": self.obs_dim,
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
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        logits, value = self.policy(state)
        logits = torch.nan_to_num(logits, nan=0.0, posinf=20.0, neginf=-20.0)
        value = torch.nan_to_num(value, nan=0.0, posinf=1e6, neginf=-1e6)

        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()

        return action.item(), dist.log_prob(action).item(), value.item()

    def store(self, transition):
        self.memory.append(transition)

    def compute_gae(self, next_value):
        states, actions, rewards, dones, log_probs, values = zip(*self.memory)

        advantages = []
        gae = 0

        values = list(values) + [next_value]

        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t+1] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.lam * (1 - dones[t]) * gae
            advantages.insert(0, gae)

        returns = [a + v for a, v in zip(advantages, values[:-1])]

        return (
            torch.tensor(np.asarray(states), dtype=torch.float32),
            torch.tensor(actions),
            torch.tensor(log_probs),
            torch.tensor(returns, dtype=torch.float32),
            torch.tensor(advantages, dtype=torch.float32),
        )

    def update(self, next_value):
        if not self.memory:
            return None

        states, actions, old_log_probs, returns, advantages = self.compute_gae(next_value)

        states = states.to(self.device)
        actions = actions.to(self.device)
        old_log_probs = old_log_probs.to(self.device)
        returns = returns.to(self.device)
        advantages = advantages.to(self.device)

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

        loss_total = 0.0
        policy_loss_total = 0.0
        value_loss_total = 0.0
        entropy_total = 0.0
        epoch_count = 0

        for _ in range(4):  # epochs
            logits, values = self.policy(states)
            logits = torch.nan_to_num(logits, nan=0.0, posinf=20.0, neginf=-20.0)
            values = torch.nan_to_num(values, nan=0.0, posinf=1e6, neginf=-1e6)
            dist = torch.distributions.Categorical(logits=logits)

            new_log_probs = dist.log_prob(actions)
            ratio = (new_log_probs - old_log_probs).exp()

            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * advantages

            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = (returns - values.squeeze()).pow(2).mean()

            loss = policy_loss + 0.5 * value_loss

            if not torch.isfinite(loss):
                self.memory.clear()
                return None

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()

            loss_total += loss.item()
            policy_loss_total += policy_loss.item()
            value_loss_total += value_loss.item()
            entropy_total += dist.entropy().mean().item()
            epoch_count += 1

        self.memory.clear()
        return {
            "loss": loss_total / epoch_count,
            "policy_loss": policy_loss_total / epoch_count,
            "value_loss": value_loss_total / epoch_count,
            "entropy": entropy_total / epoch_count,
            "batch_size": int(states.shape[0]),
        }
