import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class Policy(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
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

        self.policy = Policy(obs_dim, act_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=3e-4)

        self.gamma = 0.99
        self.lam = 0.95
        self.clip = 0.2

        self.memory = []

    def act(self, state):
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        logits, value = self.policy(state)

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
            torch.tensor(states, dtype=torch.float32),
            torch.tensor(actions),
            torch.tensor(log_probs),
            torch.tensor(returns, dtype=torch.float32),
            torch.tensor(advantages, dtype=torch.float32),
        )

    def update(self, next_value):
        states, actions, old_log_probs, returns, advantages = self.compute_gae(next_value)

        states = states.to(self.device)
        actions = actions.to(self.device)
        old_log_probs = old_log_probs.to(self.device)
        returns = returns.to(self.device)
        advantages = advantages.to(self.device)

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(4):  # epochs
            logits, values = self.policy(states)
            dist = torch.distributions.Categorical(logits=logits)

            new_log_probs = dist.log_prob(actions)
            ratio = (new_log_probs - old_log_probs).exp()

            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * advantages

            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = (returns - values.squeeze()).pow(2).mean()

            loss = policy_loss + 0.5 * value_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.memory.clear()