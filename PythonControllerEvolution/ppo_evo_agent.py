"""
PPO Evolution Agent — optimized for NVIDIA RTX 4080.

v2 improvements (over v1):
- LR linear decay  3e-4 → 1e-5  over 1 M steps  (prevents late-training regression)
- Entropy decay    0.02  → 0.005 over 1 M steps  (high early exploration, low later)
- Larger rollout   8192 steps                     (smoother gradient estimates)
- EMA reward normaliser                           (tracks recent reward scale better)
- target_kl 0.03 (tighter than v1's 0.05 to limit per-update policy drift)
"""

import math
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast

TILE_H = 16
TILE_W = 32


def _ortho_init(module: nn.Module, gain: float = math.sqrt(2)) -> nn.Module:
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    return module


class RunningNorm:
    """EMA-based reward normaliser.

    Uses exponential moving average so recent reward scale always dominates —
    avoids the Welford issue where early samples permanently dominate the
    running variance after many steps of constant-ish rewards.
    """

    def __init__(self, clip: float = 5.0, alpha: float = 0.001):
        self.mean = 0.0
        self.var = 1.0
        self.alpha = alpha   # EMA rate — larger = faster adaptation
        self.clip = clip

    def update(self, x: float) -> float:
        self.mean = (1 - self.alpha) * self.mean + self.alpha * x
        self.var  = (1 - self.alpha) * self.var  + self.alpha * (x - self.mean) ** 2
        std = math.sqrt(max(self.var, 1e-8))
        return float(np.clip(x / std, -self.clip, self.clip))

    def state_dict(self):
        return {"mean": self.mean, "var": self.var, "alpha": self.alpha}

    def load_state_dict(self, d):
        self.mean  = d["mean"]
        self.var   = d["var"]
        self.alpha = d.get("alpha", self.alpha)


class PolicyEvo(nn.Module):
    def __init__(self, scalar_dim: int, tile_dim: int, act_dim: int):
        super().__init__()
        self.scalar_dim = scalar_dim
        self.tile_dim = tile_dim

        # --- tile CNN: 1×16×32 → 32×8×16 → 64×4×8 → 64×4×8
        self.tile_cnn = nn.Sequential(
            _ortho_init(nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1)),
            nn.ReLU(),
            _ortho_init(nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)),
            nn.ReLU(),
            _ortho_init(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)),
            nn.ReLU(),
        )
        cnn_out = 64 * (TILE_H // 4) * (TILE_W // 4)  # 64*4*8 = 2048

        # --- scalar encoder
        self.scalar_enc = nn.Sequential(
            _ortho_init(nn.Linear(scalar_dim, 128)),
            nn.LayerNorm(128),
            nn.ReLU(),
        )

        # --- shared trunk
        trunk_in = cnn_out + 128  # 2048 + 128 = 2176
        self.trunk = nn.Sequential(
            _ortho_init(nn.Linear(trunk_in, 1024)),
            nn.LayerNorm(1024),
            nn.ReLU(),
            _ortho_init(nn.Linear(1024, 512)),
            nn.LayerNorm(512),
            nn.ReLU(),
        )

        self.policy_head = _ortho_init(nn.Linear(512, act_dim), gain=0.01)
        self.value_head = _ortho_init(nn.Linear(512, 1), gain=1.0)

    def forward(self, x: torch.Tensor):
        single = x.dim() == 1
        if single:
            x = x.unsqueeze(0)

        scalars = x[:, : self.scalar_dim]
        tiles = x[:, self.scalar_dim :].reshape(-1, 1, TILE_H, TILE_W)

        cnn_feat = self.tile_cnn(tiles).flatten(1)
        scalar_feat = self.scalar_enc(scalars)
        features = self.trunk(torch.cat([cnn_feat, scalar_feat], dim=1))

        logits = self.policy_head(features)
        value = self.value_head(features)

        if single:
            logits = logits.squeeze(0)
            value = value.squeeze(0)

        return logits, value


class PPOEvoAgent:
    def __init__(self, scalar_dim: int, tile_dim: int, act_dim: int):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"PPOEvoAgent device: {self.device}"
              + (f" ({torch.cuda.get_device_name(0)})" if self.device.type == "cuda" else ""))

        self.scalar_dim = scalar_dim
        self.tile_dim = tile_dim
        self.act_dim = act_dim

        self.policy = PolicyEvo(scalar_dim, tile_dim, act_dim).to(self.device)

        # PPO hyperparameters
        self.gamma = 0.99
        self.lam = 0.95
        self.clip = 0.2
        self.max_grad_norm = 0.5
        self.value_coef = 0.5
        self.target_kl = 0.03       # slightly tighter — limits per-update drift
        self.n_epochs = 6
        self.rollout_steps = 8192   # larger buffer → smoother gradient signal

        # Decaying schedules: linear from _start → _end over decay_horizon steps
        self.lr_start       = 3e-4
        self.lr_end         = 1e-5
        self.entropy_start  = 0.02   # high early → explore past the 10% wall
        self.entropy_end    = 0.005
        self.decay_horizon  = 1_000_000

        # Current values (updated by _anneal each update call)
        self.entropy_coef = self.entropy_start
        self.lr           = self.lr_start
        self.total_steps  = 0

        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.lr_start, eps=1e-5)
        self.scaler = GradScaler("cuda", enabled=(self.device.type == "cuda"))
        self.reward_norm = RunningNorm(clip=10.0)
        self.memory: list = []

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def save(self, path):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "scalar_dim": self.scalar_dim,
                "tile_dim": self.tile_dim,
                "act_dim": self.act_dim,
                "policy_state_dict": self.policy.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scaler_state_dict": self.scaler.state_dict(),
                "reward_norm": self.reward_norm.state_dict(),
                "total_steps": self.total_steps,
                "lr": self.lr,
                "entropy_coef": self.entropy_coef,
            },
            p,
        )

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(ckpt["policy_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scaler_state_dict" in ckpt:
            self.scaler.load_state_dict(ckpt["scaler_state_dict"])
        if "reward_norm" in ckpt:
            self.reward_norm.load_state_dict(ckpt["reward_norm"])
        if "total_steps" in ckpt:
            self.total_steps = ckpt["total_steps"]
        # Restore annealed values so training continues from correct schedule position
        self._anneal()
        if "lr" in ckpt:
            self.lr = ckpt["lr"]
            for g in self.optimizer.param_groups:
                g["lr"] = self.lr
        if "entropy_coef" in ckpt:
            self.entropy_coef = ckpt["entropy_coef"]

    def load_demos(self, path):
        # Stub — BC pre-training not needed for evolution agent
        pass

    # ------------------------------------------------------------------
    # Annealing
    # ------------------------------------------------------------------

    def _anneal(self) -> None:
        """Update lr and entropy_coef linearly based on total_steps."""
        frac = max(0.0, 1.0 - self.total_steps / self.decay_horizon)
        self.lr           = self.lr_end          + frac * (self.lr_start - self.lr_end)
        self.entropy_coef = self.entropy_end     + frac * (self.entropy_start - self.entropy_end)
        for g in self.optimizer.param_groups:
            g["lr"] = self.lr

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def act(self, state):
        try:
            t = torch.tensor(state, dtype=torch.float32).to(self.device)
            with torch.no_grad():
                with autocast("cuda", enabled=(self.device.type == "cuda")):
                    logits, value = self.policy(t)
            logits = torch.nan_to_num(logits.float(), nan=0.0, posinf=20.0, neginf=-20.0)
            value = torch.nan_to_num(value.float(), nan=0.0)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            return action.item(), dist.log_prob(action).item(), value.item()
        except Exception as exc:
            import traceback
            print(f"ERROR act(): {exc}")
            traceback.print_exc()
            return int(torch.randint(0, self.act_dim, (1,)).item()), 0.0, 0.0

    def store(self, transition):
        state, action, reward, done, log_prob, value = transition
        reward = self.reward_norm.update(reward)
        self.memory.append((state, action, reward, done, log_prob, value))

    # ------------------------------------------------------------------
    # GAE
    # ------------------------------------------------------------------

    def _compute_gae(self, next_value: float):
        states, actions, rewards, dones, log_probs, values = zip(*self.memory)

        gae = 0.0
        advantages = []
        vals_ext = list(values) + [next_value]

        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * vals_ext[t + 1] * (1 - dones[t]) - vals_ext[t]
            gae = delta + self.gamma * self.lam * (1 - dones[t]) * gae
            advantages.insert(0, gae)

        returns = [a + v for a, v in zip(advantages, values)]

        return (
            torch.tensor(np.asarray(states), dtype=torch.float32),
            torch.tensor(actions, dtype=torch.long),
            torch.tensor(log_probs, dtype=torch.float32),
            torch.tensor(returns, dtype=torch.float32),
            torch.tensor(advantages, dtype=torch.float32),
            torch.tensor(list(values), dtype=torch.float32),
        )

    # ------------------------------------------------------------------
    # PPO update
    # ------------------------------------------------------------------

    def update(self, next_value: float):
        try:
            return self._update_impl(next_value)
        except Exception as exc:
            import traceback
            print(f"ERROR update(): {exc}")
            traceback.print_exc()
            self.memory.clear()
            return None

    def _update_impl(self, next_value: float):
        if len(self.memory) < 64:
            self.memory.clear()
            return None

        self._anneal()  # update lr and entropy_coef before each gradient step

        states, actions, old_log_probs, returns, advantages, old_values = self._compute_gae(next_value)

        states = states.to(self.device)
        actions = actions.to(self.device)
        old_log_probs = old_log_probs.to(self.device)
        returns = returns.to(self.device)
        advantages = advantages.to(self.device)
        old_values = old_values.to(self.device)

        # Advantage normalisation
        advantages = torch.nan_to_num(advantages, nan=0.0)
        adv_std = advantages.std(unbiased=False)
        if torch.isfinite(adv_std) and adv_std.item() > 1e-8:
            advantages = (advantages - advantages.mean()) / (adv_std + 1e-8)

        if not (
            torch.isfinite(states).all()
            and torch.isfinite(old_log_probs).all()
            and torch.isfinite(returns).all()
            and torch.isfinite(advantages).all()
        ):
            self.memory.clear()
            return None

        n = states.shape[0]
        batch_size = min(512, n)  # larger mini-batches for RTX 4080

        loss_acc = policy_acc = value_acc = entropy_acc = 0.0
        update_steps = 0

        for _ in range(self.n_epochs):
            perm = torch.randperm(n, device=self.device)
            early_stop = False
            for start in range(0, n, batch_size):
                idx = perm[start : start + batch_size]
                bs, ba = states[idx], actions[idx]
                b_old_lp = old_log_probs[idx]
                b_ret = returns[idx]
                b_adv = advantages[idx]
                b_oval = old_values[idx]

                with autocast("cuda", enabled=(self.device.type == "cuda")):
                    logits, values_pred = self.policy(bs)
                    logits = torch.nan_to_num(logits.float(), nan=0.0, posinf=20.0, neginf=-20.0)
                    values_pred = torch.nan_to_num(values_pred.float(), nan=0.0).squeeze(-1)
                    dist = torch.distributions.Categorical(logits=logits.float())
                    new_lp = dist.log_prob(ba)

                    with torch.no_grad():
                        kl = (b_old_lp - new_lp).mean()
                    if kl > 1.5 * self.target_kl:
                        early_stop = True
                        break

                    ratio = (new_lp - b_old_lp).exp()
                    surr1 = ratio * b_adv
                    surr2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * b_adv
                    policy_loss = -torch.min(surr1, surr2).mean()

                    v_clipped = b_oval + torch.clamp(values_pred - b_oval, -self.clip, self.clip)
                    value_loss = torch.max(
                        (b_ret - values_pred).pow(2),
                        (b_ret - v_clipped).pow(2),
                    ).mean()

                    entropy = dist.entropy().mean()
                    loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

                if not torch.isfinite(loss):
                    continue

                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()

                loss_acc += loss.item()
                policy_acc += policy_loss.item()
                value_acc += value_loss.item()
                entropy_acc += entropy.item()
                update_steps += 1

            if early_stop:
                break

        self.total_steps += n
        self.memory.clear()

        if update_steps == 0:
            return None

        return {
            "loss": loss_acc / update_steps,
            "policy_loss": policy_acc / update_steps,
            "value_loss": value_acc / update_steps,
            "entropy": entropy_acc / update_steps,
            "batch_size": n,
            "bc_loss": 0.0,
            "lr": self.lr,
            "entropy_coef": self.entropy_coef,
        }
