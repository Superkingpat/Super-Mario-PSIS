import os
import random
import math
import time
from collections import deque, namedtuple
from dataclasses import dataclass, asdict

import cv2
import gym
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from gym_super_mario_bros.smb_env import SuperMarioBrosEnv
from nes_py.wrappers import JoypadSpace
from nes_py._rom import ROM as NesRom

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm.auto import trange, tqdm


if not hasattr(np, "bool8"):
    np.bool8 = np.bool_


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])


def _patch_nes_rom_overflow():
    """Avoid uint8 overflow in nes_py ROM arithmetic on newer NumPy versions."""
    def _prg_rom_size(self):
        return 16 * int(self.header[4])

    def _chr_rom_size(self):
        return 8 * int(self.header[5])

    NesRom.prg_rom_size = property(_prg_rom_size)
    NesRom.chr_rom_size = property(_chr_rom_size)


_patch_nes_rom_overflow()


def _patch_smb_uint8_overflow():
    """Avoid uint8 overflow in gym_super_mario_bros RAM arithmetic."""
    def _level(self):
        return int(self.ram[0x075f]) * 4 + int(self.ram[0x075c])

    def _world(self):
        return int(self.ram[0x075f]) + 1

    def _stage(self):
        return int(self.ram[0x075c]) + 1

    def _area(self):
        return int(self.ram[0x0760]) + 1

    def _life(self):
        return int(self.ram[0x075a])

    def _x_position(self):
        return int(self.ram[0x6d]) * 0x100 + int(self.ram[0x86])

    def _left_x_position(self):
        return (int(self.ram[0x86]) - int(self.ram[0x071c])) % 256

    SuperMarioBrosEnv._level = property(_level)
    SuperMarioBrosEnv._world = property(_world)
    SuperMarioBrosEnv._stage = property(_stage)
    SuperMarioBrosEnv._area = property(_area)
    SuperMarioBrosEnv._life = property(_life)
    SuperMarioBrosEnv._x_position = property(_x_position)
    SuperMarioBrosEnv._left_x_position = property(_left_x_position)


_patch_smb_uint8_overflow()


def _unwrap_reset(reset_result):
    """Handle both Gym (obs) and Gymnasium ((obs, info)) reset signatures."""
    if isinstance(reset_result, tuple):
        return reset_result[0]
    return reset_result


def _unwrap_step(step_result):
    """Handle both Gym (4-tuple) and Gymnasium (5-tuple) step signatures."""
    if len(step_result) == 5:
        obs, reward, terminated, truncated, info = step_result
        done = bool(terminated or truncated)
        return obs, reward, done, info
    obs, reward, done, info = step_result
    return obs, reward, bool(done), info


# ============================================================
# Environment wrappers
# ============================================================

class PreprocessFrame(gym.ObservationWrapper):
    """
    Convert RGB frame to grayscale and resize.
    Paper does not fully specify preprocessing, so we use a standard RL setup.
    """
    def __init__(self, env, width=84, height=84):
        super().__init__(env)
        self.width = width
        self.height = height
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(height, width, 1),
            dtype=np.uint8
        )

    def observation(self, obs):
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return resized[:, :, None]


class FrameStack(gym.Wrapper):
    def __init__(self, env, k=4):
        super().__init__(env)
        self.k = k
        self.frames = deque(maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(shp[0], shp[1], k),
            dtype=np.uint8
        )

    def reset(self, **kwargs):
        obs = _unwrap_reset(self.env.reset(**kwargs))
        for _ in range(self.k):
            self.frames.append(obs)
        return self._get_obs()

    def step(self, action):
        obs, reward, done, info = _unwrap_step(self.env.step(action))
        self.frames.append(obs)
        return self._get_obs(), reward, done, info

    def _get_obs(self):
        return np.concatenate(list(self.frames), axis=2)


class NormalizeObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        shp = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=shp, dtype=np.float32
        )

    def observation(self, obs):
        return np.array(obs, dtype=np.float32) / 255.0


class RewardScaler(gym.RewardWrapper):
    """
    Mild reward scaling for stability.
    The paper does not define reward shaping precisely.
    """
    def reward(self, reward):
        return reward / 10.0


def make_env(world=1, stage=1, version="v0", action_set=None):
    env_name = f"SuperMarioBros-{world}-{stage}-{version}"
    env = gym_super_mario_bros.make(env_name)
    action_set = action_set if action_set is not None else COMPLEX_MOVEMENT[:12]
    env = JoypadSpace(env, action_set)
    env = PreprocessFrame(env, width=84, height=84)
    env = FrameStack(env, k=4)
    env = NormalizeObservation(env)
    env = RewardScaler(env)
    return env


# ============================================================
# Replay Buffer
# ============================================================

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        return Transition(*zip(*batch))

    def __len__(self):
        return len(self.buffer)


# ============================================================
# Networks
# ============================================================

class DQN(nn.Module):
    """
    Matches the paper at a high level:
    - 2 conv layers
    - 2 linear layers
    - final output = number of actions
    The paper reports shapes leading to:
    Conv2d -> Conv2d -> Linear(512) -> Linear(12)
    """
    def __init__(self, input_shape, n_actions):
        super().__init__()
        c, h, w = input_shape

        self.features = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4),  # common Atari-style choice
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU()
        )

        with torch.no_grad():
            dummy = torch.zeros(1, c, h, w)
            n_flatten = self.features(dummy).reshape(1, -1).size(1)

        self.head = nn.Sequential(
            nn.Linear(n_flatten, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.reshape(x.size(0), -1)
        return self.head(x)


class DuelingDQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super().__init__()
        c, h, w = input_shape

        self.features = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU()
        )

        with torch.no_grad():
            dummy = torch.zeros(1, c, h, w)
            n_flatten = self.features(dummy).reshape(1, -1).size(1)

        self.shared = nn.Sequential(
            nn.Linear(n_flatten, 512),
            nn.ReLU()
        )

        self.value_stream = nn.Linear(512, 1)
        self.advantage_stream = nn.Linear(512, n_actions)

    def forward(self, x):
        x = self.features(x)
        x = x.reshape(x.size(0), -1)
        x = self.shared(x)

        value = self.value_stream(x)
        advantage = self.advantage_stream(x)

        # Q(s,a) = V(s) + A(s,a) - mean_a A(s,a)
        return value + advantage - advantage.mean(dim=1, keepdim=True)


# ============================================================
# Config
# ============================================================

@dataclass
class TrainConfig:
    episodes: int = 3000
    gamma: float = 0.99
    learning_rate: float = 1e-4
    batch_size: int = 256
    replay_capacity: int = 100_000
    min_replay_size: int = 10_000
    target_update_freq: int = 5_000
    train_freq: int = 4
    max_steps_per_episode: int = 5_000
    epsilon_start: float = 1.0
    epsilon_end: float = 0.1
    epsilon_decay_frames: int = 200_000
    log_every_episodes: int = 5
    save_dir: str = "results"
    seed: int = 42


# ============================================================
# Agent helpers
# ============================================================

def to_tensor_state(state: np.ndarray) -> torch.Tensor:
    # input state: H x W x C
    state = np.transpose(state, (2, 0, 1))  # C x H x W
    return torch.as_tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)


def to_tensor_batch(states) -> torch.Tensor:
    # input states: B x H x W x C
    batch = np.stack(states, axis=0)
    batch = np.transpose(batch, (0, 3, 1, 2))  # B x C x H x W
    return torch.as_tensor(batch, dtype=torch.float32, device=DEVICE)


def linear_epsilon(step, eps_start, eps_end, decay_frames):
    t = min(step / decay_frames, 1.0)
    return eps_start + t * (eps_end - eps_start)


def select_action(model, state, epsilon, n_actions):
    if random.random() < epsilon:
        return random.randrange(n_actions)
    with torch.no_grad():
        q_values = model(state)
        return int(q_values.argmax(dim=1).item())


def compute_dqn_loss(batch, policy_net, target_net, gamma):
    states = to_tensor_batch(batch.state)
    actions = torch.tensor(batch.action, dtype=torch.long, device=DEVICE).unsqueeze(1)
    rewards = torch.tensor(batch.reward, dtype=torch.float32, device=DEVICE).unsqueeze(1)
    next_states = to_tensor_batch(batch.next_state)
    dones = torch.tensor(batch.done, dtype=torch.float32, device=DEVICE).unsqueeze(1)

    q_values = policy_net(states).gather(1, actions)

    with torch.no_grad():
        max_next_q = target_net(next_states).max(dim=1, keepdim=True)[0]
        target = rewards + gamma * (1.0 - dones) * max_next_q

    loss = nn.MSELoss()(q_values, target)
    return loss


# ============================================================
# Training
# ============================================================

def train_one_experiment(model_name: str, config: TrainConfig):
    os.makedirs(config.save_dir, exist_ok=True)
    set_seed(config.seed)

    env = make_env()
    # Seed env and action sampling for reproducibility where supported.
    try:
        env.reset(seed=config.seed)
    except TypeError:
        env.seed(config.seed)
    if hasattr(env.action_space, "seed"):
        env.action_space.seed(config.seed)

    n_actions = env.action_space.n
    obs_shape = env.observation_space.shape  # H, W, C
    input_shape = (obs_shape[2], obs_shape[0], obs_shape[1])

    if model_name == "dqn":
        policy_net = DQN(input_shape, n_actions).to(DEVICE)
        target_net = DQN(input_shape, n_actions).to(DEVICE)
    elif model_name == "dueling_dqn":
        policy_net = DuelingDQN(input_shape, n_actions).to(DEVICE)
        target_net = DuelingDQN(input_shape, n_actions).to(DEVICE)
    else:
        raise ValueError("model_name must be 'dqn' or 'dueling_dqn'")

    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=config.learning_rate)
    replay_buffer = ReplayBuffer(config.replay_capacity)

    episode_rewards = []
    avg100_rewards = []
    global_step = 0
    log_every = max(1, int(config.log_every_episodes))

    start_time = time.time()

    with trange(
        1,
        config.episodes + 1,
        desc=f"[{model_name}]",
        dynamic_ncols=True,
        leave=True,
    ) as episode_iter:
        for episode in episode_iter:
            state = _unwrap_reset(env.reset())
            episode_reward = 0.0
            epsilon = config.epsilon_start

            for _ in range(config.max_steps_per_episode):
                epsilon = linear_epsilon(
                    global_step,
                    config.epsilon_start,
                    config.epsilon_end,
                    config.epsilon_decay_frames
                )

                state_tensor = to_tensor_state(state)
                action = select_action(policy_net, state_tensor, epsilon, n_actions)

                next_state, reward, done, info = env.step(action)
                replay_buffer.push(state, action, reward, next_state, done)

                state = next_state
                episode_reward += reward
                global_step += 1

                if len(replay_buffer) >= config.min_replay_size and global_step % config.train_freq == 0:
                    batch = replay_buffer.sample(config.batch_size)
                    loss = compute_dqn_loss(batch, policy_net, target_net, config.gamma)

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(policy_net.parameters(), 10.0)
                    optimizer.step()

                if global_step % config.target_update_freq == 0:
                    target_net.load_state_dict(policy_net.state_dict())

                if done:
                    break

            episode_rewards.append(episode_reward)
            avg100 = float(np.mean(episode_rewards[-100:]))
            avg100_rewards.append(avg100)

            episode_iter.set_postfix(
                reward=f"{episode_reward:.2f}",
                avg100=f"{avg100:.2f}",
                eps=f"{epsilon:.3f}",
            )

            if episode == 1 or episode % log_every == 0 or episode == config.episodes:
                tqdm.write(
                    f"[{model_name}] Episode {episode}/{config.episodes} | "
                    f"Reward: {episode_reward:.2f} | Avg100: {avg100:.2f} | "
                    f"Epsilon: {epsilon:.3f}"
                )

    elapsed = time.time() - start_time
    env.close()

    result = {
        "model": model_name,
        "config": asdict(config),
        "episode_rewards": episode_rewards,
        "avg100_rewards": avg100_rewards,
        "final_avg100": float(np.mean(episode_rewards[-100:])),
        "elapsed_sec": elapsed,
    }

    out_path = os.path.join(
        config.save_dir,
        f"{model_name}_lr{config.learning_rate}_bs{config.batch_size}.pt"
    )
    torch.save(result, out_path)
    print(f"Saved result to {out_path}")

    return result


# ============================================================
# Plotting / analysis
# ============================================================

def plot_comparison(result_a, result_b, save_path="comparison.png"):
    plt.figure(figsize=(10, 6))
    plt.plot(result_a["avg100_rewards"], label=result_a["model"])
    plt.plot(result_b["avg100_rewards"], label=result_b["model"])
    plt.xlabel("Episode")
    plt.ylabel("Average score over last 100 episodes")
    plt.title("Comparison of DQN and Dueling DQN")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Saved plot to {save_path}")


def print_table(title, rows):
    print("\n" + title)
    print("-" * len(title))
    for row in rows:
        print(row)


# ============================================================
# Experiment runners
# ============================================================

def run_main_3000_epoch_comparison():
    """
    Paper's main comparison:
    learning rate = 0.0001
    batch size = 256
    3000 epochs/episodes
    """
    base = TrainConfig(
        episodes=3000,
        learning_rate=1e-4,
        batch_size=256,
        save_dir="results_main"
    )

    dqn_res = train_one_experiment("dqn", base)
    duel_res = train_one_experiment("dueling_dqn", base)

    plot_comparison(dqn_res, duel_res, save_path="results_main/main_comparison.png")

    print("\nFinal Avg100")
    print(f"DQN:         {dqn_res['final_avg100']:.2f}")
    print(f"Dueling DQN: {duel_res['final_avg100']:.2f}")


def run_learning_rate_sweep():
    """
    Paper compares learning rates:
    0.0001, 0.001, 0.01, 0.1
    """
    learning_rates = [1e-4, 1e-3, 1e-2, 1e-1]
    rows = []

    for lr in learning_rates:
        cfg = TrainConfig(
            episodes=2000,
            learning_rate=lr,
            batch_size=256,
            save_dir="results_lr"
        )

        dqn_res = train_one_experiment("dqn", cfg)
        duel_res = train_one_experiment("dueling_dqn", cfg)

        rows.append({
            "learning_rate": lr,
            "dqn_final_avg100": round(dqn_res["final_avg100"], 2),
            "dueling_final_avg100": round(duel_res["final_avg100"], 2)
        })

    print_table("Learning-rate sweep", rows)


def run_batch_size_sweep():
    """
    Paper compares batch sizes:
    64, 128, 256, 512
    """
    batch_sizes = [64, 128, 256, 512]
    rows = []

    for bs in batch_sizes:
        cfg = TrainConfig(
            episodes=2000,
            learning_rate=1e-4,
            batch_size=bs,
            save_dir="results_bs"
        )

        dqn_res = train_one_experiment("dqn", cfg)
        duel_res = train_one_experiment("dueling_dqn", cfg)

        rows.append({
            "batch_size": bs,
            "dqn_final_avg100": round(dqn_res["final_avg100"], 2),
            "dueling_final_avg100": round(duel_res["final_avg100"], 2)
        })

    print_table("Batch-size sweep", rows)


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    # Choose what to run:
    # 1) main comparison
    run_main_3000_epoch_comparison()

    # 2) hyperparameter sweeps
    # run_learning_rate_sweep()
    # run_batch_size_sweep()