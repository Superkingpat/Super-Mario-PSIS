import argparse
import json
import math
import socket
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from ppo_agent import PPOAgent

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

ACTION_ORDER = ["LEFT", "RIGHT", "DOWN", "SPEED", "JUMP"]

TILE_SIZE = 16.0
RELATIVE_TILE_WINDOW_WIDTH = 32
RELATIVE_TILE_WINDOW_HEIGHT = 16
MAX_SCENE_TILES = RELATIVE_TILE_WINDOW_WIDTH * RELATIVE_TILE_WINDOW_HEIGHT
ENEMY_COUNT = 5
ENEMY_FEATURES_PER_ENEMY = 5  # dx, dy, velx, vely, type
ENEMY_FEATURE_DIM = ENEMY_COUNT * ENEMY_FEATURES_PER_ENEMY
MAX_ENEMY_TYPE_ID = 16.0
MAX_TILE_ABS_VALUE = 32.0
MAX_MARIO_X = 5000.0
MAX_MARIO_Y = 300.0
MAX_VEL_X = 15.0
MAX_VEL_Y = 20.0
MAX_REMAINING_TIME = 200000.0

# Stuck penalty: if Mario's x does not change for many consecutive steps, apply
# a ramping negative reward to discourage dithering.
STUCK_NO_PROGRESS_EPS_PX = 1.0
STUCK_STEPS_GRACE = 60
STUCK_STEPS_RAMP = 120
STUCK_PENALTY_MAX = 0.05

CURRENT_CONTROLLER = None

OBS_BRICK = 22
OBS_QUESTION_BLOCK = 24
OBS_USED_BLOCK = 30


@dataclass
class ElementPos:
    type_id: int
    x: float
    y: float


@dataclass
class LevelBlock:
    x: int
    y: int
    tile_id: int


@dataclass
class LevelData:
    width: int
    height: int
    blocks: list[LevelBlock]


@dataclass
class StepObservation:
    step: int
    mario_x: float
    mario_y: float
    vel_x: float
    vel_y: float
    mode: int
    on_ground: bool
    may_jump: bool
    can_jump_higher: bool
    remaining_time: int
    completion: float
    status: str
    enemies: list[ElementPos]
    sprites: list[ElementPos]
    scene_width: int
    scene_height: int
    scene_tiles: list[int]
    astar_actions: dict[str, list[bool]]


class MarioPythonController:
    def __init__(
        self,
        model_path: str | None = None,
        stats_path: str | None = None,
        save_every: int = 1,
        tensorboard_dir: str | None = None,
        jump_control: str = "click",
    ) -> None:
        global CURRENT_CONTROLLER
        self.level_data = None
        self.level_tile_map: dict[tuple[int, int], int] = {}

        self.prev_obs = None
        self.pending_transition = None

        self.obs_dim = 10 + ENEMY_FEATURE_DIM + MAX_SCENE_TILES

        self.jump_control = jump_control
        if self.jump_control not in ("click", "press-release"):
            raise ValueError(f"Unsupported jump_control: {self.jump_control}")

        # Discrete actions.
        # - click: action directly includes a one-frame JUMP bit (tap/hold is learned by repeating the action)
        # - press-release: action space includes explicit PRESS_JUMP and RELEASE_JUMP events; output keeps an internal jump-held state
        self._action_table_click: list[list[int]] = [
            [0, 1, 0, 1, 0],  # run right
            [0, 1, 0, 1, 1],  # jump right (tap)
            [1, 0, 0, 0, 0],  # left
            [0, 0, 0, 0, 0],  # idle
        ]

        # press-release actions:
        # 0 MOVE_RIGHT, 1 PRESS_JUMP, 2 RELEASE_JUMP, 3 MOVE_LEFT, 4 IDLE
        self._action_count_press_release = 5
        self._move_bits = [0, 1, 0, 1]  # LEFT, RIGHT, DOWN, SPEED
        self._jump_held = False

        self.act_dim = len(self._action_table_click) if self.jump_control == "click" else self._action_count_press_release

        self.agent = PPOAgent(self.obs_dim, self.act_dim)
        self.model_path = Path(model_path) if model_path else None
        self.stats_path = Path(stats_path) if stats_path else None
        self.save_every = max(1, save_every)
        self.tensorboard_dir = Path(tensorboard_dir) if tensorboard_dir else None
        self.tensorboard_writer = None
        CURRENT_CONTROLLER = self

        self.step_counter = 0
        self.update_interval = 2048
        self.episode_reward = 0.0
        self.episode_steps = 0
        self.episode_jump_reward = 0.0
        self.episode_count = 0
        self.update_count = 0

        self._stuck_steps = 0

    def compute_stuck_penalty(self, prev: StepObservation, curr: StepObservation) -> float:
        dx = float(curr.mario_x - prev.mario_x)
        if abs(dx) <= STUCK_NO_PROGRESS_EPS_PX:
            self._stuck_steps += 1
        else:
            self._stuck_steps = 0

        if self._stuck_steps <= STUCK_STEPS_GRACE:
            return 0.0

        over = self._stuck_steps - STUCK_STEPS_GRACE
        ramp = min(1.0, over / float(STUCK_STEPS_RAMP))
        return -STUCK_PENALTY_MAX * ramp

        if self.tensorboard_dir:
            if SummaryWriter is None:
                print("TensorBoard logging requested but 'tensorboard' is not installed in this environment.")
            else:
                self.tensorboard_dir.mkdir(parents=True, exist_ok=True)
                self.tensorboard_writer = SummaryWriter(log_dir=str(self.tensorboard_dir))
                print(f"TensorBoard logs will be written to {self.tensorboard_dir}")

        if self.model_path and self.model_path.exists():
            try:
                self.agent.load(self.model_path)
            except Exception as exc:
                backup_path = self.model_path.with_suffix(
                    self.model_path.suffix + f".incompatible_{int(time.time())}"
                )
                try:
                    self.model_path.rename(backup_path)
                    print(
                        f"WARNING: Failed to load PPO checkpoint from {self.model_path}: {exc}"
                    )
                    print(f"Renamed incompatible checkpoint to {backup_path} and starting fresh.")
                except OSError as rename_exc:
                    print(
                        f"WARNING: Failed to load PPO checkpoint from {self.model_path}: {exc}"
                    )
                    print(f"Could not rename checkpoint ({rename_exc}). Starting fresh anyway.")
            else:
                print(f"Loaded PPO checkpoint from {self.model_path}")

    def resolve_action_bits(self, action_idx: int) -> list[int]:
        if self.jump_control == "click":
            if 0 <= action_idx < len(self._action_table_click):
                return self._action_table_click[action_idx]
            return self._action_table_click[-1]

        # press-release mode
        if action_idx == 0:  # MOVE_RIGHT
            self._move_bits = [0, 1, 0, 1]
        elif action_idx == 1:  # PRESS_JUMP
            self._jump_held = True
        elif action_idx == 2:  # RELEASE_JUMP
            self._jump_held = False
        elif action_idx == 3:  # MOVE_LEFT
            self._move_bits = [1, 0, 0, 0]
        elif action_idx == 4:  # IDLE
            self._move_bits = [0, 0, 0, 0]

        return [*self._move_bits, 1 if self._jump_held else 0]

    def set_level_data(self, level_data):
        self.level_data = level_data
        self.level_tile_map = {
            (block.x, block.y): block.tile_id
            for block in level_data.blocks
        }

    def choose_actions(self, obs: StepObservation):
        state = obs_to_vector(obs)
        done = obs.status in TERMINAL_STATUSES

        if self.pending_transition is not None and self.prev_obs is not None:
            prev_state, prev_action_idx, prev_action_bits, prev_log_prob, prev_value = self.pending_transition
            reward = compute_reward(prev_action_bits, self.prev_obs, obs)
            jump_bonus = self.compute_jump_bonus(prev_action_bits, self.prev_obs, obs)
            reward += jump_bonus
            reward += self.compute_stuck_penalty(self.prev_obs, obs)
            self.agent.store((
                prev_state,
                prev_action_idx,
                reward,
                done,
                prev_log_prob,
                prev_value
            ))
            self.episode_reward += reward
            self.episode_steps += 1

        if done:
            self.pending_transition = None
            self.prev_obs = obs
            self._stuck_steps = 0
            return self.resolve_action_bits(self.act_dim - 1)

        action_idx, log_prob, value = self.agent.act(state)
        action = self.resolve_action_bits(action_idx)
        self.pending_transition = (state, action_idx, action, log_prob, value)
        self.prev_obs = obs
        self.step_counter += 1

        if self.step_counter % self.update_interval == 0 and self.agent.memory:
            metrics = self.agent.update(next_value=value)
            self.log_update_metrics(metrics)

        return action

    def handle_episode_end(self, end_parts: list[str]) -> None:
        status = end_parts[1] if len(end_parts) > 1 else "UNKNOWN"
        completion = float(end_parts[2]) if len(end_parts) > 2 else 0.0
        remaining_time = int(end_parts[3]) if len(end_parts) > 3 else 0
        jumps = int(end_parts[4]) if len(end_parts) > 4 else 0
        kills = int(end_parts[5]) if len(end_parts) > 5 else 0
        coins = int(end_parts[6]) if len(end_parts) > 6 else 0

        # The framework reports terminal state via END, not a final STEP, so we
        # must flush the last pending action into memory here.
        if self.pending_transition is not None and self.prev_obs is not None:
            terminal_obs = StepObservation(
                step=self.prev_obs.step + 1,
                mario_x=self.prev_obs.mario_x,
                mario_y=self.prev_obs.mario_y,
                vel_x=0.0,
                vel_y=0.0,
                mode=self.prev_obs.mode,
                on_ground=self.prev_obs.on_ground,
                may_jump=self.prev_obs.may_jump,
                can_jump_higher=self.prev_obs.can_jump_higher,
                remaining_time=remaining_time,
                completion=completion,
                status=status,
                enemies=self.prev_obs.enemies,
                sprites=self.prev_obs.sprites,
                scene_width=self.prev_obs.scene_width,
                scene_height=self.prev_obs.scene_height,
                scene_tiles=self.prev_obs.scene_tiles,
                astar_actions=self.prev_obs.astar_actions,
            )
            prev_state, prev_action_idx, prev_action_bits, prev_log_prob, prev_value = self.pending_transition
            reward = compute_reward(prev_action_bits, self.prev_obs, terminal_obs)
            jump_bonus = self.compute_jump_bonus(prev_action_bits, self.prev_obs, terminal_obs) * 0.5
            reward += jump_bonus
            reward += self.compute_stuck_penalty(self.prev_obs, terminal_obs) * 0.5

            # Episode-level bonuses (applied to the terminal transition).
            reward += float(kills) * 10.0
            reward += float(coins) * 5.0
            self.agent.store((
                prev_state,
                prev_action_idx,
                reward,
                True,
                prev_log_prob,
                prev_value
            ))
            self.episode_reward += reward
            self.episode_steps += 1

        if self.agent.memory:
            metrics = self.agent.update(next_value=0.0)
            self.log_update_metrics(metrics)

        self.episode_count += 1

        episode_stats = {
            "episode": self.episode_count,
            "status": status,
            "completion": completion,
            "remaining_time": remaining_time,
            "jumps": jumps,
            "kills": kills,
            "coins": coins,
            "reward": self.episode_reward,
            "steps": self.episode_steps,
        }
        print(
            "Episode "
            f"{self.episode_count}: status={status} completion={completion:.5f} "
            f"reward={self.episode_reward:.3f} steps={self.episode_steps}"
        )

        if self.stats_path:
            self.stats_path.parent.mkdir(parents=True, exist_ok=True)
            with self.stats_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(episode_stats) + "\n")

        if self.tensorboard_writer is not None:
            self.tensorboard_writer.add_scalar("episode/reward", self.episode_reward, self.episode_count)
            self.tensorboard_writer.add_scalar("episode/completion", completion, self.episode_count)
            self.tensorboard_writer.add_scalar("episode/remaining_time", remaining_time, self.episode_count)
            self.tensorboard_writer.add_scalar("episode/jumps", jumps, self.episode_count)
            self.tensorboard_writer.add_scalar("episode/kills", kills, self.episode_count)
            self.tensorboard_writer.add_scalar("episode/coins", coins, self.episode_count)
            self.tensorboard_writer.add_scalar("episode/steps", self.episode_steps, self.episode_count)
            self.tensorboard_writer.add_text("episode/status", status, self.episode_count)
            self.tensorboard_writer.flush()

        if self.model_path and self.episode_count % self.save_every == 0:
            self.agent.save(self.model_path)
            print(f"Saved PPO checkpoint to {self.model_path}")

        self.prev_obs = None
        self.pending_transition = None
        self.episode_reward = 0.0
        self.episode_steps = 0
        self.episode_jump_reward = 0.0
        self._stuck_steps = 0

    def log_update_metrics(self, metrics: dict | None) -> None:
        if not metrics:
            return

        self.update_count += 1
        if self.tensorboard_writer is not None:
            self.tensorboard_writer.add_scalar("train/loss", metrics["loss"], self.update_count)
            self.tensorboard_writer.add_scalar("train/policy_loss", metrics["policy_loss"], self.update_count)
            self.tensorboard_writer.add_scalar("train/value_loss", metrics["value_loss"], self.update_count)
            self.tensorboard_writer.add_scalar("train/entropy", metrics["entropy"], self.update_count)
            self.tensorboard_writer.add_scalar("train/batch_size", metrics["batch_size"], self.update_count)
            self.tensorboard_writer.add_scalar("train/environment_steps", self.step_counter, self.update_count)
            self.tensorboard_writer.flush()

    def compute_jump_bonus(self, action: list[int], prev: StepObservation, curr: StepObservation) -> float:
        jump_pressed = bool(action[4]) if len(action) > 4 else False
        jump_started = jump_pressed and prev.on_ground and not curr.on_ground
        if not jump_started or self.episode_jump_reward >= 50.0:
            return 0.0

        bonus = min(1.0, 50.0 - self.episode_jump_reward)
        self.episode_jump_reward += bonus
        return bonus


def obs_to_vector(obs: StepObservation):
    tiles = build_relative_tile_window(obs)
    tiles = [float(np.clip(tile / MAX_TILE_ABS_VALUE, -1.0, 1.0)) for tile in tiles]

    mario_x = float(np.clip(obs.mario_x / MAX_MARIO_X, -1.0, 1.0))
    mario_y = float(np.clip(obs.mario_y / MAX_MARIO_Y, -1.0, 1.0))
    vel_x = float(np.clip(obs.vel_x / MAX_VEL_X, -1.0, 1.0))
    vel_y = float(np.clip(obs.vel_y / MAX_VEL_Y, -1.0, 1.0))
    mode = float(np.clip(obs.mode / 2.0, 0.0, 1.0))
    on_ground = float(obs.on_ground)
    may_jump = float(obs.may_jump)
    can_jump_higher = float(obs.can_jump_higher)
    remaining_time = float(np.clip(obs.remaining_time / MAX_REMAINING_TIME, 0.0, 1.0))
    completion = float(np.clip(obs.completion, 0.0, 1.0))

    prev_obs = None
    if CURRENT_CONTROLLER is not None:
        prev_obs = CURRENT_CONTROLLER.prev_obs
    enemy_features = build_enemy_features(obs, prev_obs)

    return np.array([
        mario_x,
        mario_y,
        vel_x,
        vel_y,
        mode,
        on_ground,
        may_jump,
        can_jump_higher,
        remaining_time,
        completion,
        *enemy_features,
        *tiles
    ], dtype=np.float32)


def build_enemy_features(obs: StepObservation, prev_obs: StepObservation | None) -> list[float]:
    # Always select the closest enemies, regardless of distance.
    candidates: list[tuple[float, ElementPos]] = []
    for enemy in obs.enemies:
        dx = enemy.x - obs.mario_x
        dy = enemy.y - obs.mario_y
        candidates.append((dx * dx + dy * dy, enemy))

    candidates.sort(key=lambda item: item[0])
    selected = [enemy for _, enemy in candidates[:ENEMY_COUNT]]

    features: list[float] = []
    used_prev_indices: set[int] = set()
    max_match_dist_px = 3.0 * TILE_SIZE
    max_match_dist_sq = max_match_dist_px * max_match_dist_px
    dt = 1
    if prev_obs is not None:
        try:
            dt = max(1, int(obs.step) - int(prev_obs.step))
        except Exception:
            dt = 1

    for enemy in selected:
        rel_dx = enemy.x - obs.mario_x
        rel_dy = enemy.y - obs.mario_y
        enemy_vel_x = 0.0
        enemy_vel_y = 0.0

        if prev_obs is not None and prev_obs.enemies:
            best_idx: int | None = None
            best_dist_sq: float = 0.0
            best_prev: ElementPos | None = None
            for idx, prev_enemy in enumerate(prev_obs.enemies):
                if idx in used_prev_indices:
                    continue
                if prev_enemy.type_id != enemy.type_id:
                    continue
                dx = enemy.x - prev_enemy.x
                dy = enemy.y - prev_enemy.y
                dist_sq = dx * dx + dy * dy
                if best_idx is None or dist_sq < best_dist_sq:
                    best_idx = idx
                    best_dist_sq = dist_sq
                    best_prev = prev_enemy

            if best_idx is not None and best_prev is not None and best_dist_sq <= max_match_dist_sq:
                used_prev_indices.add(best_idx)
                enemy_vel_x = (enemy.x - best_prev.x) / float(dt)
                enemy_vel_y = (enemy.y - best_prev.y) / float(dt)

        features.extend(
            [
                float(np.clip(rel_dx / MAX_MARIO_X, -1.0, 1.0)),
                float(np.clip(rel_dy / MAX_MARIO_Y, -1.0, 1.0)),
                float(np.clip(enemy_vel_x / MAX_VEL_X, -1.0, 1.0)),
                float(np.clip(enemy_vel_y / MAX_VEL_Y, -1.0, 1.0)),
                float(np.clip(float(enemy.type_id) / MAX_ENEMY_TYPE_ID, -1.0, 1.0)),
            ]
        )

    if len(features) < ENEMY_FEATURE_DIM:
        features.extend([0.0] * (ENEMY_FEATURE_DIM - len(features)))
    return features[:ENEMY_FEATURE_DIM]


def build_relative_tile_window(obs: StepObservation) -> list[int]:
    mario_tile_x = int(obs.mario_x / TILE_SIZE)
    mario_tile_y = int(obs.mario_y / TILE_SIZE)
    half_w = RELATIVE_TILE_WINDOW_WIDTH // 2
    half_h = RELATIVE_TILE_WINDOW_HEIGHT // 2

    tiles: list[int] = []
    for rel_y in range(-half_h, RELATIVE_TILE_WINDOW_HEIGHT - half_h):
        for rel_x in range(-half_w, RELATIVE_TILE_WINDOW_WIDTH - half_w):
            source_x = mario_tile_x + rel_x
            source_y = mario_tile_y + rel_y

            tile = 0
            if obs.scene_width > 0 and obs.scene_height > 0 and obs.scene_tiles:
                # The scene grid sent by the Java side is already centered on Mario.
                local_x = rel_x + (obs.scene_width // 2)
                local_y = rel_y + (obs.scene_height // 2)
                if 0 <= local_x < obs.scene_width and 0 <= local_y < obs.scene_height:
                    idx = local_y * obs.scene_width + local_x
                    if idx < len(obs.scene_tiles):
                        tile = obs.scene_tiles[idx]

            # If the centered scene does not cover the requested tile, fall back to
            # the sparse full-level tile map.
            if tile == 0 and CURRENT_CONTROLLER is not None and CURRENT_CONTROLLER.level_tile_map:
                tile = CURRENT_CONTROLLER.level_tile_map.get((source_x, source_y), 0)

            tiles.append(tile)

    if len(tiles) < MAX_SCENE_TILES:
        tiles.extend([0] * (MAX_SCENE_TILES - len(tiles)))
    return tiles[:MAX_SCENE_TILES]

TERMINAL_STATUSES = {"WIN", "LOSE", "TIME_OUT"}

# Calculate a heuristic reward bonus based on what A* agents suggest, to encourage following strong consensus when available
def heuristic_reward_bonus(
    action: list[int],
    obs: StepObservation,
    base_scale: float = 0.02,
    consensus_power: float = 2.0,
    decay_steps: float = 2000.0,
) -> float:
    if not obs.astar_actions:
        return 0.0

    agent_count = len(obs.astar_actions)
    if agent_count == 0 or obs.step > decay_steps:
        return 0.0

    action_bits = [bool(a) for a in action[:len(ACTION_ORDER)]]
    if len(action_bits) < len(ACTION_ORDER):
        action_bits.extend([False] * (len(ACTION_ORDER) - len(action_bits)))

    total_score = 0.0
    for idx in range(len(ACTION_ORDER)):
        votes = 0
        for agent_action in obs.astar_actions.values():
            if idx < len(agent_action) and agent_action[idx]:
                votes += 1
        fraction = votes / agent_count
        consensus_strength = abs(fraction - 0.5) * 2.0
        majority = fraction >= 0.5
        match = action_bits[idx] == majority
        direction = 1.0 if match else -1.0
        total_score += direction * (consensus_strength ** consensus_power)

    avg_score = total_score / len(ACTION_ORDER)
    time_factor = 1.0 / (1.0 + (obs.step / decay_steps))

    # print(f'Action: {[int(x) for x in action_bits]}, A* consensus fractions: {[sum(1 for a in obs.astar_actions.values() if idx < len(a) and a[idx]) / agent_count for idx in range(len(ACTION_ORDER))]}')
    # print(f"Step {obs.step}: Heuristic reward bonus={base_scale * time_factor * avg_score:.4f}  (consensus={avg_score:.4f}, time_factor={time_factor:.4f})")

    return base_scale * time_factor * avg_score


def compute_reward(action: list[int], prev: StepObservation, curr: StepObservation) -> float:
    if prev is None:
        return 0.0

    reward = 0.0

    # forward progress (exponential in delta-x)
    dx_tiles = (curr.mario_x - prev.mario_x) / float(TILE_SIZE)
    dx_tiles = float(np.clip(dx_tiles, -2.0, 2.0))
    # For small dx: expm1(dx) ≈ dx, so this behaves near-linear,
    # but rewards larger forward moves more strongly.
    reward += 0.8 * math.expm1(dx_tiles)

    # level completion shaping
    reward += (curr.completion - prev.completion) * 10.0

    # terminal failure penalty
    if curr.status == "LOSE":
        reward -= reward/2

    # win reward
    if curr.status == "WIN":
        reward = reward * 4

    if curr.status == "TIME_OUT":
        reward -= reward/4

    # small time penalty
    reward -= 0.05

    bonus = heuristic_reward_bonus(action, curr) * 10.0
    # print(f"Step {curr.step}: Base reward={reward:.4f}, Heuristic bonus={bonus:.4f}")
    reward += bonus

    return reward

def parse_bool(value: str) -> bool:
    return value.lower() in ("1", "true", "t", "yes", "y")


def parse_step(parts: list[str]) -> StepObservation:
    if len(parts) < 13:
        raise ValueError(f"STEP message has {len(parts)} fields, expected at least 13")

    enemies_raw = parts[13] if len(parts) > 13 else "-"
    sprites_raw = parts[14] if len(parts) > 14 else "-"
    scene_raw = parts[15] if len(parts) > 15 else "-"
    astar_raw = parts[16] if len(parts) > 16 else "-"
    scene_w, scene_h, scene_tiles = parse_scene_grid(scene_raw)
    astar_actions = parse_astar_actions(astar_raw)

    return StepObservation(
        step=int(parts[1]),
        mario_x=float(parts[2]),
        mario_y=float(parts[3]),
        vel_x=float(parts[4]),
        vel_y=float(parts[5]),
        mode=int(parts[6]),
        on_ground=parse_bool(parts[7]),
        may_jump=parse_bool(parts[8]),
        can_jump_higher=parse_bool(parts[9]),
        remaining_time=int(parts[10]),
        completion=float(parts[11]),
        status=parts[12],
        enemies=parse_element_positions(enemies_raw),
        sprites=parse_element_positions(sprites_raw),
        scene_width=scene_w,
        scene_height=scene_h,
        scene_tiles=scene_tiles,
        astar_actions=astar_actions,
    )


def parse_level(parts: list[str]) -> LevelData:
    if len(parts) < 4:
        raise ValueError(f"LEVEL message has {len(parts)} fields, expected at least 4")

    width = int(parts[1])
    height = int(parts[2])
    blocks_raw = parts[3]

    blocks: list[LevelBlock] = []
    if blocks_raw and blocks_raw != "-":
        for item in blocks_raw.split(";"):
            if not item:
                continue
            coords = item.split(",")
            if len(coords) != 3:
                continue
            try:
                blocks.append(LevelBlock(x=int(coords[0]), y=int(coords[1]), tile_id=int(coords[2])))
            except ValueError:
                continue
    return LevelData(width=width, height=height, blocks=blocks)


def parse_element_positions(raw: str) -> list[ElementPos]:
    if not raw or raw == "-":
        return []

    parsed: list[ElementPos] = []
    for item in raw.split(";"):
        if not item:
            continue
        parts = item.split(",")
        if len(parts) != 3:
            continue
        try:
            parsed.append(
                ElementPos(
                    type_id=int(float(parts[0])),
                    x=float(parts[1]),
                    y=float(parts[2]),
                )
            )
        except ValueError:
            continue
    return parsed


def parse_scene_grid(raw: str) -> tuple[int, int, list[int]]:
    if not raw or raw == "-" or ":" not in raw:
        return 0, 0, []

    dims, data = raw.split(":", maxsplit=1)
    if "x" not in dims:
        return 0, 0, []

    try:
        w_str, h_str = dims.split("x", maxsplit=1)
        width = int(w_str)
        height = int(h_str)
    except ValueError:
        return 0, 0, []

    if not data:
        return width, height, []

    tiles: list[int] = []
    for value in data.split(","):
        if not value:
            continue
        try:
            tiles.append(int(value))
        except ValueError:
            tiles.append(0)

    return width, height, tiles


def parse_astar_actions(raw: str) -> dict[str, list[bool]]:
    if not raw or raw == "-":
        return {}

    parsed: dict[str, list[bool]] = {}
    for item in raw.split(";"):
        if not item or "=" not in item:
            continue
        agent_id, values_raw = item.split("=", maxsplit=1)
        agent_id = agent_id.strip()
        if not agent_id:
            continue
        values: list[bool] = []
        for token in values_raw.split(","):
            if token == "":
                continue
            values.append(parse_bool(token))
        if len(values) < len(ACTION_ORDER):
            values.extend([False] * (len(ACTION_ORDER) - len(values)))
        else:
            values = values[:len(ACTION_ORDER)]
        parsed[agent_id] = values

    return parsed


def has_block_ahead(obs: StepObservation) -> bool:
    if obs.scene_width <= 0 or obs.scene_height <= 0:
        return False

    center_x = obs.scene_width // 2
    center_y = obs.scene_height // 2
    target_positions = [
        (center_x + 1, center_y),
        (center_x + 1, center_y - 1),
        (center_x + 2, center_y),
        (center_x + 2, center_y - 1),
    ]

    for tx, ty in target_positions:
        if tx < 0 or ty < 0 or tx >= obs.scene_width or ty >= obs.scene_height:
            continue
        idx = ty * obs.scene_width + tx
        if idx >= len(obs.scene_tiles):
            continue
        tile = obs.scene_tiles[idx]
        if tile in (OBS_BRICK, OBS_QUESTION_BLOCK, OBS_USED_BLOCK):
            return True

    return False


def actions_to_line(actions: list[bool]) -> str:
    return " ".join("1" if a else "0" for a in actions)


def serve(
    host: str,
    port: int,
    model_path: str | None = None,
    stats_path: str | None = None,
    save_every: int = 1,
    tensorboard_dir: str | None = None,
    jump_control: str = "click",
) -> None:
    controller = MarioPythonController(
        model_path=model_path,
        stats_path=stats_path,
        save_every=save_every,
        tensorboard_dir=tensorboard_dir,
        jump_control=jump_control,
    )

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((host, port))
        server.listen(1)
        print(f"Python controller listening on {host}:{port}")

        while True:
            conn, addr = server.accept()
            with conn:
                print(f"Java framework connected from {addr[0]}:{addr[1]}")
                reader = conn.makefile("r", encoding="utf-8", newline="\n")
                writer = conn.makefile("w", encoding="utf-8", newline="\n")

                while True:
                    line = reader.readline()
                    if not line:
                        print("Java side closed connection")
                        break

                    try:
                        parts = line.strip().split("\t")
                        if not parts:
                            continue

                        tag = parts[0]
                        if tag == "HELLO":
                            print("Handshake received:", line.strip())
                            continue

                        if tag == "LEVEL":
                            level_data = parse_level(parts)
                            controller.set_level_data(level_data)
                            print(
                                f"Level received: {level_data.width}x{level_data.height}, "
                                f"tracked blocks={len(level_data.blocks)}"
                            )
                            continue

                        if tag == "STEP":
                            obs = parse_step(parts)
                            actions = controller.choose_actions(obs)
                            writer.write(actions_to_line(actions) + "\n")
                            writer.flush()
                            continue

                        if tag == "END":
                            print("Game ended:", line.strip())
                            controller.handle_episode_end(parts)
                            continue

                        print("Unknown message:", line.strip())
                    except Exception:
                        print("ERROR: Exception while handling message from Java:")
                        print(line.strip())
                        traceback.print_exc()
                        # Always try to respond to STEP to avoid Java-side socket reset.
                        try:
                            parts = line.strip().split("\t")
                            if parts and parts[0] == "STEP":
                                writer.write(actions_to_line(controller.resolve_action_bits(controller.act_dim - 1)) + "\n")
                                writer.flush()
                        except Exception:
                            pass
                        continue


def main() -> None:
    parser = argparse.ArgumentParser(description="Python-side Mario controller server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5050)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--stats-path", default=None)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--tensorboard-dir", default=None)
    parser.add_argument("--jump-control", default="click", choices=["click", "press-release"])
    args = parser.parse_args()

    serve(
        args.host,
        args.port,
        model_path=args.model_path,
        stats_path=args.stats_path,
        save_every=args.save_every,
        tensorboard_dir=args.tensorboard_dir,
        jump_control=args.jump_control,
    )


if __name__ == "__main__":
    main()
