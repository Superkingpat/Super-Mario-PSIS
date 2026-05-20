"""
Evolution controller — Java bridge + PPOEvoAgent.

Imports all Java protocol parsing from the parent PythonController/controller.py
so the protocol code stays in one place. Only the agent and serve() are overridden.
"""

import argparse
import json
import math
import socket
import sys
import time
import traceback
from pathlib import Path

# Resolve parent PythonController so we can reuse its protocol layer
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "PythonController"))

import controller as _proto  # noqa: E402  (import after sys.path patch)
from controller import (
    StepObservation,
    LevelData,
    LevelBlock,
    ElementPos,
    obs_to_vector,
    compute_reward,
    parse_step,
    parse_level,
    actions_to_line,
    TERMINAL_STATUSES,
    STUCK_NO_PROGRESS_EPS_PX,
    STUCK_STEPS_GRACE,
    STUCK_STEPS_RAMP,
    STUCK_PENALTY_MAX,
)
from ppo_evo_agent import PPOEvoAgent  # noqa: E402

# ── Enemy proximity penalty ─────────────────────────────────────────────────
# The agent dies ~25% of the time at the ~10% completion mark — a fixed enemy.
# Penalise each step where Mario is dangerously close to any enemy so the agent
# learns to avoid (or jump over) that obstacle instead of walking into it.
_ENEMY_DANGER_PX   = 64.0   # 4 tiles — danger zone radius
_ENEMY_PENALTY_MAX = 0.4    # reward subtracted when touching-distance

def _enemy_proximity_penalty(obs) -> float:
    if not obs.enemies:
        return 0.0
    min_dist_sq = float("inf")
    for e in obs.enemies:
        dx = e.x - obs.mario_x
        dy = e.y - obs.mario_y
        min_dist_sq = min(min_dist_sq, dx * dx + dy * dy)
    dist = math.sqrt(min_dist_sq)
    if dist >= _ENEMY_DANGER_PX:
        return 0.0
    # Linear ramp: 0 at danger edge, -_ENEMY_PENALTY_MAX at dist=0
    return -_ENEMY_PENALTY_MAX * (1.0 - dist / _ENEMY_DANGER_PX)

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None


class MarioPythonControllerEvo:
    def __init__(
        self,
        model_path: str | None = None,
        stats_path: str | None = None,
        save_every: int = 1,
        tensorboard_dir: str | None = None,
    ) -> None:
        # Register as the active controller so build_relative_tile_window works
        _proto.CURRENT_CONTROLLER = self

        self.level_data = None
        self.level_tile_map: dict[tuple[int, int], int] = {}

        scalar_dim = 10 + _proto.ENEMY_FEATURE_DIM  # 25
        tile_dim = _proto.MAX_SCENE_TILES            # 512

        # click action table — same as original
        self._action_table = [
            [0, 1, 0, 1, 0],  # run right
            [0, 1, 0, 1, 1],  # jump right
            [1, 0, 0, 0, 0],  # left
            [0, 0, 0, 0, 0],  # idle
        ]
        act_dim = len(self._action_table)

        self.agent = PPOEvoAgent(scalar_dim, tile_dim, act_dim)
        self.model_path = Path(model_path) if model_path else None
        self.best_model_path = self.model_path.with_suffix(".best.pt") if self.model_path else None
        self.stats_path = Path(stats_path) if stats_path else None
        self.save_every = max(1, save_every)
        self.tensorboard_dir = Path(tensorboard_dir) if tensorboard_dir else None
        self.tensorboard_writer = None

        self.update_interval = self.agent.rollout_steps
        self.step_counter = 0
        self.episode_reward = 0.0
        self.episode_steps = 0
        self.episode_jump_reward = 0.0
        self.episode_count = 0
        self.update_count = 0
        self.best_completion = 0.0

        self._stuck_steps = 0
        self.prev_obs: StepObservation | None = None
        self.pending_transition = None

        if self.tensorboard_dir:
            if SummaryWriter is None:
                print("WARNING: tensorboard not installed — skipping TensorBoard logging.")
            else:
                self.tensorboard_dir.mkdir(parents=True, exist_ok=True)
                self.tensorboard_writer = SummaryWriter(log_dir=str(self.tensorboard_dir))
                print(f"TensorBoard: {self.tensorboard_dir}")

        if self.model_path and self.model_path.exists():
            try:
                self.agent.load(self.model_path)
                print(f"Loaded checkpoint from {self.model_path}")
            except Exception as exc:
                backup = self.model_path.with_suffix(
                    self.model_path.suffix + f".incompatible_{int(time.time())}"
                )
                try:
                    self.model_path.rename(backup)
                except OSError:
                    pass
                print(f"WARNING: incompatible checkpoint, starting fresh. ({exc})")

    # ------------------------------------------------------------------
    # Level / step handling
    # ------------------------------------------------------------------

    def set_level_data(self, level_data: LevelData) -> None:
        self.level_data = level_data
        self.level_tile_map = {(b.x, b.y): b.tile_id for b in level_data.blocks}

    def _resolve_action(self, idx: int) -> list[int]:
        if 0 <= idx < len(self._action_table):
            return self._action_table[idx]
        return self._action_table[-1]

    def _compute_stuck_penalty(self, prev: StepObservation, curr: StepObservation) -> float:
        dx = curr.mario_x - prev.mario_x
        if abs(dx) <= STUCK_NO_PROGRESS_EPS_PX:
            self._stuck_steps += 1
        else:
            self._stuck_steps = 0
        if self._stuck_steps <= STUCK_STEPS_GRACE:
            return 0.0
        over = self._stuck_steps - STUCK_STEPS_GRACE
        ramp = min(1.0, over / float(STUCK_STEPS_RAMP))
        return -STUCK_PENALTY_MAX * ramp

    def _compute_jump_bonus(self, action: list[int], prev: StepObservation, curr: StepObservation) -> float:
        jump_pressed = bool(action[4]) if len(action) > 4 else False
        jump_started = jump_pressed and prev.on_ground and not curr.on_ground
        if not jump_started or self.episode_jump_reward >= 50.0:
            return 0.0
        bonus = min(1.0, 50.0 - self.episode_jump_reward)
        self.episode_jump_reward += bonus
        return bonus

    def choose_actions(self, obs: StepObservation) -> list[int]:
        state = obs_to_vector(obs)
        done = obs.status in TERMINAL_STATUSES

        if self.pending_transition is not None and self.prev_obs is not None:
            prev_state, prev_idx, prev_bits, prev_lp, prev_val = self.pending_transition
            reward = compute_reward(prev_bits, self.prev_obs, obs)
            reward += self._compute_jump_bonus(prev_bits, self.prev_obs, obs)
            reward += self._compute_stuck_penalty(self.prev_obs, obs)
            reward += _enemy_proximity_penalty(obs)
            self.agent.store((prev_state, prev_idx, reward, done, prev_lp, prev_val))
            self.episode_reward += reward
            self.episode_steps += 1

        if done:
            self.pending_transition = None
            self.prev_obs = obs
            self._stuck_steps = 0
            return self._resolve_action(len(self._action_table) - 1)

        action_idx, log_prob, value = self.agent.act(state)
        action_bits = self._resolve_action(action_idx)
        self.pending_transition = (state, action_idx, action_bits, log_prob, value)
        self.prev_obs = obs
        self.step_counter += 1

        if self.step_counter % self.update_interval == 0 and self.agent.memory:
            metrics = self.agent.update(next_value=value)
            self._log_update(metrics)

        return action_bits

    def handle_episode_end(self, end_parts: list[str]) -> None:
        status = end_parts[1] if len(end_parts) > 1 else "UNKNOWN"
        completion = float(end_parts[2]) if len(end_parts) > 2 else 0.0
        remaining_time = int(end_parts[3]) if len(end_parts) > 3 else 0
        jumps = int(end_parts[4]) if len(end_parts) > 4 else 0
        kills = int(end_parts[5]) if len(end_parts) > 5 else 0
        coins = int(end_parts[6]) if len(end_parts) > 6 else 0

        # Flush last transition
        if self.pending_transition is not None and self.prev_obs is not None:
            terminal_obs = StepObservation(
                step=self.prev_obs.step + 1,
                mario_x=self.prev_obs.mario_x,
                mario_y=self.prev_obs.mario_y,
                vel_x=0.0, vel_y=0.0,
                mode=self.prev_obs.mode,
                on_ground=self.prev_obs.on_ground,
                may_jump=self.prev_obs.may_jump,
                can_jump_higher=0,
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
            prev_state, prev_idx, prev_bits, prev_lp, prev_val = self.pending_transition
            reward = compute_reward(prev_bits, self.prev_obs, terminal_obs)
            reward += self._compute_jump_bonus(prev_bits, self.prev_obs, terminal_obs)
            reward += self._compute_stuck_penalty(self.prev_obs, terminal_obs)
            reward += _enemy_proximity_penalty(terminal_obs)
            self.agent.store((prev_state, prev_idx, reward, True, prev_lp, prev_val))
            self.episode_reward += reward
            self.episode_steps += 1

        if self.agent.memory:
            import torch
            if status == "TIME_OUT" and self.prev_obs is not None:
                last_state = obs_to_vector(self.prev_obs)
                t = torch.tensor(last_state, dtype=torch.float32).to(self.agent.device)
                with torch.no_grad():
                    _, last_val = self.agent.policy(t)
                next_value = last_val.item()
            else:
                next_value = 0.0
            metrics = self.agent.update(next_value=next_value)
            self._log_update(metrics)

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
            f"Episode {self.episode_count}: status={status} "
            f"completion={completion:.5f} reward={self.episode_reward:.3f} "
            f"steps={self.episode_steps}"
        )

        if self.stats_path:
            self.stats_path.parent.mkdir(parents=True, exist_ok=True)
            with self.stats_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(episode_stats) + "\n")

        if self.tensorboard_writer is not None:
            w = self.tensorboard_writer
            ep = self.episode_count
            w.add_scalar("episode/reward", self.episode_reward, ep)
            w.add_scalar("episode/completion", completion, ep)
            w.add_scalar("episode/remaining_time", remaining_time, ep)
            w.add_scalar("episode/jumps", jumps, ep)
            w.add_scalar("episode/kills", kills, ep)
            w.add_scalar("episode/coins", coins, ep)
            w.add_scalar("episode/steps", self.episode_steps, ep)
            w.add_text("episode/status", status, ep)
            w.flush()

        if self.model_path and self.episode_count % self.save_every == 0:
            self.agent.save(self.model_path)
            print(f"Saved checkpoint to {self.model_path}")

        if completion > self.best_completion and self.best_model_path:
            self.best_completion = completion
            self.agent.save(self.best_model_path)
            print(f"*** New best {completion:.4f} → {self.best_model_path}")

        self.prev_obs = None
        self.pending_transition = None
        self.episode_reward = 0.0
        self.episode_steps = 0
        self.episode_jump_reward = 0.0
        self._stuck_steps = 0

    def _log_update(self, metrics: dict | None) -> None:
        if not metrics:
            return
        self.update_count += 1
        if self.tensorboard_writer is not None:
            w = self.tensorboard_writer
            u = self.update_count
            w.add_scalar("train/loss", metrics["loss"], u)
            w.add_scalar("train/policy_loss", metrics["policy_loss"], u)
            w.add_scalar("train/value_loss", metrics["value_loss"], u)
            w.add_scalar("train/entropy", metrics["entropy"], u)
            w.add_scalar("train/batch_size", metrics["batch_size"], u)
            w.add_scalar("train/environment_steps", self.step_counter, u)
            w.add_scalar("train/lr", metrics.get("lr", 0.0), u)
            w.add_scalar("train/entropy_coef", metrics.get("entropy_coef", 0.0), u)
            w.flush()


# ---------------------------------------------------------------------------
# Socket server
# ---------------------------------------------------------------------------

def serve(
    host: str,
    port: int,
    model_path: str | None = None,
    stats_path: str | None = None,
    save_every: int = 1,
    tensorboard_dir: str | None = None,
) -> None:
    controller = MarioPythonControllerEvo(
        model_path=model_path,
        stats_path=stats_path,
        save_every=save_every,
        tensorboard_dir=tensorboard_dir,
    )

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((host, port))
        server.listen(1)
        print(f"Evolution controller listening on {host}:{port}")

        while True:
            conn, addr = server.accept()
            with conn:
                print(f"Java connected from {addr[0]}:{addr[1]}")
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
                            print("Handshake:", line.strip())
                        elif tag == "LEVEL":
                            lvl = parse_level(parts)
                            controller.set_level_data(lvl)
                            print(f"Level: {lvl.width}x{lvl.height}, blocks={len(lvl.blocks)}")
                        elif tag == "STEP":
                            obs = parse_step(parts)
                            actions = controller.choose_actions(obs)
                            writer.write(actions_to_line(actions) + "\n")
                            writer.flush()
                        elif tag == "END":
                            print("End:", line.strip())
                            controller.handle_episode_end(parts)
                        else:
                            print("Unknown:", line.strip())

                    except Exception:
                        print("ERROR handling Java message:")
                        print(line.strip())
                        traceback.print_exc()
                        try:
                            parts = line.strip().split("\t")
                            if parts and parts[0] == "STEP":
                                writer.write(actions_to_line(controller._resolve_action(len(controller._action_table) - 1)) + "\n")
                                writer.flush()
                        except Exception:
                            pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Mario Evolution controller")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5050)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--stats-path", default=None)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--tensorboard-dir", default=None)
    args = parser.parse_args()

    serve(
        args.host,
        args.port,
        model_path=args.model_path,
        stats_path=args.stats_path,
        save_every=args.save_every,
        tensorboard_dir=args.tensorboard_dir,
    )


if __name__ == "__main__":
    main()
