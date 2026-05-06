import argparse
import socket
from dataclasses import dataclass
import numpy as np
from ppo_agent import PPOAgent

ACTION_ORDER = ["LEFT", "RIGHT", "DOWN", "SPEED", "JUMP"]

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
    def __init__(self) -> None:
        self.level_data = None

        self.prev_obs = None

        self.obs_dim = 2 + 2 + 2 + 200  # tune this
        self.act_dim = 4  # discrete actions

        self.agent = PPOAgent(self.obs_dim, self.act_dim)

        self.step_counter = 0
        self.update_interval = 2048

    def set_level_data(self, level_data):
        self.level_data = level_data

    def choose_actions(self, obs: StepObservation):
        state = obs_to_vector(obs)

        action_idx, log_prob, value = self.agent.act(state)
        action = ACTIONS[action_idx]

        reward = compute_reward(action, self.prev_obs, obs)
        done = obs.status in ("DEAD", "WIN")

        if self.prev_obs is not None:
            self.agent.store((
                state,
                action_idx,
                reward,
                done,
                log_prob,
                value
            ))

        self.prev_obs = obs
        self.step_counter += 1

        # PPO update
        if self.step_counter % self.update_interval == 0:
            next_value = 0 if done else value
            self.agent.update(next_value)

        return action


def obs_to_vector(obs: StepObservation):
    tiles = obs.scene_tiles[:200]
    if len(tiles) < 200:
        tiles = tiles + [0] * (200 - len(tiles))

    return np.array([
        obs.mario_x,
        obs.mario_y,
        obs.vel_x,
        obs.vel_y,
        float(obs.on_ground),
        float(obs.may_jump),
        *tiles
    ], dtype=np.float32)

ACTIONS = [
    [0,1,0,1,0],  # run right
    [0,1,0,1,1],  # jump right
    [1,0,0,0,0],  # left
    [0,0,0,0,0],  # idle
]

# Calculate a heuristic reward bonus based on what A* agents suggest, to encourage following strong consensus when available
def heuristic_reward_bonus(
    action: list[int],
    obs: StepObservation,
    base_scale: float = 0.2,
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

    # forward progress
    reward += (curr.mario_x - prev.mario_x) * 0.1

    # death penalty
    if curr.status == "DEAD":
        reward -= 50

    # win reward
    if curr.status == "WIN":
        reward += 100

    # small time penalty
    reward -= 0.01

    bonus = heuristic_reward_bonus(action, curr)
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
    print(blocks)
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


def serve(host: str, port: int) -> None:
    controller = MarioPythonController()

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
                        controller.prev_obs = None
                        continue

                    print("Unknown message:", line.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description="Python-side Mario controller server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5050)
    args = parser.parse_args()

    serve(args.host, args.port)


if __name__ == "__main__":
    main()
