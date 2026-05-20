"""
Human demo recorder.
Play the game with keyboard and record (observation, action) pairs for
behavioral cloning.

Controls:
  Right arrow / D  : run right
  Left arrow  / A  : left
  Space / Up arrow : jump
  Hold Shift       : also hold speed (combined with right = fast run+jump)

Press ESC to stop recording early.
"""
import argparse
import ctypes
import json
import socket
import traceback
from pathlib import Path

from controller import (
    obs_to_vector,
    parse_level,
    parse_step,
    actions_to_line,
)

# ── Windows key-state reader ──────────────────────────────────────────────────
_user32 = ctypes.windll.user32

VK = {
    "LEFT":  0x25,
    "RIGHT": 0x27,
    "DOWN":  0x28,
    "UP":    0x26,
    "SPACE": 0x20,
    "SHIFT": 0x10,
    "A":     0x41,
    "D":     0x44,
    "ESC":   0x1B,
}

def key_down(vk: int) -> bool:
    return bool(_user32.GetAsyncKeyState(vk) & 0x8000)

def read_action_bits() -> list[int]:
    """Return [LEFT, RIGHT, DOWN, SPEED, JUMP] bits from current key state."""
    left  = key_down(VK["LEFT"])  or key_down(VK["A"])
    right = key_down(VK["RIGHT"]) or key_down(VK["D"])
    down  = key_down(VK["DOWN"])
    speed = key_down(VK["SHIFT"])
    jump  = key_down(VK["SPACE"]) or key_down(VK["UP"])
    return [int(left), int(right), int(down), int(speed), int(jump)]

# Map raw action bits to the nearest discrete action index used by the PPO agent.
ACTION_TABLE = [
    [0, 1, 0, 1, 0],  # 0: run right
    [0, 1, 0, 1, 1],  # 1: jump right
    [1, 0, 0, 0, 0],  # 2: left
    [0, 0, 0, 0, 0],  # 3: idle
]

def bits_to_action_idx(bits: list[int]) -> int:
    best, best_dist = 0, 999
    for idx, row in enumerate(ACTION_TABLE):
        dist = sum(a != b for a, b in zip(bits, row))
        if dist < best_dist:
            best, best_dist = idx, dist
    return best

# ── Demo server ───────────────────────────────────────────────────────────────

def serve(host: str, port: int, demo_path: Path, sessions: int) -> None:
    demos: list[dict] = []
    episode = 0

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((host, port))
        server.listen(1)
        print(f"Human demo server on {host}:{port}")
        print("Controls: Arrow keys / WASD to move, Space to jump, Shift to run")
        print(f"Recording to: {demo_path}")
        print()

        while episode < sessions:
            conn, addr = server.accept()
            with conn:
                print(f"Java connected from {addr[0]}:{addr[1]}")
                reader = conn.makefile("r", encoding="utf-8", newline="\n")
                writer = conn.makefile("w", encoding="utf-8", newline="\n")

                while True:
                    line = reader.readline()
                    if not line:
                        break
                    try:
                        parts = line.strip().split("\t")
                        if not parts:
                            continue
                        tag = parts[0]

                        if tag == "HELLO":
                            continue

                        if tag == "LEVEL":
                            level_data = parse_level(parts)
                            print(f"Level: {level_data.width}x{level_data.height}")
                            continue

                        if tag == "STEP":
                            if key_down(VK["ESC"]):
                                print("ESC pressed — stopping early.")
                                break

                            obs = parse_step(parts)
                            state = obs_to_vector(obs)
                            bits = read_action_bits()
                            action_idx = bits_to_action_idx(bits)

                            demos.append({
                                "state": state.tolist(),
                                "action": action_idx,
                            })

                            writer.write(actions_to_line(bits) + "\n")
                            writer.flush()
                            continue

                        if tag == "END":
                            episode += 1
                            status = parts[1] if len(parts) > 1 else "?"
                            completion = float(parts[2]) if len(parts) > 2 else 0.0
                            print(f"Episode {episode}/{sessions}: {status}  completion={completion:.3f}  "
                                  f"recorded {len(demos)} steps total")
                            continue

                    except Exception:
                        traceback.print_exc()
                        try:
                            parts = line.strip().split("\t")
                            if parts and parts[0] == "STEP":
                                writer.write(actions_to_line([0, 1, 0, 1, 0]) + "\n")
                                writer.flush()
                        except Exception:
                            pass

    # Save demos
    demo_path.parent.mkdir(parents=True, exist_ok=True)
    existing: list[dict] = []
    if demo_path.exists():
        with demo_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    existing.append(json.loads(line))
    with demo_path.open("w", encoding="utf-8") as f:
        for d in existing + demos:
            f.write(json.dumps(d) + "\n")
    print(f"\nSaved {len(demos)} new steps ({len(existing) + len(demos)} total) to {demo_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Human demo recorder for Mario")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5050)
    parser.add_argument("--demo-path", default="demos/human_demos.jsonl")
    parser.add_argument("--sessions", type=int, default=10)
    args = parser.parse_args()

    serve(args.host, args.port, Path(args.demo_path), args.sessions)


if __name__ == "__main__":
    main()
