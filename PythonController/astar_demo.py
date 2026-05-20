"""
Collect winning demos by following the A* agents sent by the Java framework.
Run this instead of controller.py to gather training data automatically.

Usage:
    python astar_demo.py --demo-path demos/o_astar_demos.jsonl --sessions 200
"""
import argparse
import json
import socket
import traceback
from pathlib import Path

from controller import (
    ACTION_ORDER,
    MarioPythonController,
    actions_to_line,
    obs_to_vector,
    parse_level,
    parse_step,
)

# ── action table (must match controller.py) ─────────────────────────────────
ACTION_TABLE = [
    [0, 1, 0, 1, 0],  # 0 run right
    [0, 1, 0, 1, 1],  # 1 jump right
    [1, 0, 0, 0, 0],  # 2 left
    [0, 0, 0, 0, 0],  # 3 idle
]


def astar_majority_bits(astar_actions: dict) -> list[int]:
    """Majority-vote across all A* agents → 5-bit action vector."""
    if not astar_actions:
        return [0, 1, 0, 1, 0]  # fallback: run right

    n = len(astar_actions)
    votes = [0] * len(ACTION_ORDER)
    for agent_acts in astar_actions.values():
        for i, a in enumerate(agent_acts[: len(ACTION_ORDER)]):
            if a:
                votes[i] += 1

    return [1 if v * 2 > n else 0 for v in votes]


def bits_to_discrete(bits: list[int]) -> int:
    """Map 5-bit action to nearest discrete action index (Hamming distance)."""
    best_idx, best_dist = 0, float("inf")
    for i, row in enumerate(ACTION_TABLE):
        dist = sum(a != b for a, b in zip(row, bits[:5]))
        if dist < best_dist:
            best_dist, best_idx = dist, i
    return best_idx


def serve(host: str, port: int, demo_path: Path, sessions: int) -> None:
    demo_path.parent.mkdir(parents=True, exist_ok=True)

    wins = 0
    total_steps_saved = 0

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((host, port))
        server.listen(1)
        print(f"A* demo recorder listening on {host}:{port}")
        print(f"Saving winning episodes to {demo_path}")

        episode = 0

        while episode < sessions:
            conn, addr = server.accept()
            with conn:
                print(f"Java connected from {addr[0]}:{addr[1]}")
                reader = conn.makefile("r", encoding="utf-8", newline="\n")
                writer = conn.makefile("w", encoding="utf-8", newline="\n")

                episode_buf: list[dict] = []  # (state_vec, action_idx)

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
                            parse_level(parts)  # ignore; keep for future use
                            continue

                        if tag == "STEP":
                            obs = parse_step(parts)

                            # Always output A* majority vote
                            bits = astar_majority_bits(obs.astar_actions)
                            action_bits = [bool(b) for b in bits]
                            action_idx = bits_to_discrete(bits)

                            state_vec = obs_to_vector(obs).tolist()
                            episode_buf.append(
                                {"state": state_vec, "action": action_idx}
                            )

                            writer.write(actions_to_line(action_bits) + "\n")
                            writer.flush()
                            continue

                        if tag == "END":
                            episode += 1
                            status = parts[1] if len(parts) > 1 else "UNKNOWN"
                            completion = float(parts[2]) if len(parts) > 2 else 0.0

                            if status == "WIN":
                                wins += 1
                                total_steps_saved += len(episode_buf)
                                with demo_path.open("a", encoding="utf-8") as f:
                                    for step in episode_buf:
                                        f.write(json.dumps(step) + "\n")
                                print(
                                    f"Episode {episode}/{sessions}  WIN  "
                                    f"steps={len(episode_buf)}  "
                                    f"total_wins={wins}  "
                                    f"total_steps={total_steps_saved}"
                                )
                            else:
                                print(
                                    f"Episode {episode}/{sessions}  {status}  "
                                    f"completion={completion:.3f}"
                                )

                            episode_buf.clear()
                            continue

                    except Exception:
                        print("ERROR handling message:")
                        traceback.print_exc()
                        try:
                            if parts and parts[0] == "STEP":
                                writer.write(actions_to_line([0, 1, 0, 1, 0]) + "\n")
                                writer.flush()
                        except Exception:
                            pass

    print(f"\nDone. Collected {wins} winning episodes, {total_steps_saved} steps.")
    print(f"Saved to {demo_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect A* demonstrations")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5050)
    parser.add_argument("--demo-path", default="demos/o_astar_demos.jsonl")
    parser.add_argument("--sessions", type=int, default=200)
    args = parser.parse_args()

    serve(
        args.host,
        args.port,
        demo_path=Path(args.demo_path),
        sessions=args.sessions,
    )


if __name__ == "__main__":
    main()
