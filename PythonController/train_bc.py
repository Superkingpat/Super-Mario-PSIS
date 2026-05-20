"""
Behavioral cloning: train the PPO policy to imitate recorded human demos.
Run this after recording demos with human_demo.py.

Usage:
    python train_bc.py --demo-path demos/human_demos.jsonl --model-path checkpoints/ppo_o.pt
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ppo_agent import PPOAgent


def load_demos(demo_path: Path) -> tuple[np.ndarray, np.ndarray]:
    states, actions = [], []
    with demo_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            states.append(d["state"])
            actions.append(d["action"])
    return np.array(states, dtype=np.float32), np.array(actions, dtype=np.int64)


def train_bc(
    demo_path: Path,
    model_path: Path | None,
    epochs: int,
    batch_size: int,
    lr: float,
) -> None:
    print(f"Loading demos from {demo_path}")
    states, actions = load_demos(demo_path)
    print(f"  {len(states)} steps, action distribution: "
          + str({i: int((actions == i).sum()) for i in range(4)}))

    tile_dim = 512  # MAX_SCENE_TILES = 32*16
    scalar_dim = states.shape[1] - tile_dim
    act_dim = 4  # click action space

    agent = PPOAgent(scalar_dim, tile_dim, act_dim)
    if model_path and model_path.exists():
        agent.load(model_path)
        print(f"Loaded existing checkpoint from {model_path} (will fine-tune)")
    else:
        print("Starting from random weights")

    device = agent.device
    dataset = TensorDataset(
        torch.tensor(states).to(device),
        torch.tensor(actions).to(device),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Use a fresh optimizer with a higher LR for BC pre-training
    optimizer = torch.optim.Adam(agent.policy.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    print(f"\nTraining for {epochs} epochs  (batch={batch_size}, lr={lr})")
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        correct = 0
        for b_states, b_actions in loader:
            logits, _ = agent.policy(b_states)
            loss = criterion(logits, b_actions)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.policy.parameters(), 0.5)
            optimizer.step()
            total_loss += loss.item() * len(b_states)
            correct += (logits.argmax(dim=1) == b_actions).sum().item()

        avg_loss = total_loss / len(states)
        acc = correct / len(states) * 100
        print(f"  Epoch {epoch:3d}/{epochs}  loss={avg_loss:.4f}  acc={acc:.1f}%")

    if model_path:
        agent.save(model_path)
        print(f"\nSaved to {model_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Behavioral cloning from human demos")
    parser.add_argument("--demo-path", default="demos/human_demos.jsonl")
    parser.add_argument("--model-path", default=None,
                        help="Checkpoint to fine-tune (or start fresh if absent)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    train_bc(
        demo_path=Path(args.demo_path),
        model_path=Path(args.model_path) if args.model_path else None,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )


if __name__ == "__main__":
    main()
