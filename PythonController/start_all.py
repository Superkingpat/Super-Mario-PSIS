import argparse
import socket
import subprocess
import sys
import time
from pathlib import Path


def wait_for_port(host: str, port: int, timeout_s: float) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.5):
                return True
        except OSError:
            time.sleep(0.1)
    return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Start Python controller and Java Mario game together"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5050)
    parser.add_argument("--level", default="./levels/original/lvl-1.txt")
    parser.add_argument("--timer", type=int, default=200)
    parser.add_argument("--mario-state", type=int, default=0, choices=[0, 1, 2])
    parser.add_argument("--visuals", default="true", choices=["true", "false"])
    parser.add_argument("--no-compile", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    controller_path = repo_root / "PythonController" / "controller.py"
    launcher_path = repo_root / "PythonController" / "start_java_game.py"

    controller_cmd = [
        sys.executable,
        str(controller_path),
        "--host",
        args.host,
        "--port",
        str(args.port),
    ]

    java_cmd = [
        sys.executable,
        str(launcher_path),
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--level",
        args.level,
        "--timer",
        str(args.timer),
        "--mario-state",
        str(args.mario_state),
        "--visuals",
        args.visuals,
    ]
    if args.no_compile:
        java_cmd.append("--no-compile")

    print("Starting Python controller:", " ".join(controller_cmd))
    controller_proc = subprocess.Popen(controller_cmd, cwd=str(repo_root))

    try:
        if not wait_for_port(args.host, args.port, timeout_s=10.0):
            print(f"Controller did not start on {args.host}:{args.port} in time.")
            return 1

        print("Starting Java game:", " ".join(java_cmd))
        java_result = subprocess.run(java_cmd, cwd=str(repo_root), check=False)
        return java_result.returncode
    finally:
        if controller_proc.poll() is None:
            controller_proc.terminate()
            try:
                controller_proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                controller_proc.kill()


if __name__ == "__main__":
    sys.exit(main())
