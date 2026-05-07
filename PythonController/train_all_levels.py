import argparse
import json
import socket
import subprocess
import sys
import time
from pathlib import Path


def choose_port(host: str, preferred_port: int) -> int:
    """Return preferred_port if it appears free, otherwise return an ephemeral free port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        # On Windows, SO_REUSEADDR can still allow binds even when another process
        # is listening. Prefer exclusive binds when available.
        exclusive = getattr(socket, "SO_EXCLUSIVEADDRUSE", None)
        if exclusive is not None:
            try:
                probe.setsockopt(socket.SOL_SOCKET, exclusive, 1)
            except OSError:
                pass
        try:
            probe.bind((host, int(preferred_port)))
            return int(preferred_port)
        except OSError:
            probe.bind((host, 0))
            return int(probe.getsockname()[1])


def wait_for_port(host: str, port: int, timeout_s: float, proc: subprocess.Popen | None = None) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if proc is not None and proc.poll() is not None:
            return False
        try:
            with socket.create_connection((host, port), timeout=0.5):
                return True
        except OSError:
            time.sleep(0.1)
    return False


def natural_level_key(path: Path) -> tuple[int, str]:
    stem = path.stem
    digits = "".join(ch for ch in stem if ch.isdigit())
    return (int(digits) if digits else 10**9, path.name)


def list_level_paths(framework_dir: Path, level_pack: str) -> list[str]:
    level_dir = framework_dir / "levels" / level_pack
    if not level_dir.exists():
        raise FileNotFoundError(f"Level pack not found: {level_dir}")

    level_paths = sorted(level_dir.glob("lvl-*.txt"), key=natural_level_key)
    if not level_paths:
        raise FileNotFoundError(f"No level files found in {level_dir}")

    return [f"./levels/{level_pack}/{path.name}" for path in level_paths]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Python PPO controller on all levels in a level pack")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5050)
    parser.add_argument("--level-pack", default="original")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--timer", type=int, default=200)
    parser.add_argument("--per-level-timeout-seconds", type=int, default=300)
    parser.add_argument("--mario-state", type=int, default=0, choices=[0, 1, 2])
    parser.add_argument("--visuals", default="false", choices=["true", "false"])
    parser.add_argument(
        "--reuse-java-window",
        default="auto",
        choices=["auto", "true", "false"],
        help="Run all sessions in one Java process (reuses the same window). Default: auto (enabled when --visuals true).",
    )
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--stats-path", default=None)
    parser.add_argument("--tensorboard-dir", default=None)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--no-compile", action="store_true")
    return parser.parse_args()


def run_checked(command: list[str], cwd: Path, timeout_s: float | None = None) -> None:
    try:
        completed = subprocess.run(command, cwd=str(cwd), check=False, timeout=timeout_s)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"Command timed out after {timeout_s:.0f}s: {' '.join(command)}") from exc
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {completed.returncode}: {' '.join(command)}")


def start_controller(controller_cmd: list[str], repo_root: Path, host: str, port: int) -> subprocess.Popen:
    print("Starting Python controller:", " ".join(controller_cmd))
    controller_proc = subprocess.Popen(controller_cmd, cwd=str(repo_root))
    if not wait_for_port(host, port, timeout_s=10.0, proc=controller_proc):
        exited_code = controller_proc.poll()
        if exited_code is not None:
            raise RuntimeError(
                f"Controller exited early with code {exited_code}. See controller output above for details."
            )
        if controller_proc.poll() is None:
            controller_proc.terminate()
            try:
                controller_proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                controller_proc.kill()
        raise RuntimeError(f"Controller did not start on {host}:{port} in time.")
    return controller_proc


def stop_controller(controller_proc: subprocess.Popen | None) -> None:
    if controller_proc is not None and controller_proc.poll() is None:
        controller_proc.terminate()
        try:
            controller_proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            controller_proc.kill()


def ensure_controller(
    controller_proc: subprocess.Popen | None,
    controller_cmd: list[str],
    repo_root: Path,
    host: str,
    port: int,
) -> subprocess.Popen:
    if controller_proc is not None and controller_proc.poll() is None and wait_for_port(host, port, timeout_s=1.0):
        return controller_proc

    if controller_proc is not None:
        print("Controller is not healthy. Restarting it before the next level.")
        stop_controller(controller_proc)

    return start_controller(controller_cmd, repo_root, host, port)


def is_controller_failure(message: str) -> bool:
    lowered = message.lower()
    return (
        "cannot connect to python controller" in lowered
        or "connection refused" in lowered
        or "connection reset" in lowered
        or "controller did not start" in lowered
    )


def is_controller_healthy(controller_proc: subprocess.Popen | None, host: str, port: int) -> bool:
    if controller_proc is None or controller_proc.poll() is not None:
        return False
    return wait_for_port(host, port, timeout_s=0.5, proc=controller_proc)


def main() -> int:
    args = parse_args()

    reuse_java_window = args.reuse_java_window
    if reuse_java_window == "auto":
        reuse_java_window = "true" if args.visuals == "true" else "false"
    reuse_java_window_enabled = reuse_java_window == "true"

    effective_port = int(args.port)
    if reuse_java_window_enabled:
        chosen_port = choose_port(args.host, effective_port)
        if chosen_port != effective_port:
            print(f"Port {effective_port} is busy; using free port {chosen_port} for this run.")
        effective_port = chosen_port

    repo_root = Path(__file__).resolve().parent.parent
    framework_dir = repo_root / "Mario-AI-Framework"
    controller_path = repo_root / "PythonController" / "controller.py"
    launcher_path = repo_root / "PythonController" / "start_java_game.py"
    default_artifact_dir = repo_root / "PythonController" / "checkpoints"

    model_path = Path(args.model_path) if args.model_path else default_artifact_dir / f"ppo_{args.level_pack}.pt"
    stats_path = Path(args.stats_path) if args.stats_path else default_artifact_dir / f"ppo_{args.level_pack}_stats.jsonl"
    tensorboard_dir = Path(args.tensorboard_dir) if args.tensorboard_dir else default_artifact_dir / "tensorboard" / args.level_pack
    failure_log_path = default_artifact_dir / f"ppo_{args.level_pack}_failures.jsonl"
    level_paths = list_level_paths(framework_dir, args.level_pack)

    controller_cmd = [
        sys.executable,
        str(controller_path),
        "--host",
        args.host,
        "--port",
        str(effective_port),
        "--model-path",
        str(model_path),
        "--stats-path",
        str(stats_path),
        "--tensorboard-dir",
        str(tensorboard_dir),
        "--save-every",
        str(args.save_every),
    ]

    controller_proc = None

    try:
        controller_proc = start_controller(controller_cmd, repo_root, args.host, effective_port)

        if not args.no_compile:
            run_checked(
                ["javac", "-cp", "src", "src/mff/python/PythonControllerMain.java"],
                framework_dir,
                timeout_s=300,
            )

        total_runs = args.epochs * len(level_paths)
        failures: list[dict[str, object]] = []

        if reuse_java_window_enabled:
            # Run all sessions in one Java process so the window persists.
            sessions = total_runs
            level_arg = ";".join(level_paths)
            java_cmd = [
                sys.executable,
                str(launcher_path),
                "--host",
                args.host,
                "--port",
                str(effective_port),
                "--level",
                level_arg,
                "--sessions",
                str(sessions),
                "--session-timeout-seconds",
                str(args.per_level_timeout_seconds),
                "--timer",
                str(args.timer),
                "--mario-state",
                str(args.mario_state),
                "--visuals",
                args.visuals,
                "--timeout-seconds",
                str((max(1, args.per_level_timeout_seconds) + 10) * max(1, sessions)),
                "--no-compile",
            ]
            attempt = 0
            while True:
                attempt += 1
                try:
                    run_checked(java_cmd, repo_root, timeout_s=(args.per_level_timeout_seconds + 30) * max(1, sessions))
                    break
                except RuntimeError as exc:
                    if attempt == 1:
                        unhealthy = not is_controller_healthy(controller_proc, args.host, effective_port)
                        if unhealthy or is_controller_failure(str(exc)):
                            print("Detected controller-related failure. Restarting controller and retrying Java run once.")
                            controller_proc = ensure_controller(None if controller_proc is None else controller_proc, controller_cmd, repo_root, args.host, effective_port)
                            continue

                    failure = {
                        "epoch": None,
                        "run_index": None,
                        "level": "BATCH",
                        "attempt": attempt,
                        "error": str(exc),
                    }
                    failures.append(failure)
                    failure_log_path.parent.mkdir(parents=True, exist_ok=True)
                    with failure_log_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(failure) + "\n")
                    raise
        else:
            run_index = 0
            for epoch in range(1, args.epochs + 1):
                print(f"Epoch {epoch}/{args.epochs}")
                for level_path in level_paths:
                    run_index += 1
                    print(f"[{run_index}/{total_runs}] Training on {level_path}")
                    controller_proc = ensure_controller(controller_proc, controller_cmd, repo_root, args.host, args.port)
                    java_cmd = [
                        sys.executable,
                        str(launcher_path),
                        "--host",
                        args.host,
                        "--port",
                        str(args.port),
                        "--level",
                        level_path,
                        "--timer",
                        str(args.timer),
                        "--mario-state",
                        str(args.mario_state),
                        "--visuals",
                        args.visuals,
                        "--timeout-seconds",
                        str(args.per_level_timeout_seconds),
                        "--no-compile",
                    ]
                    attempt = 0
                    while True:
                        attempt += 1
                        try:
                            run_checked(java_cmd, repo_root, timeout_s=args.per_level_timeout_seconds + 30)
                            break
                        except RuntimeError as exc:
                            if attempt == 1:
                                unhealthy = not is_controller_healthy(controller_proc, args.host, args.port)
                                if unhealthy or is_controller_failure(str(exc)):
                                    print("Detected controller-related failure. Restarting controller and retrying level once.")
                                    controller_proc = ensure_controller(None if controller_proc is None else controller_proc, controller_cmd, repo_root, args.host, args.port)
                                    continue

                            failure = {
                                "epoch": epoch,
                                "run_index": run_index,
                                "level": level_path,
                                "attempt": attempt,
                                "error": str(exc),
                            }
                            failures.append(failure)
                            failure_log_path.parent.mkdir(parents=True, exist_ok=True)
                            with failure_log_path.open("a", encoding="utf-8") as handle:
                                handle.write(json.dumps(failure) + "\n")
                            print(f"Skipping level after failure: {level_path}")
                            print(str(exc))
                            break

        print(f"Training complete. Model: {model_path}")
        print(f"Episode stats: {stats_path}")
        print(f"TensorBoard logs: {tensorboard_dir}")
        if failures:
            print(f"Skipped levels logged to: {failure_log_path}")
            print(f"Skipped level count: {len(failures)}")
        return 0
    finally:
        stop_controller(controller_proc)


if __name__ == "__main__":
    sys.exit(main())
