import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compile and launch the Java Mario framework with the Python socket agent"
    )
    parser.add_argument("--host", default="127.0.0.1", help="Python controller host")
    parser.add_argument("--port", type=int, default=5050, help="Python controller port")
    parser.add_argument("--level", default="./levels/original/lvl-1.txt", help="Level path from Mario-AI-Framework")
    parser.add_argument(
        "--sessions",
        type=int,
        default=1,
        help="How many games to run in one Java process (reuses the same window when visuals=true).",
    )
    parser.add_argument(
        "--session-timeout-seconds",
        type=int,
        default=None,
        help="Optional wall-clock timeout per game session (seconds).",
    )
    parser.add_argument("--timer", type=int, default=200, help="Timer value in framework ticks")
    parser.add_argument("--mario-state", type=int, default=0, choices=[0, 1, 2], help="0=small, 1=large, 2=fire")
    parser.add_argument("--visuals", default="true", choices=["true", "false"], help="Show game window")
    parser.add_argument("--java", default="java", help="Java executable")
    parser.add_argument("--javac", default="javac", help="Javac executable")
    parser.add_argument("--timeout-seconds", type=int, default=None, help="Optional wall-clock timeout for the Java run")
    parser.add_argument("--no-compile", action="store_true", help="Skip javac step")
    return parser.parse_args()


def run_checked(command: list[str], cwd: Path, timeout_s: float | None = None) -> None:
    print("Running:", " ".join(command))
    try:
        completed = subprocess.run(command, cwd=str(cwd), check=False, timeout=timeout_s)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"Command timed out after {timeout_s:.0f}s: {' '.join(command)}") from exc
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {completed.returncode}: {' '.join(command)}")


def main() -> int:
    args = parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    framework_dir = repo_root / "Mario-AI-Framework"
    src_dir = framework_dir / "src"
    entry_java = src_dir / "mff" / "python" / "PythonControllerMain.java"

    if not entry_java.exists():
        print(f"Entry point not found: {entry_java}")
        return 1

    try:
        if not args.no_compile:
            run_checked(
                [args.javac, "-cp", "src", str(entry_java.relative_to(framework_dir))],
                framework_dir,
            )

        java_cmd = [
                args.java,
                "-cp",
                "src",
                "mff.python.PythonControllerMain",
                args.host,
                str(args.port),
                args.level,
                str(args.timer),
                str(args.mario_state),
                args.visuals,
                str(args.sessions),
        ]
        if args.session_timeout_seconds is not None:
            java_cmd.append(str(args.session_timeout_seconds))

        run_checked(java_cmd, framework_dir, timeout_s=args.timeout_seconds)
        return 0
    except RuntimeError as exc:
        print(exc)
        return 1
    except FileNotFoundError as exc:
        print(f"Executable not found: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
