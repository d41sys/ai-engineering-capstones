from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from .models import Model, get_model
from .runner import EvalRecord, generate_leaderboard, run
from .spec import SpecError, TaskSpec


def find_task_specs(tasks_dir: Path) -> list[TaskSpec]:
    """Find all TaskSpec yaml files in a directory or its subdirectories."""
    if not tasks_dir.exists():
        raise SpecError(f"Tasks directory not found: {tasks_dir}")

    specs: list[TaskSpec] = []
    if tasks_dir.is_file() and tasks_dir.suffix.lower() in (".yaml", ".yml"):
        specs.append(TaskSpec.from_yaml(tasks_dir))
        return specs

    # Look for spec.yaml in immediate child directories or any .yaml files
    yaml_files = sorted(list(tasks_dir.glob("*/spec.yaml")) + list(tasks_dir.glob("*.yaml")))
    if not yaml_files:
        yaml_files = sorted(tasks_dir.glob("**/*.yaml"))

    seen_paths: set[Path] = set()
    for yf in yaml_files:
        resolved = yf.resolve()
        if resolved in seen_paths:
            continue
        seen_paths.add(resolved)
        try:
            spec = TaskSpec.from_yaml(resolved)
            specs.append(spec)
        except Exception as exc:
            sys.stderr.write(f"Warning: Failed to load spec from {yf}: {exc}\n")

    if not specs:
        raise SpecError(f"No valid task specs found in {tasks_dir}")

    return specs


def parse_models(models_str: str) -> list[Model]:
    model_names = [m.strip() for m in models_str.split(",") if m.strip()]
    return [get_model(name) for name in model_names]


def run_command(args: argparse.Namespace) -> int:
    tasks_path = Path(args.tasks)
    specs = find_task_specs(tasks_path)
    models = parse_models(args.models)
    out_dir = Path(args.out)

    print(f"Loaded {len(specs)} task specs from {tasks_path}")
    print(f"Loaded {len(models)} models: {[m.name for m in models]}")
    print(f"Executing evaluations -> {out_dir}")

    summary = run(specs, models, out_dir, max_workers=args.workers)

    print(f"\nDone! Evaluated: {summary.total_evaluated}, Skipped: {summary.total_skipped}")
    print(f"Results written to: {summary.results_path}")
    print(f"Leaderboard written to: {summary.leaderboard_path}\n")
    print(summary.leaderboard.to_markdown())
    return 0


def leaderboard_command(args: argparse.Namespace) -> int:
    target_dir = Path(args.dir)
    board_file = target_dir / "leaderboard.md"
    results_file = target_dir / "results.jsonl"

    if board_file.is_file() and not args.recompute:
        print(board_file.read_text(encoding="utf-8"))
        return 0

    if not results_file.is_file():
        sys.stderr.write(f"Error: Neither leaderboard.md nor results.jsonl found in {target_dir}\n")
        return 1

    records: list[EvalRecord] = []
    with open(results_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(EvalRecord.from_dict(json.loads(line)))
            except Exception:
                continue

    leaderboard = generate_leaderboard(records)
    md = leaderboard.to_markdown()
    with open(board_file, "w", encoding="utf-8") as f:
        f.write(md + "\n")

    print(md)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="evalkit",
        description="Evaluation runner and benchmark harness from first principles",
    )
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    # run command
    run_parser = subparsers.add_parser("run", help="Run evaluation benchmark")
    run_parser.add_argument(
        "--tasks",
        type=str,
        required=True,
        help="Path to tasks directory containing spec.yaml files",
    )
    run_parser.add_argument(
        "--models",
        type=str,
        default="oracle,noisy:0.3,random",
        help="Comma-separated list of models (e.g. oracle,noisy:0.3,random)",
    )
    run_parser.add_argument(
        "--out",
        type=str,
        default="runs/demo",
        help="Output directory for results and leaderboard",
    )
    run_parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of worker threads (default 4)",
    )

    # leaderboard command
    board_parser = subparsers.add_parser("leaderboard", help="View or recompute leaderboard")
    board_parser.add_argument(
        "dir",
        type=str,
        help="Directory containing results.jsonl / leaderboard.md",
    )
    board_parser.add_argument(
        "--recompute",
        action="store_true",
        help="Recompute leaderboard from results.jsonl",
    )

    args = parser.parse_args(argv)

    if args.subcommand == "run":
        return run_command(args)
    elif args.subcommand == "leaderboard":
        return leaderboard_command(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
