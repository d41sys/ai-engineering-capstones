from __future__ import annotations

import json
import threading
from collections import defaultdict
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .aggregate import (
    LeaderboardReport,
    ModelSummary,
    bootstrap_mean_ci,
    build_pairwise_matches,
    compute_elo_ratings,
    macro_average,
)
from .metrics.calibration import brier_score, expected_calibration_error
from .metrics.classical import score_classical
from .metrics.code_exec import run_code_exec
from .models import Model
from .spec import TaskSpec


@dataclass
class EvalRecord:
    task: str
    model: str
    example_id: str
    score: float
    metrics: dict[str, float]
    prediction: str
    targets: list[str]
    confidence: float
    correct: int
    token_nll: float = 0.0
    token_count: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "model": self.model,
            "example_id": self.example_id,
            "score": self.score,
            "metrics": self.metrics,
            "prediction": self.prediction,
            "targets": self.targets,
            "confidence": self.confidence,
            "correct": self.correct,
            "token_nll": self.token_nll,
            "token_count": self.token_count,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> EvalRecord:
        return cls(
            task=str(d["task"]),
            model=str(d["model"]),
            example_id=str(d["example_id"]),
            score=float(d["score"]),
            metrics={k: float(v) for k, v in d.get("metrics", {}).items()},
            prediction=str(d.get("prediction", "")),
            targets=[str(t) for t in d.get("targets", [])],
            confidence=float(d.get("confidence", 0.5)),
            correct=int(d.get("correct", 0)),
            token_nll=float(d.get("token_nll", 0.0)),
            token_count=int(d.get("token_count", 1)),
        )


@dataclass
class EvalSummary:
    records: list[EvalRecord]
    leaderboard: LeaderboardReport
    out_dir: Path
    total_evaluated: int
    total_skipped: int
    results_path: Path
    leaderboard_path: Path


def _extract_targets(example: dict[str, Any]) -> list[str]:
    if "targets" in example and isinstance(example["targets"], list):
        return [str(t) for t in example["targets"]]
    if "target" in example and example["target"] is not None:
        return [str(example["target"])]
    if "answer" in example and example["answer"] is not None:
        return [str(example["answer"])]
    if "canonical_solution" in example and example["canonical_solution"] is not None:
        return [str(example["canonical_solution"])]
    return []


def _evaluate_single(
    spec: TaskSpec,
    model: Model,
    example: dict[str, Any],
    example_id: str,
) -> EvalRecord:
    prompt = spec.render_prompt(example)
    choices = example.get("choices")
    targets = _extract_targets(example)

    # Use model prediction
    pred = model.predict(
        prompt,
        task_type=spec.type,
        choices=choices,
        example=example,
    )

    metrics_dict: dict[str, float] = {}

    if spec.type == "code":
        tests = example.get("tests", [])
        code_result = run_code_exec(pred.text, tests, timeout_s=10.0)
        primary_score = 1.0 if code_result.status == "pass" else 0.0
        correct_flag = 1 if code_result.status == "pass" else 0
        metrics_dict["code_exec"] = primary_score
        for m in spec.metrics:
            if m != "code_exec":
                try:
                    metrics_dict[m] = score_classical(m, pred.text, targets)
                except ValueError:
                    metrics_dict[m] = primary_score
    else:
        for m in spec.metrics:
            metrics_dict[m] = score_classical(m, pred.text, targets)
        primary_metric = spec.metrics[0] if spec.metrics else "exact_match"
        primary_score = metrics_dict.get(primary_metric, 0.0)
        correct_flag = 1 if primary_score >= 0.99 else 0

    return EvalRecord(
        task=spec.name,
        model=model.name,
        example_id=example_id,
        score=primary_score,
        metrics=metrics_dict,
        prediction=pred.text,
        targets=targets,
        confidence=pred.confidence,
        correct=correct_flag,
        token_nll=pred.token_nll,
        token_count=pred.token_count,
    )


def generate_leaderboard(records: Sequence[EvalRecord]) -> LeaderboardReport:
    """Build LeaderboardReport from all evaluation records."""
    if not records:
        return LeaderboardReport(models=[])

    # Group records by model and by (example_id, task)
    by_model: dict[str, list[EvalRecord]] = defaultdict(list)
    by_example: dict[str, dict[str, float]] = defaultdict(dict)
    all_tasks: set[str] = set()

    for r in records:
        by_model[r.model].append(r)
        key = f"{r.task}::{r.example_id}"
        by_example[key][r.model] = r.score
        all_tasks.add(r.task)

    # Compute pairwise Elo ratings
    matches = build_pairwise_matches(by_example)
    elo_ratings = compute_elo_ratings(matches, base_rating=1000.0, k_factor=32.0)

    # Compute win rates
    win_counts: dict[str, float] = defaultdict(float)
    total_comparisons: dict[str, int] = defaultdict(int)
    for ma, mb, sa in matches:
        total_comparisons[ma] += 1
        total_comparisons[mb] += 1
        win_counts[ma] += sa
        win_counts[mb] += 1.0 - sa

    model_summaries: list[ModelSummary] = []

    for model_name, recs in by_model.items():
        # Per-task means
        task_scores: dict[str, list[float]] = defaultdict(list)
        for r in recs:
            task_scores[r.task].append(r.score)

        per_task_means: dict[str, float] = {
            t: float(sum(scores) / len(scores)) for t, scores in task_scores.items()
        }

        # Macro average
        macro_mean = macro_average(per_task_means)

        # Bootstrap CI on all example scores
        all_scores = [r.score for r in recs]
        _, ci_lo, ci_hi = bootstrap_mean_ci(all_scores, num_resamples=1000, seed=0)

        # Calibration
        confs = [r.confidence for r in recs]
        corrs = [r.correct for r in recs]
        ece = expected_calibration_error(confs, corrs, n_bins=15)
        brier = brier_score(confs, corrs)

        # Win rate
        comps = total_comparisons.get(model_name, 0)
        win_rate = (win_counts.get(model_name, 0.0) / comps) if comps > 0 else 0.0

        elo = elo_ratings.get(model_name, 1000.0)

        model_summaries.append(
            ModelSummary(
                model_name=model_name,
                macro_mean=macro_mean,
                ci_lower=ci_lo,
                ci_upper=ci_hi,
                win_rate=win_rate,
                elo_rating=elo,
                ece=ece,
                brier=brier,
                total_examples=len(recs),
                per_task_means=per_task_means,
            )
        )

    # Sort descending by macro_mean then elo_rating
    model_summaries.sort(key=lambda m: (-m.macro_mean, -m.elo_rating, m.model_name))
    return LeaderboardReport(models=model_summaries)


def run(
    specs: Sequence[TaskSpec],
    models: Sequence[Model],
    out_dir: str | Path,
    max_workers: int = 4,
) -> EvalSummary:
    """Run evaluation benchmark across specs and models.

    Resumable: skips (task, model, example_id) already stored in out_dir/results.jsonl.
    Multi-threaded: uses ThreadPoolExecutor(max_workers).
    Outputs results.jsonl and leaderboard.md.
    """
    output_path = Path(out_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    results_file = output_path / "results.jsonl"
    leaderboard_file = output_path / "leaderboard.md"

    existing_records: list[EvalRecord] = []
    completed_keys: set[tuple[str, str, str]] = set()

    if results_file.is_file():
        with open(results_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    rec = EvalRecord.from_dict(data)
                    existing_records.append(rec)
                    completed_keys.add((rec.task, rec.model, rec.example_id))
                except Exception:
                    continue

    work_items: list[tuple[TaskSpec, Model, dict[str, Any], str]] = []
    total_skipped = 0

    for spec in specs:
        rows = spec.load_dataset()
        for idx, row in enumerate(rows):
            ex_id = str(row.get("id", row.get("task_id", f"ex_{idx:04d}")))
            for model in models:
                key = (spec.name, model.name, ex_id)
                if key in completed_keys:
                    total_skipped += 1
                else:
                    work_items.append((spec, model, row, ex_id))

    newly_evaluated = 0
    new_records: list[EvalRecord] = []
    file_lock = threading.Lock()

    if work_items:
        workers = max(1, min(max_workers, len(work_items)))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_item = {
                executor.submit(_evaluate_single, spec, model, row, ex_id): (
                    spec.name,
                    model.name,
                    ex_id,
                )
                for spec, model, row, ex_id in work_items
            }

            with open(results_file, "a", encoding="utf-8") as out_f:
                for future in as_completed(future_to_item):
                    rec = future.result()
                    new_records.append(rec)
                    newly_evaluated += 1
                    with file_lock:
                        out_f.write(json.dumps(rec.to_dict()) + "\n")
                        out_f.flush()

    all_records = existing_records + new_records
    leaderboard = generate_leaderboard(all_records)

    with open(leaderboard_file, "w", encoding="utf-8") as f:
        f.write(leaderboard.to_markdown() + "\n")

    return EvalSummary(
        records=all_records,
        leaderboard=leaderboard,
        out_dir=output_path,
        total_evaluated=newly_evaluated,
        total_skipped=total_skipped,
        results_path=results_file,
        leaderboard_path=leaderboard_file,
    )
