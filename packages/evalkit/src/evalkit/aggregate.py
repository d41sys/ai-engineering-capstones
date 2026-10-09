from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np


def bootstrap_mean_ci(
    scores: Sequence[float],
    num_resamples: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Compute mean and 95% bootstrap confidence interval (mean, ci_lo, ci_hi).

    Default: 1000 resamples, seed 0.
    """
    if len(scores) == 0:
        return (0.0, 0.0, 0.0)

    arr = np.asarray(scores, dtype=np.float64)
    n = arr.size
    mean_val = float(arr.mean())
    if n == 1 or num_resamples <= 0:
        return (mean_val, mean_val, mean_val)

    rng = np.random.default_rng(seed)
    resamples = rng.choice(arr, size=(num_resamples, n), replace=True)
    resample_means = resamples.mean(axis=1)

    alpha = 1.0 - confidence_level
    lo = float(np.percentile(resample_means, 100.0 * (alpha / 2.0)))
    hi = float(np.percentile(resample_means, 100.0 * (1.0 - alpha / 2.0)))
    return (mean_val, lo, hi)


def macro_average(task_scores: dict[str, float] | Sequence[float]) -> float:
    """Compute unweighted macro average over task scores."""
    scores = list(task_scores.values()) if isinstance(task_scores, dict) else list(task_scores)
    if not scores:
        return 0.0
    return float(np.mean(scores))


def compute_elo_ratings(
    matches: Sequence[tuple[str, str, float]],
    base_rating: float = 1000.0,
    k_factor: float = 32.0,
) -> dict[str, float]:
    """Compute Elo ratings from pairwise matches.

    matches: list of (model_a, model_b, score_a) where score_a is:
        1.0: A won
        0.5: tie
        0.0: B won
    Matches are sorted deterministically before processing.
    """
    if not matches:
        return {}

    # Deterministic ordering
    sorted_matches = sorted(matches, key=lambda m: (m[0], m[1], m[2]))

    # Initialize ratings
    all_models: set[str] = set()
    for m in sorted_matches:
        all_models.add(m[0])
        all_models.add(m[1])

    ratings: dict[str, float] = {model: base_rating for model in sorted(all_models)}

    for model_a, model_b, score_a in sorted_matches:
        ra = ratings[model_a]
        rb = ratings[model_b]

        ea = 1.0 / (1.0 + math.pow(10.0, (rb - ra) / 400.0))
        eb = 1.0 - ea

        score_b = 1.0 - score_a

        ratings[model_a] = ra + k_factor * (score_a - ea)
        ratings[model_b] = rb + k_factor * (score_b - eb)

    return ratings


def build_pairwise_matches(
    example_scores: dict[str, dict[str, float]],
) -> list[tuple[str, str, float]]:
    """Build pairwise matches from per-example model scores.

    example_scores: dict mapping example_id -> {model_id: score}
    Returns deterministically sorted matches.
    """
    matches: list[tuple[str, str, float]] = []

    for ex_id in sorted(example_scores.keys()):
        model_dict = example_scores[ex_id]
        models = sorted(model_dict.keys())
        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                ma = models[i]
                mb = models[j]
                sa = model_dict[ma]
                sb = model_dict[mb]
                if math.isclose(sa, sb, abs_tol=1e-9):
                    score_a = 0.5
                elif sa > sb:
                    score_a = 1.0
                else:
                    score_a = 0.0
                matches.append((ma, mb, score_a))

    return matches


@dataclass
class ModelSummary:
    model_name: str
    macro_mean: float
    ci_lower: float
    ci_upper: float
    win_rate: float
    elo_rating: float
    ece: float = 0.0
    brier: float = 0.0
    total_examples: int = 0
    per_task_means: dict[str, float] = field(default_factory=dict)


@dataclass
class LeaderboardReport:
    models: list[ModelSummary]

    def to_markdown(self) -> str:
        lines: list[str] = [
            "# Evaluation Leaderboard",
            "",
            (
                "| Rank | Model | Macro Mean | 95% CI | Win Rate | Elo Rating | "
                "ECE (15 bins) | Brier | Tasks Evaluated |"
            ),
            (
                "|------|-------|------------|--------|----------|------------|"
                "---------------|-------|-----------------|"
            ),
        ]
        # Sort models descending by macro_mean then elo_rating
        ranked = sorted(self.models, key=lambda m: (-m.macro_mean, -m.elo_rating, m.model_name))
        for rank, m in enumerate(ranked, start=1):
            ci_str = f"[{m.ci_lower:.3f}, {m.ci_upper:.3f}]"
            tasks_count = len(m.per_task_means)
            lines.append(
                f"| {rank} | {m.model_name} | {m.macro_mean:.3f} | {ci_str} | {m.win_rate:.3f} | "
                f"{m.elo_rating:.1f} | {m.ece:.3f} | {m.brier:.3f} | {tasks_count} |"
            )

        if ranked and ranked[0].per_task_means:
            lines.extend(["", "## Per-Task Breakdown", ""])
            all_tasks = sorted({t for m in ranked for t in m.per_task_means})
            header = "| Model | " + " | ".join(all_tasks) + " |"
            sep = "|-------|" + "|".join(["-------" for _ in all_tasks]) + "|"
            lines.append(header)
            lines.append(sep)
            for m in ranked:
                task_cells = [f"{m.per_task_means.get(t, 0.0):.3f}" for t in all_tasks]
                lines.append(f"| {m.model_name} | " + " | ".join(task_cells) + " |")

        return "\n".join(lines)
