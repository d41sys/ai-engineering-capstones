from __future__ import annotations

import math

import numpy as np
from evalkit.aggregate import (
    bootstrap_macro_ci,
    bootstrap_mean_ci,
    build_pairwise_matches,
    compute_elo_ratings,
    macro_average,
)


def test_bootstrap_mean_ci_deterministic() -> None:
    scores = [1.0, 1.0, 0.5, 0.0, 1.0, 0.8, 0.2, 0.9]
    mean1, lo1, hi1 = bootstrap_mean_ci(scores, num_resamples=1000, seed=0)
    mean2, lo2, hi2 = bootstrap_mean_ci(scores, num_resamples=1000, seed=0)

    # Deterministic output with seed 0
    assert math.isclose(mean1, float(np.mean(scores)), rel_tol=1e-9)
    assert mean1 == mean2
    assert lo1 == lo2
    assert hi1 == hi2
    assert lo1 <= mean1 <= hi1


def test_macro_average() -> None:
    task_scores = {"arithmetic": 0.9, "mcq": 0.8, "code": 0.7}
    avg = macro_average(task_scores)
    assert math.isclose(avg, 0.8, rel_tol=1e-9)
    assert macro_average([]) == 0.0


def test_compute_elo_ratings() -> None:
    # A beats B, B beats C
    matches = [
        ("model_a", "model_b", 1.0),
        ("model_a", "model_b", 1.0),
        ("model_b", "model_c", 1.0),
        ("model_b", "model_c", 1.0),
    ]
    ratings = compute_elo_ratings(matches, base_rating=1000.0, k_factor=32.0)
    assert ratings["model_a"] > ratings["model_b"]
    assert ratings["model_b"] > ratings["model_c"]
    # Total points conserved
    total = sum(ratings.values())
    assert math.isclose(total, 3000.0, rel_tol=1e-5)


def test_elo_dominant_model_ranks_first_regardless_of_pair_grouping() -> None:
    # oracle >= noisy > random on every example; noisy beats random far more often
    # than it ties oracle. Grouping matches by pair used to rank noisy above oracle.
    example_scores = {
        f"ex{i:02d}": {"oracle": 1.0, "noisy": 1.0 if i % 3 else 0.0, "random": 0.0}
        for i in range(30)
    }
    ratings = compute_elo_ratings(build_pairwise_matches(example_scores))
    assert ratings["oracle"] > ratings["noisy"] > ratings["random"]


def test_bootstrap_macro_ci_brackets_macro_mean() -> None:
    # Unequal task sizes: macro mean (0.6) differs from the pooled micro mean (0.8).
    task_scores = {"big": [1.0] * 30, "small": [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]}
    point, lo, hi = bootstrap_macro_ci(task_scores, num_resamples=1000, seed=0)
    assert math.isclose(point, (1.0 + 0.2) / 2)
    assert lo <= point <= hi
    assert bootstrap_macro_ci(task_scores, seed=0) == (point, lo, hi)


def test_build_pairwise_matches() -> None:
    example_scores = {
        "ex1": {"oracle": 1.0, "noisy": 0.5, "random": 0.0},
        "ex2": {"oracle": 1.0, "noisy": 1.0, "random": 0.0},
    }
    matches = build_pairwise_matches(example_scores)
    # ex1: (noisy, oracle, 0.0), (noisy, random, 1.0), (oracle, random, 1.0)
    # ex2: (noisy, oracle, 0.5), (noisy, random, 1.0), (oracle, random, 1.0)
    assert len(matches) == 6
