from __future__ import annotations

import math

from evalkit.metrics.calibration import (
    brier_score,
    expected_calibration_error,
    perplexity,
    reliability_diagram,
)


def test_ece_zero_perfectly_calibrated() -> None:
    # Acceptance criteria: ECE = 0 for perfectly calibrated bins.
    # Case 1: All high-confidence correct predictions (in top bin)
    confidences = [1.0] * 20 + [0.0] * 20
    correct = [1] * 20 + [0] * 20
    ece = expected_calibration_error(confidences, correct, n_bins=15)
    assert math.isclose(ece, 0.0, abs_tol=1e-9)

    # Case 2: Multi-bin calibrated predictions
    # Bin with conf ~ 0.5: 10 items with conf=0.5, 5 correct (acc=0.5)
    # Bin with conf ~ 0.2: 10 items with conf=0.2, 2 correct (acc=0.2)
    # Bin with conf ~ 0.8: 10 items with conf=0.8, 8 correct (acc=0.8)
    conf_multi = [0.2] * 10 + [0.5] * 10 + [0.8] * 10
    corr_multi = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0] + [1] * 5 + [0] * 5 + [1] * 8 + [0] * 2
    ece_multi = expected_calibration_error(conf_multi, corr_multi, n_bins=10)
    assert math.isclose(ece_multi, 0.0, abs_tol=1e-9)


def test_ece_non_zero_miscalibrated() -> None:
    # Overconfident: conf=1.0, but all wrong (correct=0)
    confidences = [1.0] * 10
    correct = [0] * 10
    ece = expected_calibration_error(confidences, correct, n_bins=15)
    # Average conf = 1.0, acc = 0.0 -> gap = 1.0
    assert math.isclose(ece, 1.0, rel_tol=1e-5)


def test_perplexity() -> None:
    # log(0.5) ≈ -0.69314718 -> avg NLL = 0.69314718 -> PPL = exp(0.69314718) = 2.0
    half_logp = -math.log(2.0)
    ppl = perplexity([half_logp, half_logp, half_logp])
    assert math.isclose(ppl, 2.0, rel_tol=1e-5)

    # Perfect logprobs = 0.0 -> PPL = 1.0
    assert math.isclose(perplexity([0.0, 0.0]), 1.0, rel_tol=1e-5)


def test_brier_score() -> None:
    # Perfect predictions: MSE = 0.0
    assert brier_score([1.0, 0.0], [1, 0]) == 0.0
    # Completely wrong: MSE = 1.0
    assert brier_score([1.0, 0.0], [0, 1]) == 1.0
    # 50-50 guess: (0.5 - 1)^2 = 0.25, (0.5 - 0)^2 = 0.25 -> avg = 0.25
    assert brier_score([0.5, 0.5], [1, 0]) == 0.25


def test_reliability_diagram_structure() -> None:
    confs = [0.1, 0.5, 0.9]
    corr = [0, 1, 1]
    res = reliability_diagram(confs, corr, n_bins=15)
    assert len(res["bin_edges"]) == 16
    assert len(res["bin_confidences"]) == 15
    assert len(res["bin_accuracies"]) == 15
    assert len(res["bin_counts"]) == 15
    assert sum(res["bin_counts"]) == 3
