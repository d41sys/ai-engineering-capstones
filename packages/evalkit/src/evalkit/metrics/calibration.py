from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np


def perplexity(
    token_logprobs: Sequence[float],
    token_counts: Sequence[int] | None = None,
) -> float:
    """Calculate perplexity from token log probabilities or token negative log likelihoods.

    If values are <= 0, they are treated as log probabilities: PPL = exp(- mean(logprobs)).
    If values are > 0, they are treated as negative log likelihoods: PPL = exp(mean(nll)).
    If token_counts is provided, weighted average is computed.
    """
    if len(token_logprobs) == 0:
        return float("nan")

    arr = np.asarray(token_logprobs, dtype=np.float64)
    if token_counts is not None:
        counts = np.asarray(token_counts, dtype=np.float64)
        total_tokens = float(np.sum(counts))
        if total_tokens <= 0:
            return float("nan")
        # Check if values are logprobs (<= 0) or NLL (> 0)
        if np.all(arr <= 0.0):
            avg_nll = -float(np.sum(arr * counts) / total_tokens)
        else:
            avg_nll = float(np.sum(arr * counts) / total_tokens)
    else:
        avg_nll = -float(np.mean(arr)) if np.all(arr <= 0.0) else float(np.mean(arr))

    return float(math.exp(avg_nll))


def _validate_conf_corr(
    confidences: Sequence[float], correct: Sequence[int | bool]
) -> tuple[np.ndarray, np.ndarray]:
    conf = np.asarray(confidences, dtype=np.float64)
    corr = np.asarray(correct, dtype=np.float64)
    if conf.shape != corr.shape:
        raise ValueError(
            f"confidences and correct must have the same length ({len(conf)} != {len(corr)})"
        )
    if conf.size > 0 and (float(conf.min()) < 0.0 or float(conf.max()) > 1.0):
        raise ValueError("confidences must be in the range [0.0, 1.0]")
    return conf, corr


def expected_calibration_error(
    confidences: Sequence[float],
    correct: Sequence[int | bool],
    n_bins: int = 15,
) -> float:
    """Expected Calibration Error (ECE) with equal-width bins (default 15 bins)."""
    if n_bins <= 0:
        raise ValueError("n_bins must be positive")
    conf, corr = _validate_conf_corr(confidences, correct)
    n = conf.size
    if n == 0:
        return 0.0

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total_ece = 0.0

    for b in range(n_bins):
        low = edges[b]
        high = edges[b + 1]
        mask = (conf >= low) & (conf <= high) if b == n_bins - 1 else (conf >= low) & (conf < high)

        bin_size = int(np.sum(mask))
        if bin_size == 0:
            continue

        bin_conf = float(np.mean(conf[mask]))
        bin_acc = float(np.mean(corr[mask]))
        total_ece += (bin_size / n) * abs(bin_conf - bin_acc)

    return float(total_ece)


def brier_score(
    confidences: Sequence[float],
    correct: Sequence[int | bool],
) -> float:
    """Brier score: mean squared error between predicted confidence and binary correctness."""
    conf, corr = _validate_conf_corr(confidences, correct)
    if conf.size == 0:
        return 0.0
    return float(np.mean((conf - corr) ** 2))


def reliability_diagram(
    confidences: Sequence[float],
    correct: Sequence[int | bool],
    n_bins: int = 15,
) -> dict[str, list[float] | list[int]]:
    """Compute bin-level statistics for reliability diagrams."""
    if n_bins <= 0:
        raise ValueError("n_bins must be positive")
    conf, corr = _validate_conf_corr(confidences, correct)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_confs: list[float] = []
    bin_accs: list[float] = []
    bin_counts: list[int] = []

    for b in range(n_bins):
        low = edges[b]
        high = edges[b + 1]
        mask = (conf >= low) & (conf <= high) if b == n_bins - 1 else (conf >= low) & (conf < high)

        count = int(np.sum(mask))
        bin_counts.append(count)
        if count == 0:
            bin_confs.append(float((low + high) / 2.0))
            bin_accs.append(0.0)
        else:
            bin_confs.append(float(np.mean(conf[mask])))
            bin_accs.append(float(np.mean(corr[mask])))

    return {
        "bin_edges": [float(x) for x in edges],
        "bin_confidences": bin_confs,
        "bin_accuracies": bin_accs,
        "bin_counts": bin_counts,
    }


def plot_reliability_diagram(
    confidences: Sequence[float],
    correct: Sequence[int | bool],
    out_path: str | Path,
    n_bins: int = 15,
    title: str = "Reliability Diagram",
) -> None:
    """Plot reliability diagram and save to file. Requires matplotlib (evalkit[plot])."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for plot_reliability_diagram. Install with `evalkit[plot]`"
        ) from exc

    data = reliability_diagram(confidences, correct, n_bins=n_bins)
    bin_edges = data["bin_edges"]
    bin_accs = data["bin_accuracies"]
    _ = data["bin_confidences"]
    bin_counts = data["bin_counts"]

    ece = expected_calibration_error(confidences, correct, n_bins=n_bins)
    brier = brier_score(confidences, correct)

    fig, ax = plt.subplots(figsize=(6, 6))
    widths = [bin_edges[i + 1] - bin_edges[i] for i in range(n_bins)]
    centers = [(bin_edges[i + 1] + bin_edges[i]) / 2.0 for i in range(n_bins)]

    # Filter populated bins
    pop_centers = [centers[i] for i, c in enumerate(bin_counts) if c > 0]
    pop_accs = [bin_accs[i] for i, c in enumerate(bin_counts) if c > 0]
    pop_widths = [widths[i] for i, c in enumerate(bin_counts) if c > 0]

    ax.bar(
        pop_centers,
        pop_accs,
        width=pop_widths,
        edgecolor="black",
        alpha=0.6,
        color="steelblue",
        label="Observed Accuracy",
    )
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect Calibration")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"{title}\nECE: {ece:.4f} | Brier: {brier:.4f}")
    ax.legend(loc="upper left")
    ax.grid(True, linestyle=":", alpha=0.5)

    save_path = Path(out_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
