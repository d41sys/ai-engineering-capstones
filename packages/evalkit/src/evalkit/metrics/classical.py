from __future__ import annotations

import math
import re
import string
from collections import Counter
from collections.abc import Sequence

TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def normalize_text(text: str) -> str:
    """Normalize text: lowercase, strip articles, remove punctuation, collapse whitespace."""
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = "".join(ch for ch in text if ch not in string.punctuation)
    return " ".join(text.split())


def tokenize(text: str) -> list[str]:
    """Tokenize text into lowercase alphanumeric runs."""
    return TOKEN_RE.findall(text.lower())


def exact_match(prediction: str, targets: Sequence[str] | str) -> float:
    """Exact match on normalized text."""
    target_list = [targets] if isinstance(targets, str) else list(targets)
    if not target_list:
        return 0.0
    norm_pred = normalize_text(prediction)
    return 1.0 if any(norm_pred == normalize_text(t) for t in target_list) else 0.0


def accuracy(prediction: str, targets: Sequence[str] | str) -> float:
    """Accuracy alias for exact match on single/multiple targets."""
    return exact_match(prediction, targets)


def single_token_f1(prediction: str, target: str) -> float:
    """Token-level F1 score between prediction and a single target string."""
    pred_tokens = tokenize(prediction)
    tgt_tokens = tokenize(target)
    if not pred_tokens and not tgt_tokens:
        return 1.0
    if not pred_tokens or not tgt_tokens:
        return 0.0
    pred_counts = Counter(pred_tokens)
    tgt_counts = Counter(tgt_tokens)
    overlap = sum((pred_counts & tgt_counts).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(tgt_tokens)
    return float(2.0 * precision * recall / (precision + recall))


def f1_score(prediction: str, targets: Sequence[str] | str) -> float:
    """Maximum token-level F1 score across all targets."""
    target_list = [targets] if isinstance(targets, str) else list(targets)
    if not target_list:
        return 0.0
    return max(single_token_f1(prediction, t) for t in target_list)


def _ngram_counts(tokens: list[str], n: int) -> Counter[tuple[str, ...]]:
    if n <= 0 or len(tokens) < n:
        return Counter()
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def _modified_precision(cand_tokens: list[str], ref_tokens: list[str], n: int) -> tuple[int, int]:
    cand_ngrams = _ngram_counts(cand_tokens, n)
    if not cand_ngrams:
        return (0, 0)
    ref_ngrams = _ngram_counts(ref_tokens, n)
    clipped = 0
    for gram, count in cand_ngrams.items():
        clipped += min(count, ref_ngrams.get(gram, 0))
    total = sum(cand_ngrams.values())
    return (clipped, total)


def _brevity_penalty(cand_len: int, ref_len: int) -> float:
    if cand_len == 0:
        return 0.0
    if cand_len >= ref_len:
        return 1.0
    return math.exp(1.0 - ref_len / cand_len)


def single_bleu_4(prediction: str, reference: str, max_n: int = 4) -> float:
    """BLEU-4 with brevity penalty and Lin-Och method 1 (+1 additive) smoothing."""
    cand = tokenize(prediction)
    ref = tokenize(reference)
    if not cand:
        return 0.0
    log_p_sum = 0.0
    for n in range(1, max_n + 1):
        clipped, total = _modified_precision(cand, ref, n)
        smoothed_num = clipped + 1
        smoothed_den = total + 1
        log_p_sum += math.log(smoothed_num / smoothed_den)
    geo_mean = math.exp(log_p_sum / max_n)
    bp = _brevity_penalty(len(cand), len(ref))
    return float(bp * geo_mean)


def bleu_4(prediction: str, targets: Sequence[str] | str, max_n: int = 4) -> float:
    """Maximum BLEU-4 across targets."""
    target_list = [targets] if isinstance(targets, str) else list(targets)
    if not target_list:
        return 0.0
    return max(single_bleu_4(prediction, t, max_n=max_n) for t in target_list)


def lcs_length(a: list[str], b: list[str]) -> int:
    """Compute length of Longest Common Subsequence between two token sequences."""
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return 0
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n):
        ai = a[i]
        for j in range(m):
            if ai == b[j]:
                dp[i + 1][j + 1] = dp[i][j] + 1
            else:
                dp[i + 1][j + 1] = max(dp[i + 1][j], dp[i][j + 1])
    return dp[n][m]


def single_rouge_l(prediction: str, reference: str, beta: float = 1.0) -> float:
    """ROUGE-L F-measure based on token-level Longest Common Subsequence."""
    cand = tokenize(prediction)
    ref = tokenize(reference)
    if not cand and not ref:
        return 1.0
    if not cand or not ref:
        return 0.0
    lcs = lcs_length(cand, ref)
    if lcs == 0:
        return 0.0
    precision = lcs / len(cand)
    recall = lcs / len(ref)
    beta_sq = beta * beta
    denom = recall + beta_sq * precision
    if denom == 0:
        return 0.0
    return float((1 + beta_sq) * precision * recall / denom)


def rouge_l(prediction: str, targets: Sequence[str] | str, beta: float = 1.0) -> float:
    """Maximum ROUGE-L across targets."""
    target_list = [targets] if isinstance(targets, str) else list(targets)
    if not target_list:
        return 0.0
    return max(single_rouge_l(prediction, t, beta=beta) for t in target_list)


def _char_ngrams(text: str, n: int, remove_whitespace: bool = True) -> Counter[str]:
    chars = "".join(text.split()) if remove_whitespace else text
    if len(chars) < n or n <= 0:
        return Counter()
    return Counter(chars[i : i + n] for i in range(len(chars) - n + 1))


def single_chrf(
    prediction: str,
    reference: str,
    char_order: int = 6,
    beta: float = 2.0,
    remove_whitespace: bool = True,
) -> float:
    """Character n-gram F-score (Popovic 2015)."""
    if prediction == reference:
        return 1.0

    precisions: list[float] = []
    recalls: list[float] = []

    for n in range(1, char_order + 1):
        cand_ngrams = _char_ngrams(prediction, n, remove_whitespace=remove_whitespace)
        ref_ngrams = _char_ngrams(reference, n, remove_whitespace=remove_whitespace)

        cand_total = sum(cand_ngrams.values())
        ref_total = sum(ref_ngrams.values())

        if cand_total == 0 and ref_total == 0:
            continue

        clipped = sum(min(c, ref_ngrams.get(g, 0)) for g, c in cand_ngrams.items())
        p_n = clipped / cand_total if cand_total > 0 else 0.0
        r_n = clipped / ref_total if ref_total > 0 else 0.0

        precisions.append(p_n)
        recalls.append(r_n)

    if not precisions:
        return 1.0 if not prediction and not reference else 0.0

    avg_p = sum(precisions) / len(precisions)
    avg_r = sum(recalls) / len(recalls)

    if avg_p + avg_r == 0:
        return 0.0

    beta_sq = beta * beta
    denom = beta_sq * avg_p + avg_r
    if denom == 0:
        return 0.0
    return float((1 + beta_sq) * avg_p * avg_r / denom)


def chrf(
    prediction: str,
    targets: Sequence[str] | str,
    char_order: int = 6,
    beta: float = 2.0,
    remove_whitespace: bool = True,
) -> float:
    """Maximum chrF across targets."""
    target_list = [targets] if isinstance(targets, str) else list(targets)
    if not target_list:
        return 0.0
    return max(
        single_chrf(
            prediction,
            t,
            char_order=char_order,
            beta=beta,
            remove_whitespace=remove_whitespace,
        )
        for t in target_list
    )


def score_classical(metric_name: str, prediction: str, targets: Sequence[str] | str) -> float:
    """Dispatch classical metric by name."""
    name = metric_name.lower().strip()
    if name in ("exact_match", "em"):
        return exact_match(prediction, targets)
    if name in ("accuracy", "acc"):
        return accuracy(prediction, targets)
    if name in ("f1", "token_f1"):
        return f1_score(prediction, targets)
    if name in ("bleu", "bleu_4", "bleu4"):
        return bleu_4(prediction, targets)
    if name in ("rouge", "rouge_l", "rougel"):
        return rouge_l(prediction, targets)
    if name == "chrf":
        return chrf(prediction, targets)
    raise ValueError(f"Unknown classical metric: {metric_name}")
