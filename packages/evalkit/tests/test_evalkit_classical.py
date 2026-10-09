from __future__ import annotations

import math

import pytest
from evalkit.metrics.classical import (
    accuracy,
    chrf,
    exact_match,
    f1_score,
    lcs_length,
    normalize_text,
    score_classical,
    single_bleu_4,
    single_chrf,
    single_rouge_l,
    single_token_f1,
)


def test_normalize_text() -> None:
    # Lowercase, punctuation removed, articles removed, whitespace collapsed
    assert normalize_text("The quick, brown fox!") == "quick brown fox"
    assert normalize_text("A cat and AN apple.") == "cat and apple"
    assert normalize_text("   42   ") == "42"


def test_exact_match_normalized() -> None:
    # "The cat" normalized is "cat", matches "cat"
    assert exact_match("The cat", ["cat"]) == 1.0
    # Punctuation stripped: "42." matches "42"
    assert exact_match("42.", ["42"]) == 1.0
    # Mismatch
    assert exact_match("dog", ["cat", "bird"]) == 0.0
    # Accuracy is alias for exact match
    assert accuracy("The cat", ["cat"]) == 1.0


def test_token_f1_hand_computed() -> None:
    # pred: "the cat sat" (3 tokens), target: "the cat sat on the mat" (6 tokens)
    # multiset overlap: "the" (1), "cat" (1), "sat" (1) -> 3
    # precision = 3/3 = 1.0, recall = 3/6 = 0.5
    # F1 = 2 * 1.0 * 0.5 / (1.0 + 0.5) = 2/3
    score = single_token_f1("the cat sat", "the cat sat on the mat")
    assert math.isclose(score, 2.0 / 3.0, rel_tol=1e-5)

    # Empty cases
    assert single_token_f1("", "") == 1.0
    assert single_token_f1("hello", "") == 0.0
    assert f1_score("cat", ["dog", "cat"]) == 1.0


def test_bleu4_standard_example() -> None:
    # Exact match: cand == ref -> 1.0
    score_perfect = single_bleu_4("the cat sat on the mat", "the cat sat on the mat")
    assert math.isclose(score_perfect, 1.0, rel_tol=1e-5)

    # Papineni standard repeated tokens test:
    # cand = "the the the the", ref = "the cat sat on the mat"
    # cand has 4 tokens, ref has 6 tokens.
    # 1-grams: ("the",): count 4, clipped to 2. Smoothed: (2+1)/(4+1) = 3/5 = 0.6
    # 2-grams: ("the", "the"): count 3, clipped to 0. Smoothed: (0+1)/(3+1) = 1/4 = 0.25
    # 3-grams: ("the", "the", "the"): count 2, clipped to 0. Smoothed: (0+1)/(2+1) = 1/3
    # 4-grams: ("the", "the", "the", "the"): count 1, clipped to 0. Smoothed: (0+1)/(1+1) = 1/2
    # geo_mean = (0.6 * 0.25 * (1/3) * 0.5) ** 0.25 = 0.025 ** 0.25 ≈ 0.397635
    # BP = exp(1 - 6/4) = exp(-0.5) ≈ 0.606531
    # expected = BP * geo_mean ≈ 0.241178
    expected = math.exp(-0.5) * (0.025**0.25)
    score_repeat = single_bleu_4("the the the the", "the cat sat on the mat")
    assert math.isclose(score_repeat, expected, rel_tol=1e-5)
    assert score_repeat < 0.5


def test_rouge_l_lcs_hand_computed() -> None:
    # LCS DP table length check
    assert lcs_length(["the", "cat", "sat"], ["the", "cat", "sat", "on", "the", "mat"]) == 3
    assert lcs_length(["a", "b", "c", "d"], ["b", "c", "d", "e"]) == 3
    assert lcs_length(["x", "y"], ["a", "b"]) == 0

    # ROUGE-L score:
    # pred: "the cat sat" (len 3), ref: "the cat sat on the mat" (len 6)
    # LCS = 3 -> precision = 3/3 = 1.0, recall = 3/6 = 0.5
    # F1 = 2 * 1.0 * 0.5 / (1.0 + 0.5) = 2/3 ≈ 0.66667
    score_rl = single_rouge_l("the cat sat", "the cat sat on the mat")
    assert math.isclose(score_rl, 2.0 / 3.0, rel_tol=1e-5)

    # pred: "a b c d", ref: "b c d e"
    # precision = 3/4 = 0.75, recall = 3/4 = 0.75 -> F1 = 0.75
    score_rl2 = single_rouge_l("a b c d", "b c d e")
    assert math.isclose(score_rl2, 0.75, rel_tol=1e-5)


def test_chrf_hand_computed() -> None:
    # Hand-computed chrF example:
    # cand = "abc", ref = "abcde", char_order = 2, beta = 2.0
    # 1-grams: cand {a, b, c} (3), ref {a, b, c, d, e} (5) -> clipped=3, P1=1.0, R1=0.6
    # 2-grams: cand {ab, bc} (2), ref {ab, bc, cd, de} (4) -> clipped=2, P2=1.0, R2=0.5
    # avg_P = (1.0 + 1.0) / 2 = 1.0
    # avg_R = (0.6 + 0.5) / 2 = 0.55
    # beta=2.0 -> beta^2 = 4
    # numerator = (1 + 4) * 1.0 * 0.55 = 2.75
    # denominator = 4 * 1.0 + 0.55 = 4.55
    # chrF = 2.75 / 4.55 = 55 / 91 ≈ 0.6043956
    expected_beta2 = 2.75 / 4.55
    score_chrf2 = single_chrf("abc", "abcde", char_order=2, beta=2.0, remove_whitespace=True)
    assert math.isclose(score_chrf2, expected_beta2, rel_tol=1e-5)

    # beta=1.0 -> beta^2 = 1
    # numerator = 2 * 1.0 * 0.55 = 1.10
    # denominator = 1.0 + 0.55 = 1.55
    # chrF = 1.10 / 1.55 = 22 / 31 ≈ 0.7096774
    expected_beta1 = 1.10 / 1.55
    score_chrf1 = single_chrf("abc", "abcde", char_order=2, beta=1.0, remove_whitespace=True)
    assert math.isclose(score_chrf1, expected_beta1, rel_tol=1e-5)

    # Identical string -> 1.0
    assert chrf("hello world", ["hello world"]) == 1.0


def test_score_classical_dispatcher() -> None:
    assert score_classical("exact_match", "42", ["42"]) == 1.0
    assert score_classical("accuracy", "positive", ["positive"]) == 1.0
    assert score_classical("f1", "the cat", ["the cat"]) == 1.0
    assert score_classical("bleu_4", "the cat", ["the cat"]) == 1.0
    assert score_classical("rouge_l", "the cat", ["the cat"]) == 1.0
    assert score_classical("chrf", "test", ["test"]) == 1.0
    with pytest.raises(ValueError, match="Unknown classical metric"):
        score_classical("unknown_metric", "a", ["a"])
