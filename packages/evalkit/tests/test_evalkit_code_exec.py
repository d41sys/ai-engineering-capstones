from __future__ import annotations

import math
import time

import pytest
from evalkit.metrics.code_exec import (
    estimate_pass_at_k,
    execute_code,
    extract_code,
    run_code_exec,
)


def test_estimate_pass_at_k_closed_form() -> None:
    # Acceptance criteria: n=10, c=3, k=1 -> 0.3
    val = estimate_pass_at_k(n=10, c=3, k=1)
    assert math.isclose(val, 0.3, rel_tol=1e-9)

    # k=2: 1 - (7/10)*(6/9) = 1 - 42/90 = 48/90 ≈ 0.5333333333
    val_k2 = estimate_pass_at_k(n=10, c=3, k=2)
    assert math.isclose(val_k2, 48.0 / 90.0, rel_tol=1e-9)

    # Edge cases
    assert estimate_pass_at_k(n=10, c=0, k=1) == 0.0
    assert estimate_pass_at_k(n=10, c=10, k=1) == 1.0
    assert estimate_pass_at_k(n=10, c=10, k=5) == 1.0
    assert estimate_pass_at_k(n=5, c=4, k=3) == 1.0  # n-c = 1 < k=3 -> 1.0

    with pytest.raises(ValueError):
        estimate_pass_at_k(n=5, c=6, k=1)
    with pytest.raises(ValueError):
        estimate_pass_at_k(n=5, c=2, k=6)


def test_code_exec_extract_code() -> None:
    fenced = "```python\ndef add(a, b):\n    return a + b\n```"
    assert extract_code(fenced) == "def add(a, b):\n    return a + b"

    bare = "def add(a, b):\n    return a + b"
    assert extract_code(bare) == bare


def test_code_exec_pass() -> None:
    code = "def add(a, b):\n    return a + b"
    tests = ["assert add(1, 2) == 3", "assert add(-1, 1) == 0"]
    result = run_code_exec(code, tests, timeout_s=5.0)
    assert result.status == "pass"
    assert result.passed is True


def test_code_exec_fail() -> None:
    code = "def add(a, b):\n    return a - b"
    tests = ["assert add(1, 2) == 3"]
    result = run_code_exec(code, tests, timeout_s=5.0)
    assert result.status == "fail"
    assert result.passed is False


def test_code_exec_error() -> None:
    # Syntax error
    code_syntax_err = "def add(a, b)\n    return a + b"
    tests = ["assert add(1, 2) == 3"]
    assert execute_code(code_syntax_err, tests, timeout_s=5.0) == "error"

    # Runtime error
    code_runtime_err = "def add(a, b):\n    raise RuntimeError('boom')"
    assert execute_code(code_runtime_err, tests, timeout_s=5.0) == "error"


def test_code_exec_timeout_within_12_seconds() -> None:
    # Acceptance criteria: 'while True: pass' returns timeout within 12 s
    code = "while True:\n    pass"
    tests = ["assert True"]
    t0 = time.time()
    status = execute_code(code, tests, timeout_s=10.0)
    elapsed = time.time() - t0

    assert status == "timeout"
    assert elapsed < 12.0
