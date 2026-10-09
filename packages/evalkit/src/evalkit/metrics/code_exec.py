from __future__ import annotations

import contextlib
import os
import re
import signal
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

try:
    import resource
except ImportError:
    resource = None  # type: ignore[assignment]


ExecStatus = Literal["pass", "fail", "timeout", "error"]

CODE_BLOCK_RE = re.compile(r"```(?:python|py)?\n(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_code(text: str) -> str:
    """Extract code from a markdown block if present, else return text as-is."""
    matches = CODE_BLOCK_RE.findall(text)
    if matches:
        return str(matches[0]).strip()
    # Strip any bare fences
    stripped = text.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 2:
            return "\n".join(lines[1:-1]).strip()
    return stripped


@dataclass
class CodeExecResult:
    status: ExecStatus
    stdout: str = ""
    stderr: str = ""

    @property
    def passed(self) -> bool:
        return self.status == "pass"


_RUNNER_TEMPLATE = r"""
import sys

def main():
    candidate_code = __CANDIDATE_CODE__
    test_suite = __TEST_SUITE__

    try:
        compiled_candidate = compile(candidate_code, "<candidate>", "exec")
    except Exception as exc:
        sys.stderr.write(f"SYNTAX_ERROR: {exc}\n")
        sys.exit(2)

    ns = {"__name__": "__main__"}
    try:
        exec(compiled_candidate, ns)
    except Exception as exc:
        sys.stderr.write(f"EXEC_ERROR: {exc}\n")
        sys.exit(2)

    if isinstance(test_suite, list):
        for assertion in test_suite:
            try:
                exec(assertion, ns)
            except AssertionError as exc:
                sys.stderr.write(f"ASSERTION_FAIL: {exc}\n")
                sys.exit(1)
            except Exception as exc:
                sys.stderr.write(f"TEST_ERROR: {exc}\n")
                sys.exit(2)
    elif isinstance(test_suite, str):
        try:
            compiled_tests = compile(test_suite, "<tests>", "exec")
        except Exception as exc:
            sys.stderr.write(f"TEST_SYNTAX_ERROR: {exc}\n")
            sys.exit(2)
        try:
            exec(compiled_tests, ns)
        except AssertionError as exc:
            sys.stderr.write(f"ASSERTION_FAIL: {exc}\n")
            sys.exit(1)
        except Exception as exc:
            sys.stderr.write(f"TEST_ERROR: {exc}\n")
            sys.exit(2)

    sys.exit(0)

if __name__ == "__main__":
    main()
"""


def _make_preexec_fn() -> Any:
    """Apply CPU and address space limits in child process."""
    if resource is None or os.name == "nt":
        return None

    def _set_limits() -> None:
        with contextlib.suppress(ValueError, OSError, AttributeError):
            resource.setrlimit(resource.RLIMIT_CPU, (5, 5))

        # Skip RLIMIT_AS on macOS (unsupported or causes immediate SIGSEGV during python init)
        if sys.platform != "darwin":
            with contextlib.suppress(ValueError, OSError, AttributeError):
                as_limit = 512 * 1024 * 1024  # 512 MB
                resource.setrlimit(resource.RLIMIT_AS, (as_limit, as_limit))

    return _set_limits


def run_code_exec(
    candidate_code: str,
    tests: Sequence[str] | str,
    timeout_s: float = 10.0,
) -> CodeExecResult:
    """Run candidate code plus its tests in an isolated subprocess.

    Returns CodeExecResult with status in 'pass', 'fail', 'timeout', 'error'.
    """
    cleaned_code = extract_code(candidate_code)
    test_obj = list(tests) if not isinstance(tests, str) else tests

    script_content = _RUNNER_TEMPLATE
    script_content = script_content.replace("__CANDIDATE_CODE__", repr(cleaned_code))
    script_content = script_content.replace("__TEST_SUITE__", repr(test_obj))

    with tempfile.TemporaryDirectory() as tmp_dir:
        runner_file = Path(tmp_dir) / "run_eval.py"
        runner_file.write_text(script_content, encoding="utf-8")

        preexec = _make_preexec_fn()
        try:
            proc = subprocess.run(
                [sys.executable, "-I", str(runner_file)],
                cwd=tmp_dir,
                capture_output=True,
                text=True,
                timeout=timeout_s,
                preexec_fn=preexec,
            )
            stdout = proc.stdout
            stderr = proc.stderr
            code = proc.returncode

            if code == 0:
                status: ExecStatus = "pass"
            elif code == 1:
                status = "fail"
            elif code == 2:
                status = "error"
            else:
                # Check if killed by SIGXCPU (CPU limit reached)
                sig_xcpu = getattr(signal, "SIGXCPU", None)
                if sig_xcpu is not None and (code == -sig_xcpu or code == 128 + sig_xcpu):
                    status = "timeout"
                else:
                    status = "error"

            return CodeExecResult(status=status, stdout=stdout, stderr=stderr)

        except subprocess.TimeoutExpired as exc:
            return CodeExecResult(
                status="timeout",
                stdout=exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or ""),
                stderr=exc.stderr.decode() if isinstance(exc.stderr, bytes) else (exc.stderr or ""),
            )
        except Exception as exc:
            return CodeExecResult(status="error", stderr=str(exc))


def execute_code(
    candidate_code: str,
    tests: Sequence[str] | str,
    timeout_s: float = 10.0,
) -> ExecStatus:
    """Convenience wrapper returning only the execution status string."""
    return run_code_exec(candidate_code, tests, timeout_s=timeout_s).status


def estimate_pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k estimator from Codex paper (Chen et al. 2021).

    Formula: 1 - prod_{i=1}^k (n - c - i + 1) / (n - i + 1)
    """
    if n < k:
        raise ValueError(f"Sample size n={n} must be >= k={k}")
    if c < 0 or c > n:
        raise ValueError(f"Passed count c={c} must be in [0, {n}]")
    if n - c < k:
        return 1.0
    prod = 1.0
    for i in range(1, k + 1):
        prod *= (n - c - i + 1) / (n - i + 1)
    return float(1.0 - prod)


def corpus_pass_at_k(results: Sequence[tuple[int, int]], k: int) -> float:
    """Calculate mean pass@k over a corpus of problems.

    results: sequence of (n, c) tuples where n is number of samples and c is number of passes.
    """
    if not results:
        return 0.0
    scores = [estimate_pass_at_k(n, c, k) for n, c in results]
    return float(sum(scores) / len(scores))
