from __future__ import annotations

import os
import random
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class Prediction:
    text: str
    confidence: float = 0.5
    logprobs: list[float] = field(default_factory=list)
    token_nll: float = 0.0

    @property
    def token_count(self) -> int:
        return max(1, len(self.text.split()))


@runtime_checkable
class Model(Protocol):
    name: str

    def generate(self, prompt: str, **kwargs: Any) -> str: ...

    def logprobs(self, prompt: str, candidate: str) -> list[float]: ...

    def choose(self, prompt: str, choices: Sequence[str], **kwargs: Any) -> str: ...

    def predict(
        self,
        prompt: str,
        task_type: str,
        choices: Sequence[str] | None = None,
        example: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Prediction: ...


def _get_target_from_example(example: dict[str, Any] | None) -> str:
    if not example:
        return ""
    if "canonical_solution" in example:
        return str(example["canonical_solution"]).strip()
    if example.get("targets"):
        return str(example["targets"][0]).strip()
    if "target" in example:
        return str(example["target"]).strip()
    if "answer" in example:
        return str(example["answer"]).strip()
    return ""


class OracleModel:
    """Perfect model that always outputs ground truth when available."""

    def __init__(self, name: str = "oracle") -> None:
        self.name = name

    def generate(self, prompt: str, **kwargs: Any) -> str:
        example = kwargs.get("example")
        ans = _get_target_from_example(example)
        return ans if ans else "correct"

    def logprobs(self, prompt: str, candidate: str) -> list[float]:
        tokens = max(1, len(candidate.split()))
        return [-0.02] * tokens

    def choose(self, prompt: str, choices: Sequence[str], **kwargs: Any) -> str:
        example = kwargs.get("example")
        ans = _get_target_from_example(example)
        if ans:
            # Check if ans is choice letter like 'A', 'B'
            if ans in choices:
                return ans
            # Or match against text of choices
            for ch in choices:
                if ans.lower() in ch.lower() or ch.lower() in ans.lower():
                    return ch
        return choices[0] if choices else ""

    def predict(
        self,
        prompt: str,
        task_type: str,
        choices: Sequence[str] | None = None,
        example: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Prediction:
        if task_type == "multiple_choice" and choices:
            text = self.choose(prompt, choices, example=example, **kwargs)
        else:
            text = self.generate(prompt, example=example, **kwargs)

        lps = self.logprobs(prompt, text)
        confidence = 0.99
        nll = float(-sum(lps))
        return Prediction(text=text, confidence=confidence, logprobs=lps, token_nll=nll)


class NoisyModel:
    """Outputs ground truth with probability (1 - noise_rate); otherwise corrupted output."""

    def __init__(self, noise_rate: float = 0.3, seed: int = 42, name: str | None = None) -> None:
        self.noise_rate = float(noise_rate)
        self.rng = random.Random(seed)
        self.name = name or f"noisy:{self.noise_rate}"

    def generate(self, prompt: str, **kwargs: Any) -> str:
        example = kwargs.get("example")
        if self.rng.random() >= self.noise_rate:
            ans = _get_target_from_example(example)
            if ans:
                return ans
        # Corrupted output
        if example and "canonical_solution" in example:
            return "    return None"
        if example and example.get("type") == "classification":
            target = _get_target_from_example(example).lower()
            return "negative" if target == "positive" else "positive"
        return "I am not certain"

    def logprobs(self, prompt: str, candidate: str) -> list[float]:
        tokens = max(1, len(candidate.split()))
        return [-0.4] * tokens

    def choose(self, prompt: str, choices: Sequence[str], **kwargs: Any) -> str:
        example = kwargs.get("example")
        target = _get_target_from_example(example)
        correct_choice = choices[0] if choices else ""
        if target:
            for ch in choices:
                if target.lower() in ch.lower() or ch.lower() in target.lower():
                    correct_choice = ch
                    break

        if self.rng.random() >= self.noise_rate and correct_choice:
            return correct_choice

        # Pick wrong choice
        wrong_choices = [c for c in choices if c != correct_choice]
        if wrong_choices:
            return self.rng.choice(wrong_choices)
        return self.rng.choice(choices) if choices else ""

    def predict(
        self,
        prompt: str,
        task_type: str,
        choices: Sequence[str] | None = None,
        example: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Prediction:
        if task_type == "multiple_choice" and choices:
            text = self.choose(prompt, choices, example=example, **kwargs)
        else:
            text = self.generate(prompt, example=example, **kwargs)

        lps = self.logprobs(prompt, text)
        confidence = float(self.rng.uniform(0.65, 0.85))
        nll = float(-sum(lps))
        return Prediction(text=text, confidence=confidence, logprobs=lps, token_nll=nll)


class RandomModel:
    """Model that outputs random guesses."""

    def __init__(self, seed: int = 42, name: str = "random") -> None:
        self.rng = random.Random(seed)
        self.name = name

    def generate(self, prompt: str, **kwargs: Any) -> str:
        example = kwargs.get("example")
        if example and "canonical_solution" in example:
            return "    pass"
        if example and example.get("type") == "classification":
            return self.rng.choice(["positive", "negative"])
        return str(self.rng.randint(0, 100))

    def logprobs(self, prompt: str, candidate: str) -> list[float]:
        tokens = max(1, len(candidate.split()))
        return [-2.0] * tokens

    def choose(self, prompt: str, choices: Sequence[str], **kwargs: Any) -> str:
        return self.rng.choice(choices) if choices else ""

    def predict(
        self,
        prompt: str,
        task_type: str,
        choices: Sequence[str] | None = None,
        example: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Prediction:
        if task_type == "multiple_choice" and choices:
            text = self.choose(prompt, choices, example=example, **kwargs)
        else:
            text = self.generate(prompt, example=example, **kwargs)

        lps = self.logprobs(prompt, text)
        confidence = float(self.rng.uniform(0.2, 0.45))
        nll = float(-sum(lps))
        return Prediction(text=text, confidence=confidence, logprobs=lps, token_nll=nll)


class AnthropicModel:
    """Real LLM model calling Anthropic API (requires extra [llm])."""

    def __init__(
        self,
        model_id: str = "claude-3-5-sonnet-20241022",
        api_key: str | None = None,
        name: str | None = None,
    ) -> None:
        try:
            import anthropic
        except ImportError as exc:
            raise ImportError(
                "anthropic package not found. Install evalkit with `evalkit[llm]`."
            ) from exc

        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY environment variable is missing and no api_key was provided."
            )

        self.client = anthropic.Anthropic(api_key=key)
        self.model_id = model_id
        self.name = name or f"anthropic:{model_id}"

    def generate(self, prompt: str, **kwargs: Any) -> str:
        max_tokens = kwargs.get("max_tokens", 256)
        temp = kwargs.get("temperature", 0.0)
        resp = self.client.messages.create(
            model=self.model_id,
            max_tokens=max_tokens,
            temperature=temp,
            messages=[{"role": "user", "content": prompt}],
        )
        content_block = resp.content[0]
        return getattr(content_block, "text", "")

    def logprobs(self, prompt: str, candidate: str) -> list[float]:
        # Anthropic API does not expose token logprobs; return proxy estimate
        tokens = max(1, len(candidate.split()))
        return [-0.5] * tokens

    def choose(self, prompt: str, choices: Sequence[str], **kwargs: Any) -> str:
        choice_list = ", ".join(choices)
        formatted_prompt = (
            f"{prompt}\n\nChoices: {choice_list}\n"
            "Respond with only the selected choice letter or text."
        )
        ans = self.generate(formatted_prompt, **kwargs).strip()
        for ch in choices:
            if ans.lower() == ch.lower() or ans.startswith(ch):
                return ch
        return ans

    def predict(
        self,
        prompt: str,
        task_type: str,
        choices: Sequence[str] | None = None,
        example: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Prediction:
        if task_type == "multiple_choice" and choices:
            text = self.choose(prompt, choices, example=example, **kwargs)
        else:
            text = self.generate(prompt, example=example, **kwargs)

        lps = self.logprobs(prompt, text)
        confidence = 0.8
        nll = float(-sum(lps))
        return Prediction(text=text, confidence=confidence, logprobs=lps, token_nll=nll)


def get_model(spec_str: str, **kwargs: Any) -> Model:
    """Parse model descriptor string like 'oracle', 'noisy:0.3', 'random', 'anthropic:model_id'."""
    raw = spec_str.strip()
    if raw == "oracle":
        return OracleModel()
    if raw.startswith("noisy"):
        parts = raw.split(":")
        rate = float(parts[1]) if len(parts) > 1 else 0.3
        return NoisyModel(noise_rate=rate, **kwargs)
    if raw == "random":
        return RandomModel(**kwargs)
    if raw.startswith("anthropic"):
        parts = raw.split(":", 1)
        model_id = parts[1] if len(parts) > 1 else "claude-3-5-sonnet-20241022"
        return AnthropicModel(model_id=model_id, **kwargs)
    raise ValueError(f"Unknown model descriptor: {spec_str}")
