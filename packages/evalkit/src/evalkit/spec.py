from __future__ import annotations

import json
import string
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


class SpecError(Exception):
    """Raised when a task specification or its dataset fails validation."""


TaskType = Literal["generation", "multiple_choice", "code", "classification"]


def extract_template_variables(template: str) -> set[str]:
    """Extract variable names from a python format string template."""
    formatter = string.Formatter()
    variables: set[str] = set()
    for _, field_name, _, _ in formatter.parse(template):
        if field_name:
            root = field_name.split(".")[0].split("[")[0]
            variables.add(root)
    return variables


class TaskSpec(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: str
    version: str = "1.0.0"
    type: TaskType
    dataset: Path
    prompt_template: str
    metrics: list[str]
    few_shot: int = 0
    few_shot_examples: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str | Path, validate_dataset: bool = True) -> TaskSpec:
        yaml_path = Path(path).resolve()
        if not yaml_path.is_file():
            raise SpecError(f"Task spec file not found: {yaml_path}")
        try:
            with open(yaml_path, encoding="utf-8") as f:
                data = yaml.safe_load(f)
        except Exception as exc:
            raise SpecError(f"Failed to parse YAML from {yaml_path}: {exc}") from exc

        if not isinstance(data, dict):
            raise SpecError(f"YAML content must be a mapping, got {type(data).__name__}")

        raw_dataset = data.get("dataset")
        if raw_dataset is None:
            raise SpecError("Task spec is missing required 'dataset' field")

        dataset_path = Path(raw_dataset)
        if not dataset_path.is_absolute():
            dataset_path = (yaml_path.parent / dataset_path).resolve()
        data["dataset"] = dataset_path

        try:
            spec = cls.model_validate(data)
        except Exception as exc:
            raise SpecError(f"Validation failed for TaskSpec {yaml_path}: {exc}") from exc

        if validate_dataset:
            spec.load_dataset()

        return spec

    def required_variables(self) -> set[str]:
        return extract_template_variables(self.prompt_template)

    def load_dataset(self) -> list[dict[str, Any]]:
        path = self.dataset
        if not path.is_file():
            raise SpecError(f"Dataset file not found: {path}")

        rows: list[dict[str, Any]] = []
        try:
            if path.suffix.lower() == ".jsonl":
                with open(path, encoding="utf-8") as f:
                    for line_no, line in enumerate(f, start=1):
                        line = line.strip()
                        if not line:
                            continue
                        item = json.loads(line)
                        if not isinstance(item, dict):
                            raise SpecError(
                                f"Row {line_no} in {path} must be a JSON object, "
                                f"got {type(item).__name__}"
                            )
                        rows.append(item)
            elif path.suffix.lower() in (".json",):
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
                if not isinstance(data, list):
                    raise SpecError(f"JSON dataset at {path} must contain a list of objects")
                for i, item in enumerate(data):
                    if not isinstance(item, dict):
                        raise SpecError(f"Item {i} in {path} must be a dictionary")
                    rows.append(item)
            else:
                raise SpecError(f"Unsupported dataset format '{path.suffix}'. Use .jsonl or .json")
        except json.JSONDecodeError as exc:
            raise SpecError(f"Failed to decode JSON from {path}: {exc}") from exc

        if not rows:
            raise SpecError(f"Dataset at {path} contains no examples")

        req_vars = self.required_variables()
        for idx, row in enumerate(rows):
            missing = req_vars - set(row.keys())
            if missing:
                missing_str = ", ".join(sorted(missing))
                raise SpecError(
                    f"Dataset row {idx} in {path} is missing template variable(s): {missing_str}"
                )

        return rows

    def render_prompt(self, row: dict[str, Any]) -> str:
        req_vars = self.required_variables()
        missing = req_vars - set(row.keys())
        if missing:
            raise SpecError(f"Row missing required variables: {', '.join(sorted(missing))}")

        prefix = ""
        if self.few_shot > 0 and self.few_shot_examples:
            examples = self.few_shot_examples[: self.few_shot]
            blocks: list[str] = []
            for ex in examples:
                ex_rendered = self.prompt_template.format(**ex)
                target = ex.get("target") or (ex.get("targets") or [""])[0]
                blocks.append(f"{ex_rendered} {target}".strip())
            prefix = "\n\n".join(blocks) + "\n\n"

        main_prompt = self.prompt_template.format(**row)
        return f"{prefix}{main_prompt}"
