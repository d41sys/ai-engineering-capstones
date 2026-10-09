from __future__ import annotations

import json
from pathlib import Path

import pytest
from evalkit.spec import SpecError, TaskSpec, extract_template_variables


def test_extract_template_variables() -> None:
    template = "Question: {question}\nContext: {context}\nAnswer:"
    vars_found = extract_template_variables(template)
    assert vars_found == {"question", "context"}


def test_task_spec_valid(tmp_path: Path) -> None:
    data_file = tmp_path / "data.jsonl"
    data_file.write_text(
        json.dumps({"question": "What is 2+2?", "target": "4"}) + "\n",
        encoding="utf-8",
    )
    spec_file = tmp_path / "spec.yaml"
    spec_file.write_text(
        f"""
name: test_task
version: 1.0.0
type: generation
dataset: {data_file.name}
prompt_template: "Q: {{question}}\\nA:"
metrics:
  - exact_match
""",
        encoding="utf-8",
    )

    spec = TaskSpec.from_yaml(spec_file)
    assert spec.name == "test_task"
    assert spec.type == "generation"
    rows = spec.load_dataset()
    assert len(rows) == 1
    assert spec.render_prompt(rows[0]) == "Q: What is 2+2?\nA:"


def test_task_spec_missing_variable_raises_spec_error(tmp_path: Path) -> None:
    data_file = tmp_path / "data.jsonl"
    # Row is missing 'context' which is in prompt_template
    data_file.write_text(
        json.dumps({"question": "What is 2+2?", "target": "4"}) + "\n",
        encoding="utf-8",
    )
    spec_file = tmp_path / "spec.yaml"
    spec_file.write_text(
        f"""
name: test_task_invalid
version: 1.0.0
type: generation
dataset: {data_file.name}
prompt_template: "Q: {{question}}\\nContext: {{context}}\\nA:"
metrics:
  - exact_match
""",
        encoding="utf-8",
    )

    with pytest.raises(SpecError, match="missing template variable"):
        TaskSpec.from_yaml(spec_file)


def test_task_spec_missing_dataset_raises_spec_error(tmp_path: Path) -> None:
    spec_file = tmp_path / "spec.yaml"
    spec_file.write_text(
        """
name: test_task_no_data
version: 1.0.0
type: generation
dataset: non_existent.jsonl
prompt_template: "Q: {q}\\nA:"
metrics:
  - exact_match
""",
        encoding="utf-8",
    )

    with pytest.raises(SpecError, match="Dataset file not found"):
        TaskSpec.from_yaml(spec_file)


def test_task_spec_few_shot_rendering(tmp_path: Path) -> None:
    data_file = tmp_path / "data.jsonl"
    data_file.write_text(
        json.dumps({"q": "What is 3+3?", "target": "6"}) + "\n",
        encoding="utf-8",
    )
    spec_file = tmp_path / "spec.yaml"
    spec_file.write_text(
        f"""
name: test_few_shot
version: 1.0.0
type: generation
dataset: {data_file.name}
prompt_template: "Q: {{q}}\\nA:"
metrics:
  - exact_match
few_shot: 1
few_shot_examples:
  - q: "What is 1+1?"
    target: "2"
""",
        encoding="utf-8",
    )
    spec = TaskSpec.from_yaml(spec_file)
    row = spec.load_dataset()[0]
    prompt = spec.render_prompt(row)
    assert "Q: What is 1+1?\nA: 2" in prompt
    assert "Q: What is 3+3?\nA:" in prompt
