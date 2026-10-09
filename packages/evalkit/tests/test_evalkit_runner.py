from __future__ import annotations

import json
from pathlib import Path

from evalkit.models import NoisyModel, OracleModel, RandomModel
from evalkit.runner import run
from evalkit.spec import TaskSpec


def test_runner_execution_and_resumability(tmp_path: Path) -> None:
    # 1. Create a small mock task
    data_file = tmp_path / "data.jsonl"
    rows = [
        {
            "id": f"q_{i}",
            "question": f"What is {i}+{i}?",
            "target": str(i + i),
            "targets": [str(i + i)],
        }
        for i in range(5)
    ]
    with open(data_file, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    spec_file = tmp_path / "spec.yaml"
    spec_file.write_text(
        f"""
name: test_arithmetic
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
    models = [OracleModel(), NoisyModel(noise_rate=0.4, seed=42), RandomModel(seed=42)]
    out_dir = tmp_path / "eval_out"

    # First run: should evaluate 5 examples * 3 models = 15 runs
    summary1 = run([spec], models, out_dir=out_dir, max_workers=2)
    assert summary1.total_evaluated == 15
    assert summary1.total_skipped == 0
    assert summary1.results_path.is_file()
    assert summary1.leaderboard_path.is_file()

    # Verify ranking: oracle > noisy > random
    ranked_names = [m.model_name for m in summary1.leaderboard.models]
    assert ranked_names[0] == "oracle"

    # Second run: Resume should skip all 15 runs!
    summary2 = run([spec], models, out_dir=out_dir, max_workers=2)
    assert summary2.total_evaluated == 0
    assert summary2.total_skipped == 15
    # Leaderboard still generated
    assert len(summary2.leaderboard.models) == 3
