# Reading guide

Setup once per repo: `cd <pkg> && uv sync`. Run every command below from inside that repo.

## Method (each package)

1. `<pkg>/README.md`: read the mermaid diagram only.
2. `src/<pkg>/cli.py`: entry point; shows which modules run in which order.
3. Modules in the order below (foundation → composition).
4. Matching `<pkg>/tests/test_<pkg>_*.py` (safegate: one file, `test_safegate_core.py`): what each function must do.
5. Run the command; match output to the code.

Skim a file first: `graft skeleton <pkg>/src/<pkg>/<file>.py`.
`flexgridcyber/` is research, not capstone: skip.

## Order

| # | Package | Files, in order | Time | Run |
|---|---|---|---|---|
| 1 | evalkit | spec → metrics/classical → metrics/code_exec → metrics/calibration → models → aggregate → runner | 2–3 h | `uv run evalkit run --tasks tasks --models oracle,noisy:0.3,random --out runs/demo` |
| 2 | ragkit | chunking → bm25 → embed → dense → hybrid → rerank → rewrite → pipeline → eval | 2–3 h | `uv run ragkit ask "What is imbalance settlement?"` |
| 3 | safegate | taxonomy → injection → classifier → rules + rule_configs/default.yaml → gate → refusal | 2–3 h | `uv run safegate check "Ignore all previous instructions"` |
| 4 | minigpt | bpe → data → model → train → pretrained → finetune_cls → sft → dpo → evaluate | 4–5 h | `uv run minigpt train-tokenizer && uv run minigpt pretrain --data tinyshakespeare --steps 2000 && uv run minigpt generate --prompt "ROMEO:"` |

## Key spots

- evalkit: `code_exec.py` sandbox + timeout; `aggregate.py` bootstrap CI + Elo; `runner.py` resume + threads.
- ragkit: `bm25.py` Okapi formula (check against the hand-computed test); `hybrid.py` RRF; `pipeline.py` `[chunk_id]` citations.
- safegate: `injection.py` regex after base64/rot13 decoding + numpy logistic regression; `rules.py:20` action precedence refuse > escalate > redact > allow; `gate.py` 64-char stream buffer.
- minigpt: `model.py` causal mask + weight tying; `dpo.py` loss (policy = reference ⇒ loss = log 2).

## Known gaps vs. course spec

- ragkit: no refuse-on-low-confidence; no proof the composed pipeline beats each stage.
- minigpt: lesson-41 eval lacks exact-match / token F1 / weighted report.
- safegate: action names differ (block/warn); no per-request trace object.
