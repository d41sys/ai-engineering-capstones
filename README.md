# AI Engineering Capstones

Four from-scratch AI engineering systems in one Python monorepo: a retrieval-augmented generation stack evaluated on an energy-flexibility corpus, a GPT trained and fine-tuned end to end (BPE → pretraining → classification → SFT → DPO), a layered LLM safety gate with streaming output filtering, and a resumable offline evaluation harness with statistically grounded leaderboards. Everything runs offline by default; tests run in CI per package.

| Package | Demonstrates | Headline | Run |
|---|---|---|---|
| [`ragkit`](packages/ragkit) | Chunking, BM25, dense + hybrid retrieval, reranking, query rewriting, retrieval/answer metrics | TBD | `uv run ragkit eval --strategies all` |
| [`minigpt`](packages/minigpt) | Byte-level BPE, GPT architecture, pretraining, GPT-2 weight loading, classification fine-tune, SFT, DPO | TBD | `uv run minigpt pretrain --data tinyshakespeare --steps 2000` |
| [`safegate`](packages/safegate) | Prompt-injection detection, refusal eval, content classification, constitutional rules, streaming output gate | TBD | `uv run safegate redteam` |
| [`evalkit`](packages/evalkit) | Task specs, classical/code/calibration metrics, sandboxed code exec, bootstrap CIs, Elo, resumable runner | TBD | `uv run evalkit run --tasks packages/evalkit/tasks --models oracle,noisy:0.3,random --out runs/demo` |

```bash
uv sync --all-packages
uv run pytest -q
```

Curriculum: AI Engineering from Scratch (MIT). All code here is an independent implementation.
