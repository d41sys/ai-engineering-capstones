# AI Engineering Capstones

Four from-scratch AI engineering systems in one Python monorepo: a retrieval-augmented generation stack evaluated on an energy-flexibility corpus, a GPT trained and fine-tuned end to end (BPE → pretraining → classification → SFT → DPO), a layered LLM safety gate with streaming output filtering, and a resumable offline evaluation harness with statistically grounded leaderboards. Everything runs offline by default; tests run in CI per package.

| Package | Demonstrates | Headline | Run |
|---|---|---|---|
| [`ragkit`](packages/ragkit) | Chunking, BM25, dense + hybrid retrieval, reranking, query rewriting, retrieval/answer metrics | Hybrid RRF: recall@5 1.00, MRR 0.944, nDCG@10 0.958 (40 questions, 25 docs) | `uv run ragkit eval --strategies all` |
| [`minigpt`](packages/minigpt) | Byte-level BPE, GPT architecture, pretraining, GPT-2 weight loading, classification fine-tune, SFT, DPO | Tiny-Shakespeare val loss 6.99 → 3.65 (−48%, MPS); SMS spam test acc 0.940; GPT-2 124M greedy parity with HF | `cd packages/minigpt && uv run minigpt pretrain --data tinyshakespeare --steps 2000` |
| [`safegate`](packages/safegate) | Prompt-injection detection, refusal eval, content classification, constitutional rules, streaming output gate | Held-out detection 39/39, benign FPR 0/42 (small self-authored set) | `uv run safegate redteam` |
| [`evalkit`](packages/evalkit) | Task specs, classical/code/calibration metrics, sandboxed code exec, bootstrap CIs, Elo, resumable runner | Macro score oracle 1.000 > noisy:0.3 0.654 > random 0.057 over 345 example-model pairs | `uv run evalkit run --tasks packages/evalkit/tasks --models oracle,noisy:0.3,random --out runs/demo` |

```bash
uv sync --all-packages
uv run pytest -q
```

Curriculum: AI Engineering from Scratch (MIT). All code here is an independent implementation.
