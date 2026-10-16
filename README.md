# AI Engineering Capstones

Four from-scratch AI engineering systems, each in its own repository, plus one research project built on two of them. This repository pins them together as git submodules.

| Repo | Demonstrates | Headline |
|---|---|---|
| [`ragkit`](https://github.com/d41sys/ragkit) | Chunking, BM25, dense + hybrid retrieval, reranking, query rewriting, retrieval/answer metrics | Hybrid RRF: recall@5 1.00, MRR 0.944, nDCG@10 0.958 (40 questions, 25 docs) |
| [`minigpt`](https://github.com/d41sys/minigpt) | Byte-level BPE, GPT architecture, pretraining, GPT-2 weight loading, classification fine-tune, SFT, DPO | Tiny-Shakespeare val loss 6.99 → 3.65 (−48%); SMS spam test acc 0.940; GPT-2 124M greedy parity with HF |
| [`safegate`](https://github.com/d41sys/safegate) | Prompt-injection detection, refusal eval, content classification, constitutional rules, streaming output gate | Held-out detection 39/39, benign FPR 0/42 (small self-authored set) |
| [`evalkit`](https://github.com/d41sys/evalkit) | Task specs, classical/code/calibration metrics, sandboxed code exec, bootstrap CIs, Elo, resumable runner | Macro score oracle 1.000 > noisy:0.3 0.714 > random 0.179 over 345 example-model pairs |
| `flexgridcyber` (private, research) | Prompt injection at the formalization step of LLM energy-flexibility agents | Feasibility pilot on Qwen2.5 3B/7B |

```bash
git clone --recurse-submodules https://github.com/d41sys/ai-engineering-capstones.git
cd ai-engineering-capstones/ragkit && uv sync && uv run pytest -q
```

Each repo has its own CI, README and quickstart. Reading order: [`docs/reading-guide.md`](docs/reading-guide.md). Improvement plan: [`docs/improvement-plan.md`](docs/improvement-plan.md).

Curriculum: AI Engineering from Scratch (MIT). All code here is an independent implementation.
