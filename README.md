# AI Engineering Capstones

Four from-scratch AI engineering systems, each in its own repository, plus one research project built on two of them. This repository pins them together as git submodules.

| Repo | Demonstrates | Headline |
|---|---|---|
| [`ragkit`](https://github.com/d41sys/ragkit) | Chunking, BM25, dense + hybrid retrieval, reranking, query rewriting, retrieval/answer metrics | Real EU corpus (94 articles from 6 grid regulations, 42 questions, agent-drafted + second-agent reviewed): BM25 recall@5 0.857 [0.75, 0.95] > dense 0.619; refusal threshold calibrated on synthetic data does not transfer (answers 20/42) |
| [`minigpt`](https://github.com/d41sys/minigpt) | Byte-level BPE, GPT architecture, pretraining, GPT-2 weight loading, classification fine-tune, SFT, DPO | Tiny-Shakespeare val loss 6.99 → 3.65 (−48%); SMS spam test acc 0.940; NL day-ahead prices, 12-fold rolling backtest: 5-seed transformer ensemble MAE 21.25 vs ridge 22.66 €/MWh (DM p = 0.016); each single seed ties with ridge; SFT cuts Alpaca ppl 409 → 39; DPO no gain over SFT |
| [`safegate`](https://github.com/d41sys/safegate) | Prompt-injection detection, refusal eval, content classification, constitutional rules, streaming output gate | Public `deepset/prompt-injections` test: full gate detects 48% [36, 61], blocks 27% [17, 40] of benign prompts (self-authored 39/39 does not hold) |
| [`evalkit`](https://github.com/d41sys/evalkit) | Task specs, classical/code/calibration metrics, sandboxed code exec, bootstrap CIs, Elo, resumable runner | Local Qwen2.5 via `hf:` adapter: 1.5B 0.986, 3B 0.968, 0.5B 0.829 macro; tie on the bundled tasks (115 examples); on 240 GSM8K problems 3B 0.729 > 1.5B 0.617 > 0.5B 0.375, every adjacent pair resolved (1.5B vs 3B Holm p = 0.004) |
| `flexgridcyber` (private, research) | Prompt injection at the formalization step of LLM energy-flexibility agents | Feasibility pilot on Qwen2.5 3B/7B |

```bash
git clone --recurse-submodules https://github.com/d41sys/ai-engineering-capstones.git
cd ai-engineering-capstones/ragkit && uv sync && uv run pytest -q
```

Each repo has its own CI, README and quickstart. Reading order: [`docs/reading-guide.md`](docs/reading-guide.md). Improvement plan: [`docs/improvement-plan.md`](docs/improvement-plan.md). Where these sit against existing tools: [`docs/gap-analysis.md`](docs/gap-analysis.md).

Curriculum: AI Engineering from Scratch (MIT). All code here is an independent implementation.
