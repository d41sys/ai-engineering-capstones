# Improvement plan

Two phases per repo. **Phase A** closes the gaps against the course capstone spec (Phase 19 end-to-end lessons). **Phase B** applies the repo to a real energy case with real data or real models. Every item ends with a test or a measured number in the repo README; negative results are reported as measured.

✅ = done (see the repo README for the measured result).

Order: ragkit A → minigpt A → safegate A → evalkit A → Phase B in the same order.

## ragkit

| # | Phase | Item | Acceptance | Estimate |
|---|---|---|---|---|
| R1 ✅ | A | **Refuse on low confidence** (lesson 69): the pipeline abstains when retrieval support is weak instead of answering from a neighbouring doc | `Answer.abstained`; unanswerable questions added to the eval set; README reports answer rate on answerable vs abstain rate on unanswerable | 0.5 day |
| R2 ✅ | A | **Heading leak**: markdown headings never appear in extractive answers | regression test on a chunk that starts with a heading | 1 h |
| R3 ✅ | A | **Staged ablation** (lesson 69 "the composed pipeline beats each stage"): one table, same qrels, each stage added in turn | `ragkit eval --ablation` prints the table; README states where the composed system does *not* win (lexical rerank currently trails BM25) | 0.5 day |
| R4 ✅ | B | Real corpus: public NL/EU grid-congestion documents (ACM/Netbeheer Nederland, ENTSO-E network codes; licences checked), harder QA set incl. multi-hop and unanswerable, drafted by the agent, reviewed by you | qa set with provenance per question; metrics reported with bootstrap CIs | 2 days + 3–5 h your review |

## minigpt

| # | Phase | Item | Acceptance | Estimate |
|---|---|---|---|---|
| M1 ✅ | A | **Lesson-41 eval pipeline**: perplexity + exact-match + token F1 + mock judge, aggregated into one weighted per-task report | `minigpt eval` prints the four metrics and the aggregate; unit tests for EM/F1 normalisation | 0.5 day |
| M2 ✅ | A | **DPO < SFT on ROUGE-L**: β sweep and an explanation, not a hidden number | table over β ∈ {0.05, 0.1, 0.5}; README states the outcome either way | 0.5 day |
| M3 ✅ | B | Day-ahead price forecasting with the same transformer on public ENTSO-E/Octopus data vs seasonal-naive baseline | MAE vs baseline on a held-out month | 2 days |

## safegate

| # | Phase | Item | Acceptance | Estimate |
|---|---|---|---|---|
| S1 ✅ | A | **Course action vocabulary**: block / redact / warn / allow alongside the existing refuse / escalate | aggregation table from lesson 87 reproduced by a test | 2 h |
| S2 ✅ | A | **Per-request trace**: one JSON object per request with every checkpoint's signal, the aggregated action and timings | trace emitted by `safegate check --trace`; schema test | 0.5 day |
| S3 ✅ | B | **External evaluation**: public prompt-injection datasets (licence checked) instead of the self-authored set | detection rate + FPR with CIs on external data; the optimistic 39/39 headline is replaced | 1 day |

## evalkit

| # | Phase | Item | Acceptance | Estimate |
|---|---|---|---|---|
| E1 ✅ | A | Replace `preexec_fn` (unsafe with threads) with a wrapper child process that sets its own limits | timeout + memory tests still pass under the thread pool | 0.5 day |
| E2 ✅ | A | **`evalkit compare runA runB`**: paired bootstrap and sign test on shared example ids | matches hand-computed values on a fixture | 0.5 day |
| E3 ✅ | B | Real local model adapter (HF transformers / MLX), run the bundled tasks on 2–3 small open models | leaderboard with CIs in README; run cost and hardware stated | 1 day |

Total: Phase A ≈ 4 days, Phase B ≈ 7 days plus your QA review.

## Phase C: differentiators (from `docs/gap-analysis.md`)

| Id | Status | Move | Result |
|---|---|---|---|
| C-E | ✅ | evalkit `compare --all-pairs`: randomisation test, Holm, MDE, n for 80% power, tie groups | Qwen 1.5B vs 3B unresolved (diff 0 on 115 paired examples); 0.5B resolved vs all |
| C-M | ✅ | minigpt 12-fold rolling NL price backtest, Diebold-Mariano (HLN) + paired bootstrap | transformer (5-seed) MAE 21.25 vs ridge 22.66, DM p = 0.016; single seed only 0.48 better (untested) |
| C-R | ✅ | ragkit abstention signals on EU set: AUROC, risk-coverage, CV thresholds | no signal separates answerable/unanswerable on EU; `query_support` AUROC 0.642 [0.48, 0.80] vs 0.987 synthetic |
| C-S | ✅ | safegate `spec-check`: deterministic bounds gate on the emitted optimiser spec | evaluated in the private flexgridcyber repo (unpublished) |
