# ARC2 inference research, 29 October 2026

Primary sources inspected:

- [NVARC authors' repository](https://github.com/1ytic/NVARC), README and linked paper/implementations.
- [Original November 2025 paper](https://github.com/1ytic/NVARC/blob/main/nvarc_2025.pdf), all 11 pages extracted locally; methodological sections 3.2--3.4 and 4.2--4.4 read in full. Cached PDF/text are under the same ignored reference directory. No layout or figure measurements are inferred from text extraction.
- [Authors' inference notebook](https://www.kaggle.com/code/sorokin/arc2-qwen3-unsloth-flash-lora-batch4-queue), downloaded read-only to `reference/arc2/nvarc-inference/` using the official API. Source is inspected, never executed locally. Its model path matches our pinned public checkpoint.
- [Competition rules and code requirements](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/rules), refreshed with the official API and preserved under `reports/arc2/current-state/`.

The reference notebook uses per-puzzle rank-256 LoRA, all seven projection
families plus embeddings/head, alpha 32, rank stabilization, a full augmented
epoch, learning rate 5e-5, and 16 augmentation repetitions. It resets adapters
between puzzles. Its BF16/Unsloth stack and four-worker execution are materially
different from our T4 FP16 pilot; copying its hyperparameters would require a
new compatibility and memory check. Our negative eight-step result does not
test that procedure.

The authors generate candidates through probability-bounded depth-first search
across geometry/color views. The `getter_kgmon` selector groups identical output
grids and ranks by occurrence count minus mean augmented negative log likelihood.
The `calc_scores` function sums answer-token log probabilities, excluding prompt
tokens. The same eight task-seeded augmented contexts score each candidate;
duplicate grids reuse cached scores. These are model preference scores, not
proof that a rule explains the demonstrations.

The paper confirms the same augmentation contexts are used across candidates
and reports a 17% slowdown for batch-invariant inference, illustrating why a
numerically different implementation must be evaluated at a matched budget.
The authors also report 1--2 point variation across identical notebook reruns.
Their corpus includes synthetic tasks derived from public evaluation-task
descriptions. Public-development measurements therefore need especially careful
contamination caveats; the original paper's score is not a prediction for our
inference-only implementation. We do not download those evaluation solutions or
reproduce their evaluation-data pipeline.

Next experiments should compare additional geometry/color views and bounded
branching against the current two greedy views at a matched runtime. Freeze a
candidate pool before comparing vote-only selection and shared-view likelihood
selection. Track coverage, tokens/second, peak allocated memory, valid completions
and paired accuracy. Isolate generation gains from selection gains. Do not infer
an improvement from training loss, more candidates or a successfully completed
notebook.

The frozen development pool already has only one recoverable output beyond the
current hybrid (282-output oracle versus 281 selected), so complex rescoring of
that exact pool has little headroom. New candidate diversity and reliable long
contexts are the better-supported priorities. Training-data contamination remains
possible; official score evidence is required to establish competition progress.

No evaluation solutions were accessed, no additional-account compute was launched,
and no paid compute or billing change was requested.
