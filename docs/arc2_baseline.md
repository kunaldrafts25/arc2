# ARC-AGI-2 starting track

Authorized separately by the user on 29 October 2026 IST. This track does not
replace the active ARC-AGI-3 objective or share its score records.

## Verified competition state

Original account kunalsinghh25 is entered in ARC-AGI-2. The official format is
`submission.json`, with two named attempts for every test output of every input
task. Scoring averages exact matches over test outputs, not over tasks. Current
leader observed at setup: Tufa Labs, 88.06. The final deadline is 2 November
23:59 UTC / 3 November 05:29 IST. Runtime limit is 12 hours, offline execution.
See the [official competition](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/).

## Starting floor and official result

`arc2/symbolic.py` searches a bounded, task-local family of geometry, color-map,
crop, object, scale, tile and split-grid transformations. Candidates must match
the task's demonstration outputs; task IDs and test target labels are not used.
Distinct predictions are retained, with an explicit copy-input fallback.

On 1,000 public training tasks, the frozen predictions matched 49/1,076 test
outputs (4.55%) in 15.67 seconds. This is a training-data development measurement,
not unseen generalization. Hosted notebook
`kunalsinghh25/arc-hydra-arc2-symbolic-baseline/1` completed and generated valid
outputs for all 240 supplied tasks / 259 test outputs. IDs and output counts
matched the sample submission. Official submission **57747726 completed at
0.00**, with no reported error. No per-task official diagnostics are available.
This solver remains a reproducible floor and fallback, not a competitive choice.

## Neural path

The public Apache-2.0 checkpoint is
`sorokin/qwen3_4b_grids15_sft139/Transformers/bfloat16/1`, about 7.27 GB on disk.
Its 16-token grid vocabulary is different from the general-language ARC-AGI-3
model; it is not a drop-in replacement for the game-playing Python agent.

`arc2/neural.py` uses standard Transformers FP16 inference, exact grid/chat
serialization, identity/transpose views and a generation grammar for rectangular
digit grids. It rejects unfinished/invalid output and records omissions, timeouts
and memory errors. This is a new inference-only baseline, not reproduction of
the full published NVARC method or its reported scores. Public notebook
`jingshuifu/arc-agi-2-solution-v2` was inspected as a reference; its latest code
contains additional induction changes, so its earlier reported 28.89 must not
be attributed to our implementation.

`kunalsinghh03/arc-hydra-arc2-neural-development/1` is COMPLETE on verified T4
hardware. Independent rescoring reproduces neural-with-fallback 258/1,076,
symbolic 49/1,076 and hybrid 281/1,076 (26.1152%). Runtime was 42,602 seconds.
There were 1,841 valid generations, 41 memory failures, 32 unfinished generations
and 238 budget skips; 154 outputs had no valid neural generation. Predictions
were frozen before public training labels were read. The pretrained checkpoint
may have seen these tasks; these scores are not held-out or official results.

## Task-local learning and matched comparison

`arc2/adaptation.py` adds rank-16 LoRA updates to 33,030,144 parameters; base
weights stay frozen. Training uses only supplied demonstration pairs, masking
input/header tokens from the loss. Geometry and color augmentations preserve
input/output correspondence. Adapters reset between puzzles.

Actual GPU probe version 1 failed before training because stock TorchAO 0.10.0
was incompatible with PEFT's minimum 0.16.0. Version 2 removed that unused
optional quantization backend; this path uses plain FP16 LoRA. Both tasks then
completed all eight requested steps, changed adapter weights, reset exactly and
produced valid grids. Peak allocated GPU memory was 7,845,596,160 bytes (7.31
GiB). Lower training losses and passing these checks do not establish improved
prediction accuracy.

`kunalsinghh03/arc-hydra-arc2-adaptation-comparison/1` is COMPLETE: the same
model/GPU, grammar and views scored 19/51 before and 17/51 after adaptation.
All 400 requested training steps completed, with zero improved and two regressed
outputs. Do not promote this eight-step setting. This pilot uses a different
subset from the full benchmark; percentages cannot be compared directly.

## Original-account inference candidate, 29 October

The official API refresh at 08:51 UTC confirmed Tufa Labs still leads at 88.06,
the deadline remains 2 November 23:59 UTC, and one ARC2 submission is available
today. Original-account free GPU time was 1,192.259 seconds, refreshing 31 October
00:00 UTC; paid scaling was disabled. Raw pages and state are preserved under
`reports/arc2/current-state/`. Rules confirm offline notebook execution up to
12 hours, public external models allowed, one submission/day, two final choices.

`kunalsinghh25/arc-hydra-arc2-inference-hybrid/1` was launched separately with a
560-second visible preview and a 42,600-second official rerun budget. It reads
the actual mounted competition test challenges and writes a complete validated
symbolic fallback before neural work. The preview uses four actual-test tasks;
the official rerun processes every supplied task. No training-development
prediction files or solution files are packaged. No new jobs were launched on
the development account.

`arc2/bounded_inference.py` keeps the measured hybrid precedence rule, schedules
the first view over every output before second views, checkpoints every output,
and uses 256-token causal prefill chunks. A small fixed parity gate falls back
to the old decoder if the two generated grids disagree. Chunking is motivated
by the old OOM range of 4,802--6,526 prompt tokens; no old valid generation had
more than 4,566 prompt tokens. The 32 unfinished generations all reached the
120-second deadline; the 960-token cap exceeds the 930 tokens needed for a
maximum-size valid grid and EOS. GPU diagnostics and official results must be
inspected before claiming either memory or accuracy improvement.

Version 1 completed in 341.72 seconds, produced eight valid generations on its
four preview tasks, and passed full 240-task/259-output format validation. Both
fixed small-prompt parity checks matched the baseline. The 6,526-token diagnostic
did not exhaust memory (8,766,268,416 peak allocated bytes), but its 150-second
diagnostic time cap stopped it after 740 tokens.

After a fresh quota check showed 830.395 free seconds, a separate version 2
preview was launched with a 500-second budget. Its scheduler allocates time
using prompt length and demonstration-output size, with a 300-second per-view
cap. It preserves the same symbolic/neural selection rule and first-view-first
ordering. The long diagnostic completed all 930 tokens in 189.91 seconds with
8,806,957,568 peak allocated bytes. This demonstrates functional completion and
memory improvement on that formerly failing prompt, not prediction accuracy.
Both small-prompt parity checks passed again. Fifteen ARC2 local tests pass.
Frozen source packages and launch journals are retained separately for both
versions. Final preview validation and official evaluation state are recorded
under `reports/kaggle/arc2-inference-hybrid-v2/` when available.

Version 2 completed in 404.56 seconds with eight valid actual-test preview
generations and full 240-task/259-output format coverage. The exact version was
submitted as **57765962** on 29 October at 09:10 UTC. At 09:11 UTC its official
status was **PENDING**, with no score or error. The daily allowance is now zero;
free GPU time remaining is 402.27 seconds, with paid scaling disabled. Best
completed ARC2 official score remains 0.00 until a new result establishes more.
A hidden read-only observer (`scripts/watch_arc2_official.py`, process 12240 at
startup) writes `official-status.json` and `official-history.jsonl` under the
version 2 report directory. It never launches or resubmits anything. If its
process stops, reconcile this same submission reference instead of retrying
the submission mutation.

Frozen-candidate analysis (`scripts/analyze_arc2_candidates.py`) found the union
of all stored neural/symbolic guesses covers only 282/1,076 training outputs,
versus 281 for the hybrid. A demonstration-only shape rule recovers that single
extra output without a regression in this development set. This diagnostic
suggests candidate generation is the larger bottleneck in the frozen pool.
The alternative selector is not in version 1 and has no held-out validation.

## Boundaries and resources

Automatic approval review rejected downloading `arc-agi_evaluation_solutions.json`
as potentially sensitive hidden-answer data that was not specifically authorized.
That download remains blocked. It was not retried through another endpoint.
Only explicitly permitted public training challenges/solutions and the sample
submission were downloaded. Official evaluations provide separate score evidence.

The user-designated compute account hosted private development notebooks using
public training inputs and publicly licensed weights. It makes no competition
entries or submissions. Account-rule compliance is not certified by technical
access. Both neural development runs completed; neither was restarted. No new
compute was launched on that account in this continuation. Paid scaling is
disabled. Original-account submissions remain separate.

The separate ARC-AGI-3 control/inspection versions both terminated before
inference: the requested RTX configuration had no NVIDIA driver. A standard
T4 probe successfully returned two real T4s. This diagnoses available hardware
but does not explain the exact backend mapping issue or make the large ARC-3
configuration compatible with T4 unchanged. No gameplay score is assigned to
those failed runs. ARC-3's best official score remains 28.45.
