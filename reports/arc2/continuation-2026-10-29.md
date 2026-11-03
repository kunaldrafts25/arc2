# ARC2 continuation evidence

Official submission: **57765962**, exact notebook
`kunalsinghh25/arc-hydra-arc2-inference-hybrid/2`. Accepted 29 October 2026,
09:10 UTC / 14:40 IST. At the initial check it was PENDING, with no score.
Do not submit it again. The exclusive mutation journal is
`reports/kaggle/arc2-inference-hybrid-v2/official-submission.json`.

The read-only observer records authoritative status in that directory's
`official-status.json` and appends `official-history.jsonl`. Process 12240 was
started hidden, with stdout/stderr redirected to `observer.log` and
`observer-error.log`. It observes only this reference for up to 18 hours and
exits on COMPLETE or ERROR. It performs no cloud mutations. Inspect the latest
status before planning any subsequent experiment.

Verified changes: 256-token cached prefill; coverage-first view scheduling;
time shares weighted by prompt size and demonstration-output size; 300-second
per-view maximum; durable complete symbolic fallback; unchanged measured hybrid
selection rule; two small-prompt numerical parity gates. Existing baseline
sources and uploaded packages remain unchanged. New version 1 and version 2
packages are frozen separately with SHA-256 manifests.

GPU evidence: both previews passed both parity checks and produced eight valid
actual-test generations. Version 1 completed in 341.72 seconds; version 2 in
404.56 seconds. Every required ID and both attempts were validated against the
competition sample structure: 240 tasks, 259 outputs. The formerly OOMing
6,526-token prompt completed a 930-token grid in version 2, in 189.91 seconds,
with 8,806,957,568 peak allocated bytes. This proves functional/memory progress,
not grid correctness or an official score improvement. Official reruns use all
supplied tasks and at most 42,600 seconds, including model initialization inside
the cell. The visible preview uses four tasks; all others have valid fallback.

Local checks: 15 ARC2 tests pass; changed/new scripts compile; diff whitespace
check passes. Both old development jobs are independently confirmed COMPLETE.
Rescoring and frozen-source checks reproduce all previous metrics. The negative
adaptation setting remains rejected. A fixed demonstration-shape filter gets
282/1,076 on stored public-training candidates versus hybrid 281, exactly the
four-candidate union oracle. It is kept separate from the official candidate.
This is training development with possible checkpoint contamination.

At 09:11 UTC: Tufa Labs still leads at 88.06; final deadline remains 2 November
23:59 UTC / 3 November 05:29 IST. One submission per UTC day, currently zero
remaining. Original-account GPU quota has 402.27 free seconds left until
31 October 00:00 UTC / 05:30 IST; paid scaling disabled. No additional-account
compute, paid resources, hidden/evaluation solutions, branch changes or ARC3
mutations were used. No shared goal state was changed.

Next work after official completion: record the score/error and compare with
the existing official 0.00. Treat training metrics and functional checks as
separate evidence. Prioritize new candidate generation (additional geometry and
color views, probability-bounded search) over complex ranking of the existing
pool, whose oracle offers only one extra training output. The authors' original
paper and implementation were reviewed; their full adaptation procedure differs
substantially from the regressed eight-step pilot. Any new adaptation experiment
needs a matched baseline and fresh free-quota check. Do not launch duplicate jobs
or retry uncertain mutations. Reverify daily allowance before the next submission.
