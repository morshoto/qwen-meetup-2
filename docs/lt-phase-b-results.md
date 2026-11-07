# LT Phase B results — not yet run

This is a status record for issue #5's separate exploratory experiment, not a
completed experiment report. The protocol remains [draft](lt-phase-b-protocol.md)
and its machine-readable configuration is marked `execution_ready: false` and
`training_authorized: false`. The strict RSI experiment in issue #2 and its
five-Oracle threshold are separate; no claim is made that those gates were met.

## Allocation and exposure

The preallocated training pool has 100 task IDs. Recomputing its intersection
with the earlier strict-RSI generation schedule finds five shared IDs:
`009d5c81`, `00d62c1b`, `03560426`, `045e512c`, and `0962bcdd`. The issue
requires both the preallocated pool and preservation of earlier reservations,
so the allocation decision is pending. The pool and the proposed separate
50-task comparison set are not frozen. No model should be exposed to these
Phase B tasks until the decision is resolved, the allocation is re-audited,
and the complete run identities and protocol are frozen.

The draft records ID and visible-content exclusions, not proof of semantic
independence or complete historical or pretraining non-exposure. The official
120-task public evaluation is reserved for Phase C; its answers are not part of
this experiment's preparation.

## Current observations

| Step | Status | Evidence and limit |
| --- | --- | --- |
| 14B self-solution collection | Not run | The allocation and run descriptor are not frozen. Attempted task count and eligible yield are unknown, not zero. |
| Verified SFT data | Not produced | There are no real Phase B collection records from which to construct it. Fixture examples test the code path only. |
| Synthetic one-step 14B canary | Not run | The run protocol, source and runtime identities are not frozen. No 14B tensor-change, reload, loss or resource evidence exists for this canary. |
| Real-data 14B LoRA | Not run | A real canary and verified self-solution yield are unavailable. No Gen 1 adapter has been trained. |
| Matched Gen 0/Gen 1 comparison | Not run | There is no trained and reloaded Gen 1 adapter or frozen comparison run. Paired predictions, accuracy, latency, cost and uncertainty are unavailable. |

No local `lt-phase-b-*` run directory was found in the issue worktree or the
main checkout's `artifacts` directory during this inspection. No Phase B worker
or MLX-LM training process appeared in the local process listing. These are
point-in-time observations, not proof of historical non-execution or of
system-owned Metal service cleanup.

## Implementation checks

The draft PR contains unit and fixture coverage for allocation, collection,
verification, private SFT preparation, trainer instrumentation, adapter
reload, sequential paired collection and offline scoring. Optional native
MLX-LM checks have exercised a freshly constructed tiny model and synthetic
fixture tasks, including a 20-step trainer path and a separate adapter reload.
Those checks did not load the pinned 14B weights, collect real ARC predictions,
or establish ARC improvement. Exact test counts and the current PR head are
reported in the PR validation section.

The next experiment report must replace these not-run entries with the actual
run identities, counts, rejection reasons, canary and resource evidence, and,
when the yield and canary permit training, real adapter and paired comparison
results. A changed adapter alone would demonstrate parameter adaptation, not
improved ARC generalization or strict RSI success.
