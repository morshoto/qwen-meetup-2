# LT Phase A: Qwen ARC comparison status

**Status as of 2026-11-06: incomplete. No ARC comparison was run.** This is
an exploratory protocol separate from the original RSI experiment in issue #2.

| Measure | Qwen3-14B | Qwen3-32B |
| --- | --- | --- |
| Planned comparison tasks | 200 | 200 |
| Frozen comparison task IDs | Not selected | Not selected |
| Real comparison tasks attempted | 0 | 0 |
| Task-level pass@1 / pass@2 | Not measured | Not measured |
| Test-output accuracy | Not measured | Not measured |
| Tokens, runtime, peak memory and swap | Not measured | Not measured |

Zero attempts is not a zero-percent accuracy result. No baseline or model
ranking can be inferred, and no ARC improvement, training success or official
benchmark score has been established by this Phase A work.

The local training dataset has 1,000 task IDs. A bounded historical inventory
identifies 317 as known exposed, leaving 683 outside that known union. Project
history and model-pretraining exposure remain uncertain; the 683 cannot be
called unseen. Three development-only task IDs have been proposed from known prior exposures;
they are not an execution-ready registration. Phase B training tasks and the
200 comparison tasks have not been selected or frozen. The official 120-task
public evaluation is reserved for Phase C and was not used here.

The existing agent, guarded MLX transport and bounded Docker sandbox are
available as components. Offline tests cover task isolation, scoring, frozen
settings and fixture journaling. A local synthetic container probe returned and
timed out as expected, with neither probed container listed afterward. These
are not real-model ARC results or independent proof of containment and cleanup.
The live runner, loaded-model identity, evaluator isolation and owned-process
cleanup remain unverified, so development inference and the comparison have
not started. [Details and current evidence](lt-phase-a-progress.md).

Next, establish the local execution and cleanup boundary, demonstrate the
existing agent on development-only data, and freeze the task and model/run
identities before attempting the two models sequentially. If Phase B reuses
the 200 tasks, its training design must be frozen before those results are
inspected; otherwise the reuse must be described as exploratory.
