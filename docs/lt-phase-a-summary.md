# LT Phase A: Qwen ARC comparison status

**Status as of 2026-11-06: the 200-task comparison is incomplete.** This is
an exploratory protocol separate from the original RSI experiment in issue #2.

| Measure | Qwen3-14B | Qwen3-32B |
| --- | --- | --- |
| Planned comparison tasks | 200 | 200 |
| Frozen comparison task IDs | Not selected | Not selected |
| Real comparison tasks attempted | 0 | 0 |
| Task-level pass@1 / pass@2 | Not measured | Not measured |
| Test-output accuracy | Not measured | Not measured |
| Tokens, runtime, peak memory and swap | Not measured | Not measured |

A separate real 14B direct-grid development run (`lt-dev-20261106-direct-01`)
at revision `a4d9b2df59d2c150bef02fcbe0d91046b7ca33a4` attempted three
known-exposed tasks with two candidates each. The trusted scorer recorded
0/3 task-level pass@1 and 0/3 pass@2: `84db8fc4`, `a64e4611`, and `e5c44e8f`
all failed because neither candidate conformed to the agent output schema.
Four responses used output objects rather than grids; two were truncated at
the 2,048-token limit. This is a development format result, not the frozen
comparison. The run recorded 37,695 total prompt and generated tokens, 367.8
seconds across tasks, and 11.12 GB process-reported peak memory; swap was not
measured. The launched model process exited with code zero and its recorded
PIDs were absent afterward; external Metal service cleanup was not verified.
No generated Python or Docker containers were used. Local raw responses and
scores are in `artifacts/lt-dev-20261106-direct-01/`; `run.json` records
configuration and SHA-256 hashes of the receipt, raw results and score. The
run record is a local observation, and model and cleanup identities were not
independently verified.

Zero attempts is not a zero-percent accuracy result. No baseline or model
ranking can be inferred, and no ARC improvement, training success or official
benchmark score has been established by this Phase A work.

The local training dataset has 1,000 task IDs. A bounded historical inventory
identifies 317 as known exposed, leaving 683 outside that known union. Project
history and model-pretraining exposure remain uncertain; the 683 cannot be
called unseen. Three development-only task IDs have been proposed from known prior exposures;
they are not an execution-ready registration. After also excluding the original
RSI protocol’s 250 reserved IDs, 433 training IDs remain before Phase B
allocation; these are not proven unexposed. Phase B training tasks and the
200 comparison tasks have not been selected or frozen. The official 120-task
public evaluation is reserved for Phase C and was not used here.

The existing agent, guarded MLX transport and bounded Docker sandbox are
available as components. Offline tests cover task isolation, scoring, frozen
settings and fixture journaling. A local synthetic container probe returned and
timed out as expected, with neither probed container listed afterward. These
are not real-model ARC results or independent proof of containment and cleanup.
The live runner, loaded-model identity, evaluator isolation and owned-process
cleanup remain unverified, so the frozen comparison has not started. [Details and current evidence](lt-phase-a-progress.md).

Next, establish the local execution and cleanup boundary, demonstrate the
existing agent on development-only data, and freeze the task and model/run
identities before attempting the two models sequentially. If Phase B reuses
the 200 tasks, its training design must be frozen before those results are
inspected; otherwise the reuse must be described as exploratory.
