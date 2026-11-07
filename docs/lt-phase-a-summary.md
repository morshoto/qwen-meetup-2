# LT Phase A: Qwen ARC comparison status

**Phase A comparison complete as of 2026-11-07.** Both Qwen3-14B and Qwen3-32B completed the same frozen 200 tasks with two recorded candidates per task. Qwen3-14B scored pass@1 3/200 and pass@2 3/200, with 328/400 valid candidates (82.0%), 1,264,719 tokens, 13,160.9 seconds summed runtime, and 11.07 GB process-reported peak memory. Qwen3-32B scored 2/200 and 4/200, with 335/400 valid candidates (83.75%), 1,241,566 tokens, 28,066.8 seconds summed runtime, and 22.61 GB peak memory. Neither run had errors or timeouts; swap was unavailable. Exact raw candidate duplicates occurred on 147/200 14B tasks and 56/200 32B tasks, so pass@2 counts attempts and does not establish independent sampling. Full score and frozen manifest are in `artifacts/lt-phase-a-frozen-20261107-01/score-final.json`; protocol and lifecycle details are in [the progress record](lt-phase-a-progress.md).

This is an exploratory protocol separate from the original RSI experiment in issue #2. The Phase A run uses the frozen 200-task set, direct-grid agent, two candidates per task, a 4,096-token limit, and sequential model runs. Its 14B task allocation excludes development and Phase B tasks; the official public evaluation remains reserved for Phase C.

| Measure | Qwen3-14B | Qwen3-32B |
| --- | --- | --- |
| Planned comparison tasks | 200 | 200 |
| Frozen comparison task IDs | 200 | 200 |
| Tasks completed | 200 | 200 |
| Scored checkpoint | 200; complete | 200; complete |
| Task-level pass@1 / pass@2 | 3/200 / 3/200 | 2/200 / 4/200 |
| Valid candidates | 328/400 (82.0%) | 335/400 (83.75%) |
| Tokens, runtime, peak memory and swap | 1,264,719; 13,160.9 s; 11.07 GB; null | 1,241,566; 28,066.8 s; 22.61 GB; null |

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

The 14B model solved one more task at pass@1; 32B had one more task at pass@2
because it solved two additional tasks only when considering either recorded
candidate. Two pass@1 tasks were shared. This exploratory comparison does not
establish an official benchmark score or training effect.

The local training dataset has 1,000 task IDs. A bounded historical inventory
identifies 317 as known exposed, leaving 683 outside that known union. Project
history and model-pretraining exposure remain uncertain; the 683 cannot be
called unseen. Three development-only task IDs have been proposed from known prior exposures;
they are not an execution-ready registration. After also excluding the original
RSI protocol’s 250 reserved IDs, 433 training IDs remain before Phase B
allocation; these are not proven unexposed. A 100-task Phase B pool and a disjoint 200-task Phase A comparison set are
registered in the frozen Phase A configuration; the Phase B recipe is not
frozen. The official 120-task
public evaluation is reserved for Phase C and was not used here.

The latest three-task development run (`lt-dev-20261107-direct-v3-off-01`)
produced 6/6 valid direct-grid candidates but 0/3 pass@1 and pass@2. The
zero-repair code-agent run (`lt-dev-20261106-code-off-01`) and the subsequent
one-repair run (`lt-dev-20261107-code-repair-14b-01`) each produced 0/6 valid
predictions and 0/3 pass@1 and pass@2. The zero-repair candidates ran but
returned incorrect grids for the training examples; the recorded failures
were not syntax, API, runtime, or formatting failures. Repair feedback did not
make the candidates scoreable. No generated Python was used in the frozen
comparison. All 200 task records per model confirm owned worker cleanup; this
does not certify macOS system services or complete system isolation. Phase B
training was not started, and the official 120 public tasks remain reserved for
Phase C. [Full protocol and lifecycle evidence](lt-phase-a-progress.md).
