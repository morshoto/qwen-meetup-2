# E6-F Oracle Reliability

E6-F diagnoses failed Qwen-authored Oracle programs, asks the same Qwen generation to repair eligible failures, and collects verified generator-distillation records. It does not invoke a trainer, update an adapter, or promote a model. The five-distinct-Oracle gate is unchanged. Passing an Oracle is not an independent `self_solved_synthetic` success.

## Method and boundaries

- The offline diagnostic checks the sealed E6 source and Qwen request/response hashes, then reproduces the original candidate on the original deterministic validation inputs only inside the existing resource-limited Docker image. It makes no model calls. Historical exit codes were not retained; a reproduced exception is identified separately from the incomplete historical traceback.
- A candidate receives at most two model-authored repairs. The repair prompt contains the preceding Qwen code, a generic function and grid contract, and an allowlisted error category or exception class. It contains no test grid, hidden output, ARC label, traceback text, or human-written ARC solution. Each repair receives fresh public validation inputs. A failure on the separate private acceptance inputs only rejects the candidate; it cannot trigger repair.
- Eligible syntax, contract, runtime, and output-format failures may be repaired. Unsafe code, repeated code, infrastructure failure, timeouts, and exhausted budgets stop the lineage. All generated code executes only in Docker for Live; Fake fixtures use a dispatcher and are never Live evidence.
- Old records are re-audited against their sealed source, Qwen response, code hashes, stored train/held-out labels, current grid and determinism checks, and non-identity checks. Task ID, AST, normalized program shape, and outputs on five common private inputs are used to reject duplicates. These are sampled checks; they cannot prove all programs semantically distinct or detect every hardcoded behavior.
- The SFT user message is the existing `solve_prompt` inference prompt. The completion is the Qwen-authored Oracle; verifier inputs and outputs remain in private sidecars. Real ARC Training split IDs and the existing diagnostic exclusions are kept separate and never supplied as teacher labels or repair feedback.
- Resume verifies the configuration, model revision, implementation and data hashes, source seals, Docker image, split manifest, journals, request-evidence prefixes, and completed-run seal. The current implementation also pins `src/arc/training.py`, which supplies the static code filter. A started but incomplete candidate is marked interrupted and is not reissued. An uncertain pending model request stops the run. Live profiles have fixed run directories to prevent creating another run to bypass their budgets.

## Results, 2026-11-01

The [offline diagnosis](../artifacts/e6f/diagnosis/report.md) classified all six historical sandbox failures as runtime exceptions. The original six candidate hashes were identical. Reproduction found `IndexError` for each case; the original truncated records did not preserve the final exception or exit code. The diagnosis used 14 Docker starts and no Qwen calls. Both old E6 Oracles passed the current source, provenance, and quality re-audit and remained distinct on the common probes.

| Run | Initial validation | After repair | Unique Oracles | Calls | Client reserved / observed tokens | Docker starts | Wall time | Stop |
| --- | --- | --- | ---: | ---: | --- | ---: | ---: | --- |
| CPU Fake | 4/5 | 5/5 | 5 fixture Oracles | 6 fixture calls | 13,203 / unavailable | 101 fixture calls | 0.07 s | target reached |
| [Live Smoke](../artifacts/e6f/live-smoke/report.md) | 0/2 | 0/2 | 2 re-audited E6 Oracles; 0 new | 6 | 13,786 / 1,304 | 48 | 22.38 s | insufficient verified Oracles |
| [Live Collection](../artifacts/e6f/live-collection/report.md) | 4/19 evaluated | 4/19 evaluated | 2 re-audited E6 Oracles; 0 new | 49 | 111,938 / 10,348 | 200 | sandbox budget exhausted |

The collection started 20 candidates; the last ran out of sandbox budget during initial validation. The table excludes that incomplete candidate from the success-rate denominator. The sealed run metrics conservatively record 4/20 including it. All four fully verified candidates matched an existing Oracle by exact AST, so none increased the unique count. Other candidate outcomes were eight repair limits, six repeated-code stops, one private-acceptance rejection, and one budget stop. No repaired candidate passed validation in either Live run.

The matched collection comparison used 20 initial calls and 112 candidate-validation Docker starts, reserving 43,800 client tokens and observing 3,498. Initial candidates plus repairs used 49 calls and 165 candidate-validation starts, reserving 111,938 and observing 10,348 tokens. The additional 35 Docker starts in the run re-audited the two prior Oracles. Thus the observed-token efficiency was 1.14 initial validation passes per 1,000 tokens versus 0.39 after repair; these cost denominators include the budget-interrupted candidate and the four validation passes were duplicates. No independent self-solve or generalization improvement follows from these figures.

Smoke and Collection together used 55 Qwen calls, 125,724 client-reserved tokens, 11,652 observed tokens, and 248 Docker starts. The guarded server independently reserved 4,084 tokens in Smoke and 33,155 in Collection; server reservations, conservative client reservations, and observed usage have different definitions. Two additional offline source-audit preflights used 40 and 35 Docker starts respectively, without model calls. The test suite's Docker integration starts are not included in those experiment counts.

Both live runs recorded zero training steps, no adapter update, and `ready_for_e6_training: false`. The [artifact audit](../artifacts/e6f/audit-v2/report.md) verified the run, diagnosis, and source seals; SFT provenance and held-out separation; 400/400/200 split disjointness; request and sandbox budgets; and client/server token ledgers. The sealed E5 and E6 source artifacts remained unchanged. Sensitive data is kept under owner-only directories; the completed run's sensitive logs and cache files were additionally set to `0600` without changing sealed contents.

The two Live identities pinned the pipeline and most dependencies but omitted the separate `src/arc/training.py` hash used by the static filter. The prior pipeline snapshot matches the hash recorded in both Live identities; the filter file was captured only after the runs and cannot establish its historical hash. The current pipeline fixes this omission for future runs and sets the sensitive runtime files to `0600`. This limitation does not change the sealed historical results.

The final full suite passed all 143 tests. The three E6-F configuration checks, Python compilation, `git diff --check`, completed-run resume checks, and the CPU Fake tests for both five-oracle readiness and insufficient data also passed. After the identity hardening, `artifacts/e6f/cpu-fake-e2e-v3` reached five fixture Oracles, verified completed-run resume, and recorded the static-filter hash in its identity.

## Reproduction and operation

```sh
uv run python -m src.cli.e6f_diagnose
uv run python -m src.cli.e6f --config configs/e6f.cpu.fake.yml --check
uv run python -m src.cli.e6f --config configs/e6f.live.smoke.yml --check
uv run python -m src.cli.e6f --config configs/e6f.live.collection.yml --check
uv run python -m src.cli.e6f --config configs/e6f.cpu.fake.yml
uv run python -m unittest discover -s tests -p 'test*.py'
```

The diagnosis command refuses to overwrite its existing result. The one authorized Live Smoke and Collection have already been consumed. Their original commands are retained for provenance, not for a fresh retry:

```sh
uv run python -m src.cli.e6f --config configs/e6f.live.smoke.yml --live
uv run python -m src.cli.e6f --config configs/e6f.live.collection.yml --live --allow-collection
```

Before the post-run hardening, `--resume` verified both completed Live runs and returned without inference. The current pipeline has a different implementation hash, so resuming those historical runs now fails closed; their sealed prior pipeline snapshot and the independent audit remain available. New runs resume only with their original implementation identity. No additional collection budget or fresh Live run is authorized by these commands. To reach the existing training gate, future work needs separately authorized collection and a strategy that produces genuinely different, robust Qwen-authored Oracles. The two retained records are inspectable in `artifacts/e6f/live-collection/sft_generator_distillation.jsonl`; they are not sufficient for E6-D training.
