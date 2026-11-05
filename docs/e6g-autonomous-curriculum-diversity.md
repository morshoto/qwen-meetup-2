# E6-G Autonomous Curriculum Diversity

E6-G compares the existing generator and repair loop (`G0_baseline`) with the same loop given a domain-neutral reminder about a duplicate Qwen-authored program (`G1_novelty_feedback`). The reminder is a human-designed experimental intervention. It contains only previously authenticated Qwen code and general instructions to produce a different function; it contains no human-authored ARC transform, hidden grid, expected output, or evaluation result. Each condition receives a separate mutable copy of the same sealed initial pool. New records from one condition are considered for a combined dataset only after both conditions finish.

All Live candidate code runs in the existing Docker Sandbox. The pipeline reuses E6-F's bounded model-authored repair and guarded MLX server. `src/arc/operator_search.py` is absent from the execution import graph. Raw source, canonical AST, normalized shape, and sampled behavior are tracked separately: source/AST equality is a definite duplicate; a matching normalized shape or sampled behavior is a suspicion that is conservatively excluded from the novel count. Finite probes cannot prove semantic equivalence or detect every hardcoded behavior.

## Historical diagnosis and seed pool

The [offline diagnosis](../artifacts/e6g/diagnosis-v2/report.md) read 66 sealed E5/E6/E6-F generator responses without model calls. Four E6-F Collection candidates passed Oracle validation and matched an existing seed AST. The [stage review](../artifacts/e6g/diagnosis-review/report.md) found that three had the same model-generated text as the seed; one had different extracted source but the same canonical AST. It found no extraction-only collapse among those four. Across the 66 responses, 59 repeated previously observed raw text and 60 repeated a previously observed AST; these are descriptive counts across runs, not causal evidence about model settings.

The fixed-seed Docker behavior audit recorded 30 probe executions per diagnosis pass, including successes and execution failures. A second pass produced an identical candidate CSV and pinned the current diagnostic dependencies. A separate [HTTP-body audit](../artifacts/e6g/provenance-audit-v2/report.md) verified all 66 saved response bodies against their generated text and both seed Oracles against their code, without model calls or Sandbox starts.

The [stage audit](../artifacts/e6g/stage-audit/report.md) joins each response hash to the recorded generation/repair, static, Sandbox, and registration stages. In the historical records, three Sandbox stages lack enough evidence for classification; they remain `unknown_insufficient_evidence`. A candidate that passed Sandbox and was rejected as an Oracle duplicate is recorded separately from an unexecuted repeated repair.

The two prior Oracles were independently re-audited against their sealed Qwen request/body, model revision, code and record hashes, saved labels, Docker quality checks, and sampled behavior. Both passed and are registered as `historical_verified`; the fixed [seed-pool manifest](../artifacts/e6g/seed-pool-v2/pool_manifest.json) records their individual provenance and flags. Their original E6/E6-F identities did not separately pin `src/arc/training.py`, so the historical static-filter identity is `legacy_identity_incomplete`. The new re-audit pins that file; it does not retroactively prove the historical filter hash. The two source records qualify under the current re-audit, yielding **2 eligible Oracles out of the unchanged gate of 5**.

## Pre-registration and budgets

The Live [experiment protocol](../artifacts/e6g/live-smoke/experiment_protocol.json) and its hash were written before the first model call. It records both symmetric condition caps, the fixed pool hash, candidate/repair seed rule, execution order, prompt and feedback-template hashes, tokenizer and model revision, Docker image ID, data/split/exclusion hashes, dependency hashes including `src/arc/training.py`, and private snapshots of the Git diffs. The seed chose `G1_novelty_feedback` first, then `G0_baseline`. Both arms requested the same seed for corresponding attempts. The guarded driver source applies the requested seed, while the request log labels seed support `unverified`; deterministic reproduction is not asserted.

| Profile | Candidates per arm | Calls per arm / total | Reserved tokens per arm / total | Sandbox starts per arm / total | Wall seconds per arm / total |
| --- | ---: | --- | --- | --- | --- |
| `configs/e6g.fake.yml` | 3 | 9 / 18 fixture calls | 40,000 / 80,000 fixture reservations | 60 / 120 fixture calls | 300 / 600 |
| `configs/e6g.live.smoke.yml` | 2 | 8 / 16 | 25,000 / 50,000 | 20 / 40 | 450 / 900 |
| `configs/e6g.live.compare.yml` | 10 | 30 / 60 | 140,000 / 280,000 | 100 / 200 | 1,200 / 2,400 |

The full comparison was configuration-checked and exercised with Fake only. Its Live profile requires both `--live` and `--approve-full-comparison`, and was not run. The one authorized Live smoke is already consumed. Fixed Live run directories prevent budget bypass via a fresh run. Reservation, observed usage, and the guarded server's own reservation ledger are recorded separately; the server's contract is self-reported and is not remote attestation.

## Fake comparison

The [CPU Fake E2E](../artifacts/e6g/cpu-fake-e2e-v2/comparison.json) uses scripted fixtures designed to exercise feedback; it is interface evidence, not a model result.

| Condition | Completed | Valid | New fixture Oracles | Raw / AST duplicates vs pool | Behavior suspects / evaluated | Repair success | Calls | Reserved tokens | Sandbox fixture calls |
| --- | ---: | ---: | ---: | --- | --- | --- | ---: | ---: | ---: |
| G0 | 3/3 | 3 | 0 | 3/3 / 3/3 | 3/3 | not attempted | 3 | 6,570 | 60 |
| G1 | 3/3 | 3 | 2 | 1/3 / 1/3 | 1/3 | not attempted | 3 | 7,881 | 60 |

Tests also cover zero novel records, five fixture Oracles with no training, two arms generating the same novel records without cross-arm leakage, full-profile caps under Fake, invalid/syntax/runtime responses, repair limits, Sandbox timeout, budget exhaustion, dependency changes, interrupted trials, uncertain requests, and completed-run seal/resume.

## Live Smoke result

The [sealed comparison](../artifacts/e6g/live-smoke/comparison.json) used Qwen3-8B-4bit revision `545dc4251c05440727734bcd94334791f6ab0192`, 4-bit quantization, temperature 0.2, thinking off, no adapter, the same two re-audited seed Oracles, and the same declared condition caps.

| Condition | Completed / limit | Valid | Verified novel | Raw / AST duplicates vs pool | Behavior suspects / evaluated | Repair success | Execution-error candidates | Calls | Reserved / observed tokens | Sandbox | Wall | Stop |
| --- | --- | ---: | ---: | --- | --- | --- | ---: | ---: | --- | ---: | ---: | --- |
| G0 | 2/2 | 0 | 0 | 0/2 / 0/2 | not measured, 0 evaluated | 0/2 | 2 | 6 | 13,786 / 1,304 | 13 | 12.73 s | completed |
| G1 | 1/2 | 1 | 0 | 1/1 / 1/1 | 1/1 | not attempted | 0 | 1 | 2,681 / 263 | 20 | 8.22 s | condition Sandbox cap |

G1's completed candidate was exactly the previously generated seed program. Its 20 Sandbox starts exhausted that arm's cap before the second candidate; G0 had seven unused Sandbox starts, which were not transferred across arms. G0's pool-relative duplicate rate is zero because neither invalid candidate matched the verified pool; independently, two of its six model emissions repeated source previously emitted within G0. Both arms produced zero **verified novel** Oracles. The registered decision is `no_demonstrated_gain`; the unequal number of completed candidates and tiny sample make the relative effect inconclusive.

For Live, both arms consequently recorded zero novel Oracles per 100,000 client-reserved tokens and per 100,000 observed tokens. In Fake, G0 recorded zero and G1 recorded 25.38 novel fixture Oracles per 100,000 reserved fixture tokens; observed-token efficiency is unavailable for the Fake backend. These figures include feedback and repair costs and do not imply real-model improvement.

Together the Live smoke used 7 model calls, 16,467 client-reserved tokens, 1,567 observed tokens, 33 Docker starts, and 20.96 seconds. The two guarded server ledgers reserved 4,084 and 744 tokens respectively under their separate 25,000-token caps. These reservations are not observed token usage. Outside the Live smoke, the two offline diagnosis passes used 60 Docker starts and the two seed re-audits used 70, all without Qwen calls. Docker integration tests are outside those experiment counts.

No new Oracle entered the eligible union: `eligible_unique_oracles=2`, `eligible_for_future_training=false`, and the result is `insufficient_verified_oracles`. No training, weight update, adapter promotion, Public Evaluation, or independent self-solve success occurred. The [independent audit](../artifacts/e6g/audit-v5/report.md) checks both condition and root seals, the pre-registration, all seven hashed HTTP response bodies, reconstruction of both prompt boundaries, requested seeds, SFT provenance, split disjointness, budgets, and owner-only permissions.

### Post-run metric correction

The sealed smoke's original `sampled_behavior_suspect_rate` used all completed candidates as the denominator, including candidates for which no behavior fingerprint was obtained. Therefore its G0 value `0.0` must not be read as a measured behavior duplicate rate. The independent audit reports G0 **not measured** (0 fingerprints) and G1 **1/1 suspect**. The current implementation uses only fingerprinted candidates as that denominator, with a regression test. The original Live artifact and protocol remain unchanged; the [archived prior pipeline](../artifacts/e6g/implementation-snapshot-before-metric-fix/manifest.json) matches its protocol hash. Since the current pipeline has changed, the historical Live `--resume` now fails closed on implementation identity. Before the correction, completed Live resume was verified without new budget use; the corrected Fake run also resumes successfully.

## Commands and verification

```sh
uv run python -m src.cli.e6g --config configs/e6g.fake.yml --check
uv run python -m src.cli.e6g --config configs/e6g.live.smoke.yml --check
uv run python -m src.cli.e6g --config configs/e6g.live.compare.yml --check
uv run python -m src.cli.e6g --config configs/e6g.fake.yml
uv run python -m unittest discover -s tests -p 'test*.py'
```

The executed smoke command was `uv run python -m src.cli.e6g --config configs/e6g.live.smoke.yml --live`. The fixed run directory prevents a second smoke or budget bypass. A future full comparison requires separate explicit authorization and the `--live --approve-full-comparison` flags; neither is implied by this result. At present, stop without additional Live budget. The two retained SFT records are inspectable in the private `artifacts/e6g/live-smoke/sft_generator_distillation.jsonl` and do not meet the five-Oracle training gate.

The final repository suite passed **163 tests** with no skips. All three E6-G configuration checks, Python compilation, `git diff --check`, and mypy 2.4.1 on the 12 E6-G source files passed. The mypy invocation followed imports silently and ignored missing third-party stubs; it is not a whole-repository type audit. The corrected Fake run's resume passed, the changed-code Live resume was refused without budget use, and the historical and new artifact seals were verified.
