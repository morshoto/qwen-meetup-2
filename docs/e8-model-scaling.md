# E8 Qwen3 model scaling

E8 compares independent G0 generation slots across Qwen3-8B, 14B and 32B at 4 bits. It reuses the E7 candidate loop, E6-F verifier, model-authored repair, novelty rules, guarded MLX server and Docker Sandbox. Human-authored ARC transforms and `operator_search.py` are excluded. E8 never starts training, adapter updates or promotion. The two historical Oracles are only an initial duplicate-detection pool; their missing historical dependency hash keeps them out of strict training-eligible counts.

The shared protocol fixes system/user prompts, requested seeds, the initial pool, temperature 0.2, thinking off, max output 512, at most two repairs and the same Docker verifier. Identical seed requests do not imply deterministic outputs. Model-specific tokenizer templates and files are hashed individually; engine and model numerical differences remain. Each model gets an independent mutable pool and separate sealed run. Two canaries are charged to the model budget. Per model Smoke limits are 8 calls, 60,000 client-reserved tokens, 60 Sandbox starts and 900 seconds, with at most 30 Sandbox starts per candidate. This allows two 20-start passing candidates plus up to 20 additional repair starts without weakening validation. A 32B Live stage requires a separate `--allow-32b` authorization. Full also requires explicit approval and exact matching budget values; neither is implied by Smoke.

## Current Live evidence

The [campaign protocol](../artifacts/e8/live-smoke-20261101/protocol.lock.yml) was locked before the first model call. Its SHA-256 is `7b09ea350860466419c8ef1910686cf86d10338e1ab4ca739a758ed4aaa2e71b`. The [8B run](../artifacts/e8/live-smoke-20261101/8b/result.json) completed 2/2 slots. It generated no strict novel Oracle: both candidates failed runtime validation after bounded repairs. It used 8 model calls including two successful canaries, 17,282 reserved / 1,425 observed tokens, 13 Sandbox starts and approximately 35.7 seconds. No OOM was observed. Eight periodic samples recorded a peak sampled server RSS of 4,805,072 KiB; this is not an actual peak or a GPU-only measurement. Swap was 786.44 MiB before and after, and sampled minimum system free memory percentage was 77%. The [independent 8B audit](../artifacts/e8/audit-8b-live-20261101-v3/summary.json) verified all six generation response bodies and prompts, two canary HTTP bodies, request seeds, Sandbox traces, ledgers and model file hashes.

Fake runs use scripted fixture responses, not real model evidence. The [sealed Fake comparison](../artifacts/e8/report-fake-36a1821d53f0/report.md) ran all three model identities through two slots and verifies pipeline behavior only.

## Reproducible commands

```sh
uv run python -m src.cli.e8 --config configs/e8.fake.yml --check
uv run python -m src.cli.e8 --config configs/e8.fake.yml
uv run python -m src.cli.e8 --config configs/e8.smoke.yml --live --stage 8b --campaign-dir artifacts/e8/live-smoke-20261101
uv run python -m src.cli.e8 --config configs/e8.smoke.yml --live --stage 14b --campaign-dir artifacts/e8/live-smoke-20261101
```

Interrupted in-flight inference is recorded as incomplete and is not replayed; a completed stage resumes by verifying its seal without another call. Data and model artifacts under a stage are owner-only. The resource watchdog checks available disk, sampled system free memory, swap increase and RSS; it can terminate only the guarded server whose arguments contain the stage path. It stops for free memory below 10%, swap growth above 4 GiB, or insufficient remaining disk. The specified limits are not claims that all pressure or OOM events can be detected.

### 14B and 32B status

The initial 14B model acquisition reached the registered 600-second download timeout. The [sealed 14B record](../artifacts/e8/live-smoke-20261101/14b/result.json) reports 0/2 slots started, zero model calls and zero Sandbox starts. A weight file had completed, but the full pinned snapshot was not available. This is a download-time limit, not evidence of 14B model failure or a Mac OOM. System free memory was 85% before and 84% after; reported swap remained 786.44 MiB. The partial model cache has been retained.

At that point, 32B had not been downloaded or loaded. Its 18.45 GB repository size is a metadata estimate; safe inference and peak memory remain unmeasured. The prerequisite 14B stage had not completed, and 32B also needs separate user authorization. Accordingly no 8B/14B/32B performance ranking or hardware conclusion was available. The [sealed comparison](../artifacts/e8/report-live-20261101-v2/report.md) marks unmeasured values with a dash. The [independent audit](../artifacts/e8/audit-live-20261101-v3/summary.json) checks the pool against the sealed source, verifier identity, both stage seals and recomputed slot totals, and confirms that the initial 14B attempt used no inference or Sandbox budget.

A possible next step is one separately authorized 14B acquisition retry, capped at 1,200 seconds. The proposed [retry preparation CLI](../src/cli/e8_retry_prepare.py) does not run inference, validates the original seals and protocol, and creates a new campaign containing an unchanged copy of the completed 8B evidence only after a successful download. A subsequent 14B stage would use the original per-model Smoke limits. The partial cache may be reusable, but that is not guaranteed. This retry must not be executed without explicit authorization because the original download time budget is spent. Any 32B execution remains a separate decision.

The final unittest run passed **199 tests** without reported skips or failures. `git diff --check`, compilation and all three configuration checks passed. The retry preparation CLI has a separate test for its explicit-approval guard and a fixture-only test of copying the sealed 8B evidence; neither test downloads a model or starts inference.

### Authorized 14B retry

Following explicit user authorization, one additional acquisition with a 1,200-second limit completed in 573.49 seconds. Its [sealed acquisition record](../artifacts/e8/14b-retry-20261101/result.json) records the pinned revision and hashes. The [new campaign](../artifacts/e8/live-smoke-retry-20261101/protocol.lock.yml) retains the exact common protocol and an unchanged sealed copy of the 8B stage.

The 14B stage completed 2/2 slots, with zero valid or strict novel Oracles. Both candidates were rejected by the existing static safety filter, so no Sandbox was started. The stage used four calls including two successful canaries, 7,876 reserved and 550 observed tokens, and 47.28 seconds including hashing and startup. Its sampled peak server RSS was 8,414,928 KiB and minimum sampled free memory was 71%; swap remained 786.44 MiB before and after. Request-level generation throughput was about 23.3 completion tokens/s, including prefill. No OOM was recorded. Both models used the same chat-template hash; their tokenizer and model files remain individually pinned.

The [updated comparison](../artifacts/e8/report-live-retry-20261101-v1/report.md) and [independent audit](../artifacts/e8/audit-live-retry-20261101-v1/summary.json) cover both completed stages. Each produced zero strict new Oracles; two slots per model cannot establish relative model capability. At that point, the 32B stage awaited its separately required permission.

The separate [emission repeat audit](../artifacts/e8/repeats-live-retry-20261101-v1/summary.json) includes rejected programs: 8B repeated source and AST in 2 of 6 emissions; 14B repeated source and AST in 1 of 2 emissions. These within-model counts differ from the comparison table's duplicate counts against each model's verified pool. Both 14B emissions were rejected by the static safety filter, so their behavior was not measured.

The sealed [offline static analysis](../artifacts/e8/static-analysis-14b-20261101-v1/analysis.json) identified the rejection predicate: both candidates have an empty-list return that matches the existing fixed-grid check because `all()` over zero rows is true. This explains their rejection, but does not establish that either candidate would pass execution, determinism, novelty or provenance checks. The recorded comparison uses the original locked filter; no candidate was executed and no acceptance rule was changed in this analysis.

### Authorized 32B attempt

After separate user authorization, the pinned 32B download reached its 600-second limit before the snapshot completed. The [sealed stage record](../artifacts/e8/live-smoke-retry-20261101/32b/result.json) reports 0/2 slots, zero calls and zero Sandbox starts. Free memory was 83% before and after and swap remained 786.44 MiB; no model was loaded, so load time, inference memory, throughput and model capability remain unmeasured. This is an acquisition timeout, not proof of a hardware limit. The [three-stage audit](../artifacts/e8/audit-live-retry-20261101-v2/summary.json) verified the seals and recorded counts; the [comparison](../artifacts/e8/report-live-retry-20261101-v2/report.md) includes the incomplete stage.

The retry preparation CLI supports `--stage 32b` and preserves verified copies of both completed stages if a separately authorized, 1,200-second acquisition succeeds. It does not start inference. A retry needs explicit additional authorization because the registered download limit has been spent; a later 32B Smoke would still use the unchanged protocol and per-stage budget. A fixture-only test checks the copy and approval behavior without downloading or loading a model.

### Completed 32B Smoke

After the user authorized changing time constraints, the additional 1,200-second [acquisition](../artifacts/e8/32b-retry-20261101/result.json) timed out. A further acquisition capped at 3,600 seconds completed in 1,884.24 seconds including file hashing and copying the prior stages; its [sealed record](../artifacts/e8/32b-retry-3600-20261101/result.json) contains the 32B file hashes. Acquisition time limits changed; the locked generation and validation protocol did not.

The [32B stage](../artifacts/e8/live-smoke-32b-retry-3600-20261101/32b/result.json) completed 2/2 slots: two validated candidates, one strict new independent Oracle and one behavior match against its verified pool. It used four calls including two canaries, 7,876 reserved / 702 observed tokens, 40 Sandbox starts, and 140.01 seconds total. Model load took 5.77 seconds; request-level generation throughput was 11.20 completion tokens/s including prefill. Peak sampled server RSS was 16,036,112 KiB, sampled minimum free memory was 46%, swap stayed at 786.44 MiB and no OOM was recorded. These are samples, not guaranteed peaks.

The final [comparison](../artifacts/e8/report-live-32b-20261101/report.md) and [independent audit](../artifacts/e8/audit-live-32b-20261101/summary.json) cover all three completed stages. Strict novel counts were 0, 0 and 1 for 8B, 14B and 32B respectively. Two slots cannot establish a model ranking. All models remain below the five-Oracle threshold; no training or promotion was performed. Full comparison requires its own explicit approval.
