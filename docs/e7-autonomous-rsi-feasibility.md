# E7 autonomous RSI feasibility gate

E7 compares two arms using paired generation slots, an unchanged verifier, separate mutable Oracle pools and equal limits. Each slot receives the same requested seed and model settings. The arms differ by the duplicate feedback supplied to G1. Requested seed equality does not guarantee identical output across runtimes. No human ARC transform or operator search enters the generation path. E7 does not train an adapter or promote a model.

## Historical Sandbox diagnosis

The [sealed offline diagnosis](../artifacts/e7/diagnosis-3b9320efa8b9/diagnostics/report.md) checked the E6-G run and condition seals without changing them. The recorded G1-0 attempt increased its counter from 0 to 20 and passed. The pinned implementation checks five public and five independent private inputs twice each. Its 10 public and 10 private starts are therefore inferred from successful completion and code; individual Docker timestamps and input digests were not recorded. The per-start CSV explicitly labels this reconstruction as inferred. G0-0 used 5 + 5 + 0 starts; its last response repeated a prior repair and was not run. G0-1 used 1 + 1 + 1. Static validation uses zero containers. G1-1 was unstarted because G1's old arm cap was exhausted. Two earlier diagnostic passes used 60 Sandbox starts and two separate seed audits used 70; those are not candidate costs.

No Sandbox execution is cached: the second execution for each input checks determinism. E7 adds pre-dispatch candidate, arm and total limits for Sandbox, calls, reserved tokens and time. A candidate exhaustion is incomplete; an arm or total exhaustion makes the comparison inconclusive. The 20-start candidate cap is the minimum for a passing un-repaired candidate. Smoke has a total of 80 starts because four passing slots need 4 × 20. Full's default 240 reflects the same calculation for 12 slots. Repairs can still exhaust a candidate before verification completes. Full requires a separate user-authorized budget and `--live --allow-full --full-budget 48,300000,240,2400`; the values must match an explicitly authorized budget and the selected configuration. It was not run.

## Live Smoke, 2026-11-01

The [sealed run](../artifacts/e7/live-smoke-20261101-e7a/selection_report.md) registered [the slot schedule and protocol](../artifacts/e7/live-smoke-20261101-e7a/protocol.lock.yml) before inference. Order was G0-0, G1-0, G1-1, G0-1. The fixed seed pool, Qwen3-8B-4bit revision `545dc4251c05440727734bcd94334791f6ab0192`, thinking off, temperature 0.2, tokenizer template, verifier, and Docker image were shared. Client reservation and measured usage are distinct.

| Arm | Completed / planned | Valid | Novel | AST duplicates / completed | Repair success / attempted | LLM calls | Reserved / observed tokens | Docker starts |
| --- | --- | ---: | ---: | --- | --- | ---: | --- | ---: |
| G0 | 2 / 2 | 0 | 0 | 0 / 2 | 0 / 2 | 6 | 13,786 / 1,304 | 13 |
| G1 | 2 / 2 | 2 | 0 | 2 / 2 | not attempted | 2 | 5,362 / 526 | 40 |

[Comparison graph](../artifacts/e7/visualization-live-20261101/comparison.png).

G0-0 ended on repeated repair code; G0-1 reached its repair limit. Both G1 slots passed verification but reproduced an existing seed source and AST. The table counts AST duplicates against the verified pool; G0 also repeated one repair response within a candidate. Their sampled behavior also matched the seed on 2/2 evaluated candidates; finite probes cannot establish semantic equivalence. All eight model attempts returned syntactically valid code (G0 6/6, G1 2/2). Sandbox executions succeeded 8/13 in G0 and 40/40 in G1; these denominators are individual starts, not candidates. The first generation prompt used 135 observed tokens in each G0 slot and 231 in each G1 slot; G0 additionally made repair calls. Total: 8 calls, 19,148 reserved and 1,830 observed tokens, 53 starts, 30.49 seconds. Slot-level costs, requested seeds and prompt lengths are in the run's `slots.json`; raw model responses and code are private. The preregistered selection is **`no_gain_observed`**, an observation from two pairs, not proof that feedback has no effect.

The [independent audit](../artifacts/e7/independent-audit-live-20261101-v1/report.md) reconstructs all eight prompts, verifies Qwen HTTP bodies, requested seeds, pool separation, Sandbox and server ledgers, source hashes, file permissions and seals. Its correction is important: the sealed `comparison.json` reports zero sampled-behavior suspects because its original metric counted only the first novelty classification. Both source duplicates also match sampled behavior; the corrected G1 count is 2/2, and G0 had no behavior fingerprints. The original run is unchanged. The [archived implementation](../artifacts/e7/implementation-snapshot-live-20261101-v2/seal.json) matches its preregistered hashes. The current code fixes this metric. Completed Live resume was verified before the correction with no further inference; the changed implementation now fails the old run's strict identity check. New runs use the corrected code.

## Gate and provenance

| Measure | Count | Evidence and limit |
| --- | ---: | --- |
| `legacy_reported_oracles` | 2 | Historical reporting and records |
| `currently_reverified_oracles` | 2 | Re-audited under the pinned current verifier and Docker image |
| `strict_training_eligible_oracles` | 0 | Both old records lack proof of the historical `src/arc/training.py` dependency; E7 added no new independent Oracle |

The live outcome is **`no_novel_oracles_in_valid_comparison`**. Stop additional exploration under these conditions; do not automatically increase the budget. The five-Oracle threshold is unmet. E7 performed no QLoRA, adapter update, promotion or recursive self-improvement demonstration. Fake fixtures exercised the interface, budget and resume behavior only, and are not model results.

## Resource analysis and proposed E8 (not executed)

The machine reported Apple M4 Pro with 64 GiB unified memory, macOS 26.5.1, MLX-LM 0.30.7. During Smoke the server's sampled resident memory was about 4.6 GiB; this is a few `ps` samples, not a measured peak. Swap used was 786.44 MiB before and after; pageout counters increased, so those snapshots do not establish zero memory pressure. No OOM was recorded. G0 and G1 spent approximately 8.36 s and 2.45 s in logged inference requests; the remaining slot time includes server startup and Sandbox work. G0 used 6.5 Docker starts per completed slot and G1 20. G0's observed completion tokens divided by request elapsed time were about 36 tokens/s; G1's were about 26 tokens/s. These are request-level estimates including prefill, not isolated decoder benchmarks. Reserved tokens were about 10.5 times actual usage. The strongest observed limits are duplicate output and candidate validation cost; this small Smoke does not identify a causal model-capacity bottleneck.

| Next experiment | Hold fixed / change | Resources, cost and verification before execution |
| --- | --- | --- |
| Hardware-only | Hold Qwen revision, quantization, prompts, seeds, verifier, slots and budget; change compute | Current 4-bit weights require roughly 4 GB before overhead; measure actual GPU/host peaks. Kaggle T4x2 has 16 GiB per T4, not pooled memory; availability and quotas vary. MLX and CUDA engines differ, so verify weight conversion, quantization and output differences before attributing speed changes. Kaggle documents retirement of P100 in September 2026. |
| Model-scale | Change to a larger Qwen under the same registered slot protocol | 14B at 4 bits is roughly 7 GB of weights; 32B is roughly 16 GB, both before runtime overhead. Measure actual VRAM/RAM and choose an engine and GPU only after compatibility testing. This changes the model as well as likely the hardware. |
| Budget-scale | Hold model, engine, hardware and protocol fixed; increase only caps | Estimate from observed calls, tokens, 20-start passing slots and startup time; report novel Oracles per cost, not just count. It cannot repair missing historical provenance. |

Kaggle GPU hours and availability can change. [Kaggle's announcement](https://www.kaggle.com/product-announcements/735239) says P100 was retired and identifies T4x2 as an option; verify the actual account allocation before planning. [Google Cloud GPU pricing](https://cloud.google.com/products/compute/gpus-pricing?hl=en) lists a T4 GPU component at USD 0.35/hour on the displayed on-demand price sheet, excluding VM, storage and networking; check region and the current quote before spending. Google AI Pro lists consumer AI features and does not itself establish a general-purpose GPU allocation ([Google One](https://one.google.com/about/google-ai-plans/?gl=US)). Any cloud plan needs encrypted transfer, hashes of model/data/outputs, permission checks and a reproducible environment. Existing MLX adapters are not assumed compatible with a CUDA training stack; validate conversion and inference parity before reuse. Runtime and total price are unmeasured until a bounded benchmark is run.

Priority: retain the current hardware for diagnosis, resolve the missing legacy provenance separately, and consider a separately authorized small model-scale pilot if the objective justifies changing the model. A hardware-only pilot is useful when measured turnaround or memory constraints require it; increasing the same exploration budget is not justified by this zero-novelty Smoke. A minimal E8 proposal is two paired slots per arm with a separately chosen larger open-weight Qwen, the same verifier and an explicitly approved budget; do not execute it as part of E7.

## Commands

```sh
uv run python -m src.cli.e7 --config configs/e7.cpu.fake.yml --check
uv run python -m src.cli.e7 --config configs/e7.cpu.fake.yml --audit-only
uv run python -m src.cli.e7 --config configs/e7.cpu.fake.yml
uv run python -m src.cli.e7 --config configs/e7.live.smoke.yml --live --smoke --run-dir artifacts/e7/live-smoke-20261101-e7a
uv run python -m src.cli.e7_audit artifacts/e7/live-smoke-20261101-e7a
uv run python -m src.cli.e7_independent_audit artifacts/e7/live-smoke-20261101-e7a --snapshot artifacts/e7/implementation-snapshot-live-20261101-v2 --output artifacts/e7/independent-audit-live-20261101-v1
```

The final repository-wide unittest run passed **180 tests**, with no reported skips or failures. All three profile configuration checks, Python compilation, `git diff --check`, the final Fake E2E and its seal audit passed. The guarded-server errors printed during unittest are expected negative-test output. The E7 Smoke profile's parsed settings were included in the protocol before inference and are protected by its hash. The raw `configs/e7.live.smoke.yml` file currently hashes to `2bdec6b7aaf3a8a73db50ae6af927de8fb89458c2491884da97f153c26d97d78`; that byte-level file hash was measured after the run, rather than separately registered beforehand.
