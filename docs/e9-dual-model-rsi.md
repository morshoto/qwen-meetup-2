# E9 dual-model RSI collection

E9 is a model-isolated Gen 0 collection for Qwen3-14B and Qwen3-32B. It reuses the E6-F repair, E6-G novelty, E7 sandbox/budget, and E8 pinned-model checks. A run uses one model only. The only cross-model material in the fixed duplicate-detection pool is reference data; it is never written into the other model's SFT set.

## Frozen identities and rules

| Model | Revision | Quantization | Chat-template SHA-256 |
| --- | --- | --- | --- |
| Qwen3-14B | `a4d9b2df59d2c150bef02fcbe0d91046b7ca33a4` | 4-bit, group size 64 | `87a2728cb8dc9fe424d624542f6060ec05a1d285ebbec578bb078900e33396b5` |
| Qwen3-32B | `bcaaf7f538adf166c1080a2befdb4f6019f66639` | 4-bit, group size 64 | `87a2728cb8dc9fe424d624542f6060ec05a1d285ebbec578bb078900e33396b5` |

Both models used arm64 Python 3.12.9, MLX-LM 0.30.7, MLX 0.32.4, Transformers 5.22.1, tokenizers 0.23.3, NumPy 2.5.4, and Docker image `sha256:a3013c6339a68a667aa9a5f26377693a1f4357a7f7a12db6470a8f06732b91f1`. Thinking was off. Repair remained limited to two model-authored retries. Static checks, public/private probes, Docker isolation, duplicate checks, and provenance requirements were unchanged.

The E8 Full rerun contributed one same-model, strictly verified existing Oracle for each model after its sealed audit and required dependency hashes were checked. These are counted only in `strict_eligible_existing`; they are not E9-new data. The two older E7 Oracles remain excluded because their original static-checker dependency hash is missing. E8 14B and 32B records are both duplicate-detection references, but each model can train only on its own record.

## Live collection

| Run | Planned / completed | Valid candidates | Strict new E9 | Strict eligible existing | Strict total | Calls | Reserved / observed tokens | Sandbox starts | Elapsed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 14B initial, slots 0–9 | 10 / 10 | 3 | 0 | Excluded with run | 0 eligible | 12 | 25,396 / 2,291* | 60 | 91.47 s |
| 14B approved extra, slots 10–39 | 30 / 30 | 15 | 0 | 1 | 1 | 32 | 69,196 / 6,639 | 300 | 256.16 s |
| 32B initial, slots 0–9 | 10 / 10 | 6 | 0 | 1 | 1 | 12 | 25,396 / 3,045 | 200 | 259.51 s |

The 14B extra batch used the approved 1,200-second maximum; it finished in 256.16 seconds. Its 30 slots ended as 15 duplicate candidates and 15 invalid candidates. The 32B slots ended as 6 duplicate candidates and 4 hidden-verification failures. No resource or budget stop occurred. The 14B initial run ended as 3 duplicate candidates and 7 invalid candidates.

`*` The first 14B run is preserved and its tree seal verifies, but the E9 pipeline and CLI source changed after it ran while implementation was still being finalized. Its registered source hashes therefore do not match the final implementation. The sealed run is recorded as `excluded_from_strict_gate` in [its exclusion audit](../artifacts/e9/audits/14b-initial-20261105-excluded/summary.json); it contributes zero training-eligible examples. It was not replayed. Its observed-token value above is independently summed from the sealed request logs; its original result field was null.

The final-source 14B extra batch and 32B initial run passed independent audits: [14B audit](../artifacts/e9/audits/14b-extra30-20261105/summary.json) and [32B audit](../artifacts/e9/audits/32b-initial-20261105/summary.json). Their protocol hashes are `a223e955500925eb58b633a35e1c81fa3a96b2b9d759c78ac2c66c25e92befa0` and `f3b9078412c79eacec4b7f773a3a1c8ef6f539465410a8f0cfe287176bb1a881`, respectively. Each audit checked source hashes, model and chat-template identity, raw request/response provenance, prompt/seed/repair policy, sandbox traces, budget ledgers, SFT provenance, and seals.

During the audited runs, sampled minimum free memory was 73% for 14B and 58% for 32B; sampled peak server RSS was 8,436,512 KiB and 18,321,792 KiB. Swap was 0 MiB before and after each run. These are sampled observations, not guaranteed hardware peaks.

## Strict gate and training outcome

| Model | Strict eligible existing | Strict new E9 | Strict total | Gate | Training / weights / reload / promotion |
| --- | ---: | ---: | ---: | --- | --- |
| 14B | 1 | 0 | 1 / 5 | Not met | Not run / unchanged / not run / not applicable |
| 32B | 1 | 0 | 1 / 5 | Not met | Not run / unchanged / not run / not applicable |

The SFT audit requires the model's own revision, distinct task/AST/behavior fingerprints, the pinned verifier/static-checker hash, sealed source provenance, the same chat template, Thinking off, and prompt-masked assistant-completion loss. The accepted E8 example is labeled `generator_distillation`. No `self_solved_synthetic` or `self_solved_real_arc` examples were produced. The E7 legacy records were rejected from the strict gate.

Because neither model reached five strict Oracles, no QLoRA preflight or training ran, no Adapter was written or reloaded, and no B/C comparison or promotion decision was made. The separate training authorization was not part of the 30-slot approval. Monitoring and Public Evaluation were not used. No cloud environment was started.

## Commands

```bash
uv run python -m src.cli.e9 --config configs/e9.14b.yml --check
uv run python -m src.cli.e9 --config configs/e9.32b.yml --check
uv run python -m src.cli.e9 --config configs/e9.fake.yml

# Initial ten-slot Live runs, one model at a time.
uv run python -m src.cli.e9 --config configs/e9.14b.yml --live --collect-only \
  --run-dir artifacts/e9/14b/initial-20261105
uv run python -m src.cli.e9 --config configs/e9.32b.yml --live --collect-only \
  --run-dir artifacts/e9/32b/initial-20261105

# Additional slots require a separate exact range approval and a time cap.
uv run python -m src.cli.e9 --config configs/e9.14b.extra30.yml --live --collect-only \
  --allow-extra-slots --approve-slot-range 10:40 \
  --run-dir artifacts/e9/14b/extra30-20261105

uv run python -m src.cli.e9_audit --run-dir artifacts/e9/14b/extra30-20261105 \
  --output artifacts/e9/audits/14b-extra30-20261105
uv run python -m src.cli.e9_audit --run-dir artifacts/e9/32b/initial-20261105 \
  --output artifacts/e9/audits/32b-initial-20261105
```

The collection CLI fails closed on `--allow-training`. This delivery implements the collection and strict-gate skip path; it does not expose a QLoRA trainer entrypoint. Neither model qualified for training, and the 30-slot approval covered collection only. `src.arc.operator_search` is not imported or called by E9.
