# E8 Full rerun: 2026-11-04

The user authorized a new 30-slot comparison after the previous Full protocol was found to omit an execution dependency hash. The earlier run remains sealed and its results remain provisional. This rerun used a corrected preregistered protocol, ten fresh slots (20–29) per model, requested attempt seeds 102–131 and separate model pools. Smoke and the previous Full run are excluded.

| Model | Completed / planned | Verified | Strict new | AST duplicates | Sampled behavior matches | Calls | Reserved / observed tokens | Docker starts | Stage seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8B | 10 / 10 | 3 | 0 | 3 | 3 | 26 | 58,299 / 5,167 | 94 | 169.22 |
| 14B | 10 / 10 | 6 | 1 | 5 | 5 | 12 | 25,396 / 2,288 | 120 | 197.08 |
| 32B | 10 / 10 | 4 | 1 | 2 | 3 | 12 | 25,396 / 3,054 | 200 | 387.48 |

AST duplicate and behavior-match counts can overlap. A finite sampled behavior match is a conservative suspicion, not proof of semantic equivalence. Slot outcomes: 8B had 3 passed, 3 repeated-code and 4 repair-limit; 14B had 6 passed and 4 static-safety rejections; 32B had 4 passed and 6 private-acceptance rejections. 8B had 18 runtime-error attempts; no parser failures or successful repairs were recorded. No stage was budget-truncated or stopped early.

| Model | Generation completion tokens/s | Peak sampled RSS KiB | Minimum sampled free memory | Swap before / after MiB | Reserved tokens / strict new | Seconds / strict new |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 8B | 34.67 | 4,811,152 | 85% | 0 / 0 | N/A | N/A |
| 14B | 22.51 | 8,424,992 | 79% | 0 / 0 | 25,396 | 197.08 |
| 32B | 10.83 | 18,335,072 | 58% | 0 / 0 | 25,396 | 387.48 |

Throughput includes prompt prefill in request time. Memory values are samples, not guaranteed peaks. Swap stayed unchanged and no OOM was recorded. Stage time includes preparation and final checks. Each model was limited to 32 calls, 200,000 reserved tokens, 300 Docker starts and 3,600 seconds; each slot was limited to 3 calls, 15,000 reserved tokens and 30 starts. Two canaries count against each model's allowance.

## Provenance and verification

- [Protocol](../artifacts/e8/full-rerun-preregistered-20261102/protocol.lock.yml) SHA-256: `92d2dd93fa5ff0cf9d29bd431c1e1b5a678b5e7823c83b9746a9240860681f97`. Its dependency list includes `src/arc/e7_config.py` and `src/arc/training.py`; model files and tokenizer hashes are recorded in the protocol and stage records.
- Revisions: 8B `545dc4251c05440727734bcd94334791f6ab0192`; 14B `a4d9b2df59d2c150bef02fcbe0d91046b7ca33a4`; 32B `bcaaf7f538adf166c1080a2befdb4f6019f66639`. All are 4-bit group size 64 and use chat-template SHA-256 `87a2728cb8dc9fe424d624542f6060ec05a1d285ebbec578bb078900e33396b5`.
- Initial pool digest: `008a5dd8a68c200e77f5261f566da56fdef40bc283e9d0603e66e9fc1cc143b3`; the two historical Oracles are duplicate references and contribute zero to the strict gate. Docker image digest: `sha256:a3013c6339a68a667aa9a5f26377693a1f4357a7f7a12db6470a8f06732b91f1`.
- On restart, Docker Desktop was started and the temporary native environment was recreated. Observed runtime: arm64 Python 3.12.9, MLX-LM 0.30.7, MLX 0.32.4, Transformers 5.22.1, tokenizers 0.23.3 and NumPy 2.5.4. The MLX-LM `generate_step` SHA-256 matched the registered `1aee463145a78556a66604ace7b6d85dc10035226aa4f909cbefccfa81b49de6`. These package versions describe the restored environment; versions beyond the registered MLX-LM and guarded function hash were not preregistered, so comparisons to earlier runs have that additional limitation.
- [Comparison](../artifacts/e8/report-full-rerun-20261104/report.md), [request, seal and budget audit](../artifacts/e8/audit-full-rerun-20261104/summary.json), [paired execution audit](../artifacts/e8/audit-full-rerun-pairs-20261104/summary.json), and [prior-run exclusion audit](../artifacts/e8/audit-full-rerun-exclusion-20261104/summary.json) are sealed. These checks do not prove semantic equivalence or improvement on ARC tasks.

The suite passed 204 tests before and after the rerun. The three-model 30-slot Fake E2E was completed before Live execution; its fixture results are not included here. Completed-stage resume and interrupted-slot behavior were tested on the preceding implementation and campaign; no completed rerun slot was replayed.

Actual Live commands used the same budget for each stage:

```sh
.venv/bin/python -m src.cli.e8 --config configs/e8.full.rerun.yml --live --allow-full --full-budget 32,200000,300,3600 --stage 8b --campaign-dir artifacts/e8/full-rerun-preregistered-20261102
.venv/bin/python -m src.cli.e8 --config configs/e8.full.rerun.yml --live --allow-full --full-budget 32,200000,300,3600 --stage 14b --campaign-dir artifacts/e8/full-rerun-preregistered-20261102
.venv/bin/python -m src.cli.e8 --config configs/e8.full.rerun.yml --live --allow-full --allow-32b --full-budget 32,200000,300,3600 --stage 32b --campaign-dir artifacts/e8/full-rerun-preregistered-20261102
```

## Decision

No model reached the preregistered threshold of five strict independent Oracles. The 14B and 32B results are each one candidate under this fixed protocol, not evidence of model improvement through RSI. No QLoRA training, adapter update, promotion, cloud migration or added exploration was performed. Whether 32B QLoRA training fits available hardware remains unmeasured and requires a separate resource assessment and authorization.
