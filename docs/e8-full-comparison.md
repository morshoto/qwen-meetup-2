# E8 Full: Qwen3 model comparison

The local Full experiment compares autonomous Oracle generation, not RSI performance improvement. No training, weight or adapter update, promotion, or cloud execution occurred. Each model used ten new G0 slots (IDs 10–19; requested attempt seeds 72–101), the same duplicate-only initial pool, verifier and bounded budget. The prior Smoke data were not included in the Full counts. The two historical seed Oracles lack complete historical provenance and contribute zero to the strict training gate.

## Results

**Provenance limitation:** the Full protocol omitted the hash of `src/arc/e7_config.py`, an execution dependency. Its current hash matches an earlier sealed E7 protocol, but that does not establish complete Full preregistration. The [sealed dependency-gap audit](../artifacts/e8/audit-full-dependency-gap-20261102/summary.json) records the finding. Consequently the 0/1/1 counts below are the pipeline's recorded results, provisional under the specified strict provenance gate; they must not be used to claim strict training eligibility.

| Model | Full slots | Verified | Strict new | AST duplicates | Sampled behavior matches | Calls | Reserved / observed tokens | Docker starts | Wall seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8B | 10 / 10 | 2 | 0 | 2 | 2 | 27 | 60,559 / 5,325 | 98 | 158.22 |
| 14B | 10 / 10 | 3 | 1 | 1 | 2 | 12 | 25,396 / 2,291 | 60 | 176.59 |
| 32B | 10 / 10 | 6 | 1 | 3 | 5 | 12 | 25,396 / 3,039 | 200 | 386.24 |

Duplicate counts are candidate-level observations and can overlap behavior matches; finite sampled matches are suspicion, not proof of semantic equivalence. The slot outcomes were: 8B, six repair-limit, one repeated-code, one private-acceptance rejection and two passed; 14B, seven static-safety rejections and three passed; 32B, four private-acceptance rejections and six passed. Eight-B recorded 21 runtime-error attempts; no successful repairs or parser failures were recorded for any model. Rejected private results were not sent back as repair feedback.

| Model | Generation tokens/s | Peak sampled RSS KiB | Minimum sampled free memory | Swap before / after MiB | Reserved tokens / strict novel | Wall seconds / strict novel |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 8B | 35.31 | 4,804,448 | 73% | 786.44 / 786.44 | N/A | N/A |
| 14B | 22.97 | 8,421,248 | 67% | 786.44 / 786.44 | 25,396 | 176.59 |
| 32B | 11.18 | 16,378,000 | 44% | 786.44 / 786.44 | 25,396 | 386.24 |

Throughput is completion tokens divided by request time including prefill; RSS and free memory are periodic samples, not guaranteed peaks. No OOM, swap increase, budget truncation or stage stop was recorded. Ten slots are a small sample; differences between model files, tokenizer and numerical execution remain. All three model snapshots use 4-bit group size 64 and the same recorded chat-template hash.

Previous Smoke, shown separately: 8B 0/2, 14B 0/2 and 32B 1/2 strict new Oracles. The Full counts above do not include the Smoke result.

## Identity and evidence

- Full protocol SHA-256: `33f069d5f35076d860f8e21d8cfa6e08590f4694974e4d8567b977df23ab34c8`.
- Initial pool digest: `008a5dd8a68c200e77f5261f566da56fdef40bc283e9d0603e66e9fc1cc143b3`; historical pool entries are used for duplicate detection only.
- Docker image: `sha256:a3013c6339a68a667aa9a5f26377693a1f4357a7f7a12db6470a8f06732b91f1`; MLX-LM `0.30.7`.
- Revisions: 8B `545dc4251c05440727734bcd94334791f6ab0192`; 14B `a4d9b2df59d2c150bef02fcbe0d91046b7ca33a4`; 32B `bcaaf7f538adf166c1080a2befdb4f6019f66639`.
- The registered verifier dependency list includes `src/arc/training.py`, SHA-256 `a1be2d273ed3e61f09abb6424726ad169fc60a7e380ed62dcdcb1b3e5ef8d5ac`. Model and tokenizer file hashes, template hashes and settings are in the [protocol](../artifacts/e8/full-20261102/protocol.lock.yml).
- Stage seal SHA-256: 8B `ced69dd0de52a6bd0a3203eafefea7a8bf24293146e15f13dd8538069953d215`; 14B `013f9c316979113688b12b44c2a4aae7b68221ea608941ab004db35f1bd3faa2`; 32B `90b3fed6627b79f326584ed312c9cd13e0dbb6d368080742c621695b8563e5b9`.
- [Comparison](../artifacts/e8/report-full-20261102/report.md), [request and budget audit](../artifacts/e8/audit-full-20261102/summary.json), and [paired execution and Smoke separation audit](../artifacts/e8/audit-full-pairs-20261102/summary.json) are independently sealed. The audits check recorded evidence; they do not prove semantic equivalence or real ARC performance.

The preregistered limits per model were 32 calls, 200,000 reserved tokens, 300 Sandbox starts and 3,600 seconds; each slot had at most 3 calls, 15,000 reserved tokens and 30 starts. The three-model aggregate is bounded by their sum. Canaries count against model budgets. Model preparation is included in the measured stage limit, making it stricter than a separate loaded-model allowance.

## Commands and verification

```sh
.venv/bin/python -m src.cli.e8 --config configs/e8.full.yml --check
.venv/bin/python -m src.cli.e8 --config configs/e8.full.fake.yml --campaign-dir artifacts/e8/fake-full-20261102-v2
.venv/bin/python -m unittest discover -s tests -q
.venv/bin/python -m src.cli.e8_full_prepare --config configs/e8.full.yml --smoke artifacts/e8/live-smoke-32b-retry-3600-20261101 --campaign-dir artifacts/e8/full-20261102
.venv/bin/python -m src.cli.e8 --config configs/e8.full.yml --live --allow-full --full-budget 32,200000,300,3600 --stage 8b --campaign-dir artifacts/e8/full-20261102
.venv/bin/python -m src.cli.e8 --config configs/e8.full.yml --live --allow-full --full-budget 32,200000,300,3600 --stage 14b --campaign-dir artifacts/e8/full-20261102
.venv/bin/python -m src.cli.e8 --config configs/e8.full.yml --live --allow-full --allow-32b --full-budget 32,200000,300,3600 --stage 32b --campaign-dir artifacts/e8/full-20261102
.venv/bin/python -m src.cli.e8_audit artifacts/e8/full-20261102 --output artifacts/e8/audit-full-20261102
.venv/bin/python -m src.cli.e8_report artifacts/e8/full-20261102 --output artifacts/e8/report-full-20261102
.venv/bin/python -m src.cli.e8_full_audit artifacts/e8/full-20261102 --smoke artifacts/e8/live-smoke-32b-retry-3600-20261101 --output artifacts/e8/audit-full-pairs-20261102
```

The 30-slot Fake E2E finished; its fixture values are not Live results. The suite passed 203 tests before Live execution and 204 after the supplemental audit was added. All three completed stages were resumed without changing their seals or counters, as recorded in the [resume audit](../artifacts/e8/audit-full-resume-20261102/summary.json); interrupted-slot and hash-change cases are covered by automated tests. Existing Smoke's historical independent audit remains sealed; re-running its old audit against changed Full implementation correctly rejects changed dependencies.

## Decision

No model reached five recorded new independent Oracles, and the dependency gap prevents treating the recorded counts as fully strict provenance evidence. No model qualifies for the planned Gen 0→1 QLoRA technical experiment. The 14B and 32B results show one provisional candidate each under these conditions, not proven RSI or ARC performance improvement. A fully compliant rerun would require a new protocol with the complete dependency hashes and separate approval for additional budget; a partial rerun cannot be compared against the old stages. The Mac's 32B inference evidence does not establish whether 32B QLoRA training fits: training memory, optimizer and activation requirements remain unmeasured and require separate evaluation and authorization.

The corrected [comparison](../artifacts/e8/report-full-corrected-20261102/report.md) leaves the original sealed report intact. The implementation now records `e7_config.py` and the other identified imported modules in future protocol dependency hashes. A new [pre-registered protocol](../artifacts/e8/full-rerun-preregistered-20261102/protocol.lock.yml) uses fresh slot IDs 20–29 and attempt seeds 102–131, excluding both Smoke and this Full run. Its SHA-256 is `92d2dd93fa5ff0cf9d29bd431c1e1b5a678b5e7823c83b9746a9240860681f97`. No model call or Sandbox execution has been made in that campaign. The same per-model ceilings would apply if a separate rerun were approved.
