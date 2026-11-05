# E6 Self-Generated Curriculum Bootstrapping

E6 reuses a Qwen-authored, Docker-verified transformation program as a supervised target. This condition is called `generator_distillation`; it does not count as an independent solve. E5's original `self_solved_synthetic` and `self_solved_real_arc` paths remain separate, and E6 does not change the failed E5 self-solve outcomes.

Human-authored code in this phase is limited to experiment plumbing, generic prompt contracts, independent checks, budgets, trainer control, evaluation, and rollback. The E6 execution path does not import `src/arc/operator_search.py`. Fake programs are test fixtures only and cannot be admitted to MLX training.

## Data and provenance

Each accepted teacher record must identify the Qwen request ID, raw response-body SHA-256, model revision, generation, and source type. Live training reopens the retained request and raw-response sidecars, checks their hashes, confirms thinking-off metadata, and verifies that the saved oracle exactly matches the extracted Qwen response. The raw response is never copied into the E6 record.

The oracle is run twice on each of at least five distinct generated inputs through Docker. Checks cover safe function shape, rectangular integer grids, output bounds, output size, repeatability, input dependence, duplicate tasks, duplicate ASTs, and normalized near-copy shapes. Full hardcode detection is impossible; the filter catches obvious cases and tests sampled behavior.

Each generated family is split into visible training pairs, one fresh query input, and separate verifier-only inputs. The SFT user message is exactly the B/C inference prompt from `solve_prompt(task)`, with the shared system instruction. It contains training pairs and a query input, but no verifier-only inputs, outputs, or oracle source. The assistant target is the unchanged Qwen oracle. Verifier-only inputs and outputs are stored separately with mode `0600`.

The five-task gate counts distinct accepted task IDs after exact-oracle and normalized-program-shape deduplication. When the gate is not met, E6 records `skipped_insufficient_data`; it does not lower the live threshold. E5's imported oracle counts only after its sealed task, generation request ID, raw response hash, train pairs, and private held-out verifier are rechecked.

## Conditions and bounded profiles

| Profile | Purpose | Maximums |
| --- | --- | --- |
| `configs/e6.fake.yml` | CPU Fake Gen 0–2 data flow, source isolation, fixed-data D interface, and resume | 100 fake calls, 1,000 fake sandbox fixtures, zero real training steps |
| `configs/e6.local.smoke.yml` | One guarded Qwen collection/verification smoke and actual tokenizer-mask audit | 12 calls, 100,000 reserved tokens, 45 Docker starts, 900 seconds, zero training steps |
| `configs/e6.local.bootstrap.yml` | One Gen 0→1 QLoRA candidate, reload, canaries, and matched gate | 60 calls, 500,000 reserved tokens, 180 Docker starts, 2,400 seconds, 5 training steps, one candidate generation |

Validate settings without starting inference:

```sh
uv run python -m src.cli.e6 --config configs/e6.fake.yml --check
uv run python -m src.cli.e6 --config configs/e6.local.smoke.yml --check
uv run python -m src.cli.e6 --config configs/e6.local.bootstrap.yml --check
```

Run the CPU Fake path:

```sh
uv run python -m src.cli.e6 --config configs/e6.fake.yml
```

The smoke requires `--live`; bootstrap requires both `--live --allow-training`. Resume must use the same config and run directory. Completed results are sealed and returned without regenerating data or retraining. A mismatched config, model, implementation, data hash, or source E5 seal fails closed.

```sh
uv run python -m src.cli.e6 --config configs/e6.local.smoke.yml --live
uv run python -m src.cli.e6 --config configs/e6.local.bootstrap.yml --live --allow-training
```

## Training, evaluation, and rollback

`C1-GD` trains only `generator_distillation` records, uses the pinned MLX-LM QLoRA runtime, and calls the existing prompt-mask/token-span audit before training. A separate tokenizer audit also runs when training is skipped, so thinking-off, chat-template alignment, the shared system prompt, and assistant-completion-only loss are recorded. Fake and non-live records are rejected by the MLX trainer.

Training requires at least five unique verified tasks. The trainer hashes base weights before and after, verifies a changed adapter and a nonzero loaded-adapter probe, then the guarded server loads that exact adapter hash. A generic Python canary and JSON canary must pass. B and C1-GD then use the same fixed Promotion Gate task IDs, agent, prompts, model revision, quantization, thinking mode, and budgets; only the adapter differs. The independent verifier returns scores only to the experiment controller. Monitoring is not used to tune or promote, and Public Evaluation is not read.

Promotion requires complete gate evidence, no runtime errors, and no regression from B. Training failure, weight-delta failure, adapter-load failure, canary failure, incomplete evaluation, or gate regression keeps the prior active adapter. The candidate artifact remains available for audit. `trained`, `weights_updated`, `candidate_loaded`, `promoted`, `rolled_back`, `active_adapter_hash`, and `reason` are recorded separately.

`C1-SS` is reported separately and is skipped when its self-solved dataset is empty. `D` is the fixed Gen 0 data control. In Gen 0→1, C1-GD and D can use the same data; recursive effects are only interpretable from Gen 2 onward. The Fake run verifies the Gen 1 self-generation→Gen 2 data connection. This phase does not run live Gen 2–20.

## Artifacts and interpretation

Each run stores the config snapshot, Git head/dirty paths, implementation and data hashes, source E5 seal, task split manifest, `synthetic_tasks.jsonl`, `verified_oracles.jsonl`, separate `sft_generator_distillation.jsonl` and `sft_self_solved.jsonl`, rejection counts, token-mask audit, budget ledger, training hashes, matched evaluation, decision, and root artifact seal. Request/response logs, oracle and training rows, held-out verifier labels, and rejection JSONL files use mode `0600`; generation directories use mode `0700`. The smoke artifact predates category-only rejection summaries and its owner-only generation manifest contains a bounded sandbox traceback. Raw Qwen response text remains only in its private response log. The follow-up implementation groups dynamic exception details into fixed rejection categories.

## Verification record (2026-11-01)

- The full Python suite passed: 128 tests. All three E6 profile checks passed, and the Fake Gen 0–2 run retained five distinct task and oracle shapes per generation.
- The pinned MLX-LM 0.30.7 tokenizer audit passed on five Fake SFT rows and on both live collections. The live audit used Qwen revision `545dc4251c05440727734bcd94334791f6ab0192`, thinking off, and chat-template SHA-256 `87a2728cb8dc9fe424d624542f6060ec05a1d285ebbec578bb078900e33396b5`.
- Smoke artifact `artifacts/e6/e6-local-smoke-fa35c09b83` completed with one Qwen call, 17 Docker starts, and 2,190 reserved tokens. It retained one sealed E5 oracle; its one newly generated candidate failed isolated execution. Training was disabled for this profile.
- The single bootstrap artifact `artifacts/e6/e6-local-bootstrap-e526bae72f` completed with 8 Qwen calls, 58 Docker starts, 17,520 reserved tokens, and 1,396 observed tokens. It retained two distinct verified tasks/oracle shapes; six candidates failed isolated execution and one duplicated an oracle. The five-task gate skipped training, so training steps remained zero and no adapter or matched Promotion Gate comparison was produced.
- Both E6 root seals verify, and the source E5 root seal is unchanged. The saved E5 diagnostic classifies both E5 self-solve attempts as visible-training mismatches; neither reached held-out evaluation.

The acceptance outcomes are reported separately:

- **Implementation complete:** Yes. Code, Fake Gen 0–2 E2E, 128 tests, docs, seals, and tokenizer audits pass.
- **Live technical demo complete:** No. It requires five or more distinct Qwen oracles, real QLoRA, verified parameter change, adapter load, matched B/C gate, and a saved decision. The bootstrap retained two oracles, so training, adapter checks, and the matched gate were skipped.

Insufficient data is a valid live result but does not meet the live technical-demo bar. This run completed E6-A through E6-C's data path and E6-E's Fake Gen 1-to-2 connection check; conditional E6-D training, adapter reload, canary, and matched gate remain unexecuted because the live collection did not meet its threshold. Even a completed Gen 0→1 training demo is not evidence of recursive improvement or ARC accuracy improvement; five synthetic tasks are a small engineering sample.

