# E5 Autonomous RSI

E5 tests whether a Qwen checkpoint can author synthetic tasks, solve them independently, and provide verifier-approved SFT records for a later adapter update. Human-authored code is limited to experiment control, grid/data validation, Docker isolation, verification, training, scoring, budgets, and artifact handling. E5 does not import or call `src/arc/operator_search.py`.

## Conditions and data flow

- **A** uses direct JSON predictions.
- **B** uses the fixed E5 code agent with the base model.
- **C1** uses the same agent, prompts, task windows, model limits, and verifier as B. Only the adapter can differ after a promotion.
- **C2** is a separate Fake-only proposal path. Qwen proposals are stored under `gen-N/plugins/proposals/`; the fake evaluator checks the static interface and does not execute proposal code. It cannot change or run C1.

For each C1 generation, E5 selects a disjoint window from the ARC Training **Exploration** split, asks the current model for transform code, runs it in the configured sandbox on generated rectangular integer grids, and constructs training pairs plus held-out and independent validation inputs. The solver receives only the training pairs and test inputs. Generator source and held-out outputs stay outside its prompt. Only a self-solved program that passes the independent hidden-output and validation checks can become `self_solved_synthetic` SFT. Real ARC records are separately typed as `self_solved_real_arc` and require the ARC verifier to pass.

`generator_distillation` is rejected by the E5 training audit. Obvious fixed output, direct grid-literal lookup, unsafe code, execution errors, degenerate generators, and repeated synthetic task content are rejected. Generation prompts describe grid shape, input/output, and sandbox constraints; they do not specify transformations.

Rejected generator candidates receive only their own source, a sanitized execution category, and the failed random input's dimensions. This allows Qwen to repair indexing without showing it grid values or outputs. Solver retries receive training-pair mismatch indices or a generic output-format error; E5 does not repeat large training grids in feedback, and it never returns held-out outputs.

The official 400/400/200 Training split remains Exploration/Promotion Gate/Monitoring. Per-generation Exploration windows and per-promotion Promotion Gate windows do not overlap. Promotion Gate data is used only to compare the previous and candidate adapters. Monitoring data is used only at configured generations. Neither split's labels or score details are included in model prompts. Public Evaluation is not read.

## Configuration and commands

The `e5` group in `config.yml` contains the protocol defaults. Three runnable profiles are provided:

| Config | Purpose | Boundaries |
| --- | --- | --- |
| `configs/e5.cpu.fake.yml` | Gen 0 through Gen 2 wiring and resume test | Fake model/sandbox, dry-run training, small fixture dataset |
| `configs/e5.live.smoke.yml` | Local Qwen smoke | Gen 0 only, 16 calls, 150,000 reserved tokens, 100 Sandbox runs, 16,384 prompt-token limit, and a minimum-sample threshold that prevents training |
| `configs/e5.local.3gen.yml` | Prepared local three-generation pilot | Qwen3-8B guarded MLX, Docker, QLoRA, fixed limits; not auto-started |

Validate a profile without running inference:

```sh
python -m src.cli.e5 --config configs/e5.live.smoke.yml --check
```

Live execution requires an explicit flag and an explicit run directory:

```sh
python -m src.cli.e5 --config configs/e5.live.smoke.yml --live \
  --run-dir artifacts/e5-live-smoke
```

Resume a stopped run with the same command arguments and `--resume`. E5 checks the configuration, data, split, implementation, and pinned Docker image identity; verifies completed generation/root manifests and mutable journal/cache seals; and refuses to replay an inference request whose completion is uncertain. A changed identity or artifact hash fails closed.

## Adapter update and promotion

New verified records are deduplicated and combined with the bounded replay buffer. Training is skipped when the number of new verified tasks is below the largest of the E5 and P3B minimums. When enabled and eligible, `train_adapter(..., source="e5_selfplay")` runs the existing MLX-LM QLoRA path, resumes from the prior adapter when configured, writes a new immutable adapter, and verifies its hash and reload probe. The candidate and previous adapter are then evaluated under matched settings on the Promotion Gate. Promotion requires complete evidence and the preregistered score delta; incomplete or regressing candidates keep the prior adapter. Each generation records `trained`, `weights_updated`, `promoted`, and `rolled_back` separately.

## Verification record

The following are local runs from 2026-11-01. Fake metrics are test-fixture evidence and are not model performance.

### CPU Fake Gen 0 → 1 → 2

Artifact: `artifacts/e5-cpu-fake-20261101-final-verified/`.

- All three generations completed, with two generated/valid/self-solved synthetic tasks and two synthetic SFT records per generation.
- Each generation also recorded one verifier-passing fake ARC task as `self_solved_real_arc`; the three ARC task IDs were disjoint.
- Each generation had three new verified tasks, below the configured five-task training threshold. QLoRA was skipped; `trained=false`, `weights_updated=false`, and `promoted=false`.
- The C2 fake proposal passed its static contract fixture, was written only below `plugins/proposals`, and records `used_by_c1=false`.
- Total: 24 Fake model calls, 53,751 reserved-token estimate, 108 Fake Sandbox runs, 0.086 seconds wall time. Fake token usage is not observable.
- Resume returned the completed run after hash verification. A separate test modified a sealed generation artifact and confirmed resume rejection.
- Per-task results retain task IDs, predicted outputs, code hashes, and pass/error status; held-out labels are not sent to the model.

### Local Qwen3-8B Live smoke

Artifact: `artifacts/e5-live-smoke-20261101-compact-feedback-final/`.

- The guarded MLX server used `mlx-community/Qwen3-8B-4bit` revision `545dc4251c05440727734bcd94334791f6ab0192` and the Docker image ID recorded in `identity.json`.
- Gen 0 used 12 model calls, 38,573 observed tokens (67,609 reserved), 40 Sandbox runs, and 125.9 seconds wall time. No training steps were used.
- Qwen's first generator candidate failed on a non-square grid. After sanitized dimension and Python-index feedback, its second candidate passed execution, hidden-output, and independent validation checks, producing one synthetic task.
- Both configured self-solver candidates failed the independent checks. E5 recorded 0 self-solved synthetic tasks and 0 synthetic SFT records. The model did not receive the generator code or held-out outputs in its solver prompt.
- B and C1 each had one malformed/non-runnable ARC solution attempt; no real ARC successes or real SFT records were collected. Monitoring also had no valid solution attempts.
- `trained=false`, `weights_updated=false`, `promoted=false`, and `rolled_back=false`. This is a negative smoke result, not evidence of RSI or ARC improvement.

The smoke result is mixed and retained as evidence: Qwen repaired a generated task's rectangular indexing after generic code-level feedback, but could not independently solve that task or produce a verified SFT record. No ARC transformation hint or human-written rule was added.

## Tests and remaining pilot work

`python -m unittest discover -s tests -p 'test*.py' -v` passed all 114 tests on 2026-11-01. The E5 tests cover hidden-label isolation, self-solved provenance, generator repair feedback, compact solver repair feedback, rejection rules, the training handoff, C2 proposal paths, three-generation Fake execution, disjoint Exploration windows, resume integrity, profile validation, and fail-closed Promotion Gate decisions. Fake C2 evidence only confirms static interface validation and proposal-path containment; it does not validate runtime behavior.

The local three-generation QLoRA pilot remains unrun. It should begin only after the live generator can produce sandbox-valid tasks and the model can independently solve enough of them to meet the training threshold. Actual MLX training, adapter reload, Promotion Gate promotion/rollback, and post-promotion inference still need pilot evidence. C2 remains Fake-only and is not active in C1.
