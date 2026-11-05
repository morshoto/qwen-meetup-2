# CPU-first ARC experiment (work in progress)

## Setup and execution

From the repository root, install `uv` and run:

```sh
uv run python -m src.cli.experiment --config config.yml --check
uv run python -m src.cli.experiment --config configs/pilot.cpu.yml
uv run python -m src.cli.experiment --config configs/pilot.cpu.yml --resume --run-dir artifacts/<run_id>
uv run python -m unittest discover -s tests/arc -v
uv run python -m src.cli.e6 --config configs/e6.fake.yml --check
uv run python -m src.cli.e6 --config configs/e6.fake.yml
```

The pilot uses small synthetic identity tasks. It does not download a model, use an API, execute generated Python, or access Public Evaluation. `config.yml` is the default configuration example; the CLI requires an explicit `--config`. Challenge and solution paths are relative to the working directory. To use official training, provide the two training JSON paths and split counts 400/400/200. Never point a pilot at Public Evaluation.

## Components and artifacts

`src/core/config.py` validates settings; `src/arc/data.py` loads ARC challenges and isolates solutions in the verifier; `scoring.py` scores complete candidate outputs; `agent.py` implements direct and shared B/C agents; `sandbox.py` defines execution boundaries; `training.py` implements exploration-only records and the dry-run trainer. The separate `src/arc/mlx_lora.py` backend invokes pinned MLX-LM QLoRA, and `src/pipelines/p3b.py` reconstructs SFT records by rerunning the independent verifier over completed B runs. `src/cli/p3b.py` separates synthetic adapter validation from real Exploration training.

A run saves the configuration snapshot, environment, input hashes, split manifest, metrics, per-generation predictions, replay records, JSONL training data and state. A generation is committed by directory rename; resume checks configuration, data hashes and split identity. FakeLLM and FakeSandbox are deterministic demonstrations. FakeSandbox maps fixed symbolic operations and never executes Python. DockerSandbox can execute in an explicitly configured isolated runner; it fails closed when Docker or the image is unavailable. See [p3a-local.md](p3a-local.md) for restrictions and residual risks. DryRunTrainer records a plan and metadata and **does not train or update weights**; its promotion result is `not_applicable`.

## E6 self-generated curriculum

E6 adds Qwen-provenance-checked generator distillation, verifier-only synthetic holdouts, a five-unique-task live training gate, an MLX-LM prompt-mask audit, a guarded adapter reload, matched Promotion Gate evaluation, and rollback. The CPU Fake profile covers Gen 0 through Gen 2 and the fixed Gen 0 data-control interface. The 2026-11-01 live bootstrap retained two distinct verified tasks, so QLoRA, adapter reload, canaries, and the matched gate correctly remained unrun. See [e6-self-generated-curriculum.md](e6-self-generated-curriculum.md) for the full verification record, profile budgets, artifacts, and interpretation boundaries.

Use `configs/e6.local.smoke.yml` with `--live` for the one bounded, training-disabled live smoke. `configs/e6.local.bootstrap.yml` requires `--live --allow-training` and will train only at five or more distinct verified Qwen oracles. Do not repeat a completed live run. A skipped run is not a live training demo.

## Remaining work beyond E6

The local QLoRA path is pinned to the native ARM MLX environment and model revision in `configs/p3b.local.qlora.yml`. The synthetic P3-B1 run measured nonzero adapter weights and a base-versus-adapter inference change while retaining identical base weight hashes. E6 uses the guarded HTTP server, which loads an adapter only when its expected SHA-256 matches and exposes that identity in its contract.

P3-B2's 20-task real B collection and P3-B3's trained-adapter comparison remain separate milestones. E6 training skips below five verified generator-distillation tasks. Reporting, final public evaluation, and comparison figures remain PRD work. A synthetic adapter smoke test is not evidence of ARC improvement.

## Evaluation, budgets, and recovery

`experiment.evaluation_generations` selects evaluation points, and `evaluation_split` selects `exploration` or `monitoring`. The CPU pilot uses exploration only. Set the latter to `monitoring` only when the protocol permits consuming that held-out set. Collection for training always uses exploration IDs, independently of the selected evaluation split. Promotion Gate is reserved and is not consumed by this dry run. No public evaluation is loaded by the pipeline unless a user explicitly supplies it as the data source; do not do so during a pilot.

In the CPU dry-run profile, `budget.max_seconds` is a wall-clock budget for each CLI invocation. It is checked between tasks and before committing a generation; an in-progress backend call is not preempted. An incomplete generation is discarded and `run_state.json` records the next generation; resume starts that generation again with a fresh per-invocation budget. This is suitable for the deterministic fake backend; a real backend requires cancellation, cumulative cost accounting and a non-duplicating request journal before use. The fake sandbox does not enforce the configured container time or memory limits because it does not execute generated code. `max_gpu_hours` is recorded and validated, but no GPU is used in this implementation.

Each committed generation includes hashes for its state, predictions, metrics, replay and training data; resume validates those hashes and rejects gaps or mismatched configuration and dataset hashes. `metrics.csv` records task-level pass@1/pass@2, first-candidate output counts, malformed and timed-out candidates, sample counts, elapsed time, and the checkpoint identity. Token counts are blank because FakeLLM has no tokenizer; GPU seconds are zero. Training JSONL contains exploration prompts and verified generated code, with provenance and training-example repair feedback. The replay buffer is deduplicated, and the configured replay ratio controls sampling of previous records when available.

## Scope and outstanding PRD requirements

The CPU dry-run profile performs no real Qwen inference, QLoRA, GPU resource control, or promotion gate decision. The separate P3-A local profile adds real inference and Docker isolation. It also does not provide the full PRD's report/figure CLI, final Public Evaluation workflow, cost telemetry for real models, or verified weight reload. The next phase must implement and validate a `TrainerBackend` that actually updates and reloads LoRA weights, add a predeclared promotion and rollback rule, and establish GPU and inference budget enforcement before running the three-generation P3 pilot. Monitoring and Public Evaluation must remain protected according to the experiment protocol.

The current quality filter rejects invalid Python, oversized source, missing `transform`, and obvious literal-grid returns after independent verification. It is a conservative heuristic, not proof that a candidate avoids all forms of answer memorization; assess and strengthen it before collecting real-model training data.
