# Kaggle Template

### Description

Develop machine learning models to predict. The goal is to improve understanding.

```bash
├── data           <---- Data directory
├── cli            <---- Command line tools
├── docs           <---- Documents, logs
│   ├── Log.md     <---- Day tracking work
│   ├── Paper.md   <---- Paper research
│   └── Scoring.md <---- Score tracking table
├── paper          <---- Papers to read, get inspired
├── nb             <---- Created on jupyter notebook
├── nb_download    <---- Public notebook from kaggle
├── README.md
├── requirements.txt
├── scripts        <----Utility scripts
├── src            <----Reusable code
│   ├── cli        <----Minimal logic Command-line entry point to call pipelines
│   ├── core       <----Shared config & utilities
│   ├── data       <----Data I/O + preprocessing helpers
│   ├── features   <----Feature engineering
│   ├── models     <----Model implementations
│   └── pipelines  <----Orchestration flows
```

### CLI Usage 

**cli/get_discussion**
This command line tool let you gather kaggle discussion and let you create local markdown files. This levarages you to summarize discussions with other tools

> [!NOTE]
> Initialize `.env` file from `.env.example` file to read competition

### Dataset

The dataset provided for this competition consists of.

| Name | Detail | Size | Link     |
| ---- | ------ | ---- | -------- |
| name |        |      | [Link]() |

### Evaluation

Use **fit -> predict -> evaluate** for validation. Models are responsible for
producing predictions. Evaluation is responsible only for scoring those predictions.
This runnable example uses the placeholder mean baseline and toy validation data:

```pycon
>>> import numpy as np
>>> from src.models.baseline import fit, predict
>>> from src.core.evaluation import evaluate
>>> from src.core.metrics import rmse
>>> X_train, y_train = np.array([[0.0], [1.0]]), np.array([1.0, 3.0])
>>> X_valid, y_valid = np.array([[2.0], [3.0]]), np.array([2.0, 4.0])
>>> model = fit(X_train, y_train)
>>> pred_valid = predict(model, X_valid)
>>> scores = evaluate(y_valid, pred_valid, metrics={"rmse": rmse})
>>> round(scores["rmse"], 4)
1.4142

```

`src/core/metrics.py` defines individual metrics; `src/core/evaluation.py` executes
any mapping of metric names to callables and returns `dict[str, float]`. Add more
metrics to the mapping or configure them with closures or `functools.partial`.
An empty mapping raises `ValueError`; metric exceptions propagate to the caller.
Each metric owns its input-shape requirements, including multiclass matrices.

Keep data splitting, training, inference, and OOF assembly in your pipeline.
The same evaluator accepts baseline, ensemble, post-processed, or OOF predictions.
See `src/pipelines/train.py` for the integration example; its training function
remains a placeholder.

Run the evaluation tests and executable examples with the existing NumPy dependency:

```bash
uv run --no-project --with 'numpy>=2.2.6' python -m unittest discover -s tests/core -v
uv run --no-project --with 'numpy>=2.2.6' python -m doctest README.md src/pipelines/train.py
```

### ARC-AGI-2 CPU experiment

See [docs/implementation.md](docs/implementation.md) for the offline FakeLLM / DryRunTrainer pilot, configuration, artifacts, and remaining GPU work.

For local model inference and isolated code execution, see [docs/p3a-local.md](docs/p3a-local.md).

For the local QLoRA synthetic pilot and verifier-backed Exploration training path, see [docs/p3b-local.md](docs/p3b-local.md).

For offline P3-C failure diagnosis, paired S0–S3 search comparison, and
verifier-backed solution collection without training, see
[docs/p3c-exploration.md](docs/p3c-exploration.md).

The deterministic P3-C fixture path is:

```bash
uv run --locked python -m src.cli.exploration_compare \
  --config configs/p3c.exploration.cpu.yml --dry-run \
  --output-dir artifacts/e2-exploration-cpu-fake
uv run --locked python -m src.cli.exploration_collect \
  --config configs/p3c.exploration.cpu.yml \
  --selection-dir artifacts/e2-exploration-cpu-fake --dry-run \
  --output-dir artifacts/e3-exploration-cpu-fake
```

Live E2 Smoke requires the guarded MLX server and Docker to be started
separately; inspect its bounded preflight before inference:

```bash
uv run --locked python -m src.cli.exploration_compare \
  --config configs/e2.live.smoke.yml --check --live
uv run --locked python -m src.cli.exploration_compare \
  --config configs/e2.live.smoke.yml --live
```

See the P3-C guide for server setup, Resume, E3 gates, and the separate Full
comparison and collection budgets.

For E4 saved-trace diagnosis and the ten-task synthetic transformation-search
comparison, see [docs/e4-analysis.md](docs/e4-analysis.md).

After a complete live E2 Full run, the current run's corrected offline
selection and E3 live commands are documented in
[docs/p3c-exploration.md](docs/p3c-exploration.md). For the retained 2026-11-01
run, the exact progression was:

```bash
uv run --locked python -m src.cli.exploration_compare \
  --reselect-from artifacts/e2-live-full \
  --output-dir artifacts/e2-live-full-reselected

# Start a fresh guarded server with the E3 Smoke token ledger before this check.
uv run --locked python -m src.cli.exploration_collect \
  --config configs/e3.live.smoke.yml \
  --selection-dir artifacts/e2-live-full-reselected --check --live
uv run --locked python -m src.cli.exploration_collect \
  --config configs/e3.live.smoke.yml \
  --selection-dir artifacts/e2-live-full-reselected --live \
  --output-dir artifacts/e3-live-smoke

# E3 Collection is a separate bounded run with its own fresh guarded ledger.
uv run --locked python -m src.cli.exploration_collect \
  --config configs/e3.live.collection.yml \
  --selection-dir artifacts/e2-live-full-reselected --check --live
uv run --locked python -m src.cli.exploration_collect \
  --config configs/e3.live.collection.yml \
  --selection-dir artifacts/e2-live-full-reselected --live \
  --output-dir artifacts/e3-live-collection
```

For the 14B/32B E9 autonomous RSI collection, strict Oracle gate, audit results,
and exact additional-slot approval command, see
[docs/e9-dual-model-rsi.md](docs/e9-dual-model-rsi.md).

For the proposed task-conditioned RSI research design, see the
[v1 protocol](docs/rsi-research-protocol.md) and the
[v2 benchmark amendment](research/rsi/design-v2.md). The amendment supersedes
v1's 26% reference with a 42.4% verified Qwen reference. Both are design
records; neither authorizes execution or establishes an RSI result.
The [extended exposure review](research/rsi/extended-exposure-review.json)
records a bounded scan of additional retained artifacts. It does not establish
that prompts, missing history, or model pretraining are free of exposure.

### Configured single-model ARC experiment

Issue #9 adds a separate exploratory single-model registration on the same
200 Phase A tasks. It uses the Phase A direct-grid prompt, two candidates and
MLX runtime. It does not alter the historical two-model registration or start
inference during preparation. H100/vLLM support is tracked separately in #11.

`configs/lt.single.14b.example.json` illustrates the four required fields:
Hugging Face `model_id`, immutable commit `revision`, SHA-256 hashes of every
file in the snapshot, and the SHA-256 of its tokenizer chat-template string.
For another MLX-compatible model, prepare its snapshot locally under
`~/.cache/huggingface/hub/models--OWNER--NAME/snapshots/REVISION/` and make a
new JSON config with the exact flat file inventory. The runner checks that
inventory and every file hash before inference; it does not download models.
The model must support the existing MLX runtime and chat-template options.
The example does not establish compatibility with every Qwen model.

Place the Phase A ARC training challenges and solutions in `data/raw/` as
`arc-agi_training_challenges.json` and `arc-agi_training_solutions.json`.
The existing allocation, source and model-audit files must also be present.
On the Phase A macOS/MLX environment, create a new, non-existing run directory:

```sh
python -m src.cli.lt_single prepare --config configs/lt.single.14b.example.json --directory artifacts/my-single-run
python -m src.cli.lt_single run --directory artifacts/my-single-run
python -m src.cli.lt_single score --directory artifacts/my-single-run > artifacts/my-single-score.json
```

Each `run` processes at most ten pending tasks and can be repeated. An
interrupted task stops the run; inspect its process evidence and use
`python -m src.cli.lt_single reconcile --directory artifacts/my-single-run --task TASK_ID`
to recover or allow a retry only after the owned process group is absent.
The MLX runner uses `/opt/homebrew/bin/python3.12`, offline `uv`,
`mlx-lm==0.30.7` and `mlx==0.32.4`, matching Phase A. The report records the
configured identity, task and source registration and actual denominators;
missing work remains incomplete. Requested seeds do not establish independent
samples. These results are exploratory and are not official evaluation scores.

For the experimental Linux H100/vLLM backend, see [the setup and run guide](docs/lt-vllm-h100.md). Hardware validation remains outstanding.
