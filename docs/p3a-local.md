# P3-A: local inference and Docker sandbox

P3-A runs inference only. It never downloads a model or starts an HTTP server as part of an experiment. It does not train or update weights. The default profile uses the guarded local MLX-LM OpenAI-compatible service with `mlx-community/Qwen3-4B-4bit` at revision `4dcb3d101c2a062e5c1d4bb173588c54ea6c4d25`. This machine also has an Ollama profile for the already installed `qwen3.8:27b-mlx`; its model and quantization differ from the MLX default.

## Prepare the data and services

From the repository root, extract only the official **Training** challenge and solution files:

```sh
unzip -o arc-prize-2025.zip arc-agi_training_challenges.json arc-agi_training_solutions.json -d data/raw
uv venv --python 3.12 ~/.venvs/qwen-p3a
uv pip install --python ~/.venvs/qwen-p3a/bin/python mlx-lm==0.30.7 pyyaml
~/.venvs/qwen-p3a/bin/python -c "from huggingface_hub import snapshot_download; print(snapshot_download(repo_id='mlx-community/Qwen3-4B-4bit', revision='4dcb3d101c2a062e5c1d4bb173588c54ea6c4d25'))"
```

Use a native arm64 Python and uv installation for MLX; the project environment on this machine reported x86_64 under Rosetta, so verify the architecture before installing MLX. The install and snapshot download use the network. Start the guarded service as described below, then start Docker Desktop and build the runner separately:

```sh
docker build -t arc-sandbox:local sandbox
uv run python -m src.cli.experiment --config configs/p3a.local.mlx.yml --check
uv run python -m src.cli.experiment --config configs/p3a.local.mlx.yml
uv run python -m src.cli.experiment --config configs/p3a.local.mlx.yml --resume --run-dir artifacts/<run_id>
```

`--check` validates config and local data without contacting a model server or Docker. The default profile selects three Exploration tasks and evaluates A/B at Gen 0. Promotion Gate, Monitoring and Public Evaluation are not used. The official `mlx_lm.server` can instead be started with `mlx_lm.server --model mlx-community/Qwen3-4B-4bit` and used with `configs/p3a.local.external-mlx.yml`. That profile explicitly selects `best_effort` token accounting and does not guarantee a hard token cap. For an already running Ollama server, `configs/p3a.local.ollama.yml` likewise selects `best_effort` and points to loopback and the local model named above. Before inference, the Ollama profile compares its configured full model digest with `/api/tags` and stops on a mismatch. The check is a preflight observation; it does not cryptographically bind later requests to the digest if the server changes its tag during a run.

## Boundaries and evidence

The HTTP backend defaults to loopback, disables proxy use and refuses redirects. Remote URLs require explicit `allow_remote: true`; do not put credentials in config. A request is sent per candidate. Request logs contain hashes, generation parameters, elapsed time, errors and server-reported usage; an absent usage value stays null. Seed is requested but server support is unverified. The generic OpenAI-compatible backend does not claim control of model thinking; malformed or unfinished `<think>` sections are rejected. The optional Ollama-native profile requests `think: false`; support depends on the model and is recorded as requested but unverified. The installed model reported `false` in `/api/show`. See the [Ollama thinking documentation](https://docs.ollama.com/capabilities/thinking). The model server must enforce its own generation limit.

The Docker runner receives only generated code and one grid over stdin. It runs without host mounts or forwarded secrets, with a read-only root, no network, a non-root user, no capabilities, `no-new-privileges`, and configured process, CPU, memory, time, code-size and output limits. Docker and the prebuilt image must be available; there is no fake fallback. Containers reduce risk but do not provide a complete security boundary against hostile code or kernel/container escapes. Use a dedicated machine, keep Docker updated, and do not treat these restrictions as a guarantee.

Each run stores config, environment, split manifest and hashes, request logs, candidate predictions, error records, scores, runtime state and budget journal. Test answers remain in the independent verifier. A/B results use only completed responses; failed requests do not become fabricated zero scores. The sample filter from the earlier dry-run workflow is not used for P3-A training, since there is no training in this phase.

`budget.token_enforcement` defaults to `strict`, which requires the guarded MLX service. External OpenAI-compatible and Ollama profiles must explicitly set it to `best_effort`; that setting is not a hard token guarantee. The run reserves call, candidate, container and conservative token budgets before dispatch. `max_tokens` in the budget is a conservative reservation based on message UTF-8 bytes, a 1,024-token allowance for server templates, and requested completion tokens, not a tokenizer measurement; server-reported usage is kept separately and the run stops if it exceeds the reservation or contains invalid or inconsistent counts. Such an accounting violation permanently halts that run, including on resume. Because a generic server can ignore limits or use an unknown template, this is not a guaranteed bound on server-side token use. Time is checked between operations; HTTP and container deadlines bound individual operations. A server may continue work after the client disconnects, so uncertain requests require inspection. If a request's outcome is uncertain after a disconnect or crash, resume refuses automatic re-dispatch until the journal has been inspected. Per-candidate files and successful request cache prevent reissuing completed requests; source fingerprints and the runner image identity must also match before an unfinished run continues. A completed run is resumed without contacting either service, after checking the candidate and aggregate file hashes. Budget counters, including wall-clock time, are lifetime limits for the run. On recovery from an unclean stop, the elapsed interval until recovery is conservatively charged to the run. A backwards wall-clock change prevents recovery until the journal can be inspected. The inherited `max_gpu_hours` field is not GPU-inference accounting; it remains zero because this phase performs no training. Treat the journal as operational state and do not edit it without understanding whether the server received a request. An OS lock rejects simultaneous writers for one run. Atomic file replacement is used, but power-loss durability is not guaranteed.

## P3-B and later

Implement audited model/adapter identity, real GPU training and weight reload, cost and cancellation enforcement, candidate comparison and promotion/rollback, and validate the security boundary on the deployment host. Reporting and final Public Evaluation remain separate PRD work. A real-model smoke run does not establish improvement.

`agent.max_repairs` is the number of repairs after the first code generation, so the maximum code attempts are `max_repairs + 1`. `agent.candidates` is per task; `budget.max_candidates` is the run-wide cap. `model.max_tokens` is the per-request completion cap. With one candidate, pass@2 is left blank because a second attempt was not made. Different servers may interpret seeds and thinking options differently; the request log distinguishes requested options from verified capabilities.

A local probe on Ollama 0.34.4 with `qwen3.8:27b-mlx` requested `num_ctx: 256` and `truncate: false`, yet the server reported `prompt_eval_count: 1012`. The response and request parameters are preserved in `artifacts/p3a-token-limit-probe/probe.json`. This observation is specific to that installation and is why a requested context size is not treated as proof of a hard token limit.

## Experimental guarded MLX service

An additional local service is available for testing a stricter token boundary. It tokenizes the chat input inside the generation process, reserves the exact prompt length, the requested output limit, and one lookahead token in a persistent SQLite ledger **before** starting model generation. Reservations are atomic and are not refunded after an early stop or crash. The same prompt token IDs are passed to the local MLX-LM driver, and iteration is stopped at the requested output limit. A loopback HTTP endpoint exposes the ledger contract; the `guarded_mlx` experiment profile refuses a server whose declared limit is above the run budget or whose ledger identity changes on resume.

This is an experimental path. The HTTP service and ledger have been tested with a fake token driver and with Qwen3-4B on a native arm64 Mac using MLX-LM 0.30.7. A 36-token ledger admitted two 18-token reservations and rejected the third request before generation; see `artifacts/p3a-guarded-mlx-cap36-proof.json`. The one-token lookahead reservation is based on inspection of the allowlisted `generate_step` source, and the token limit assumes that reviewed driver and the local server are trusted. The server's contract and source hashes are self-reported over localhost, not remote attestation, and the model ID and revision are operator supplied. Pin the reviewed MLX-LM version and local model snapshot before using the path for a hard limit; re-audit before admitting a different generation implementation. The [MLX-LM generation source](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/generate.py) shows why the current adapter reserves a lookahead token; future versions may differ. The external MLX HTTP and Ollama profiles explicitly select `best_effort` and retain the earlier unverified token budget behavior. The default profile requires the guarded service and strict policy.

With a native arm64 environment containing the selected MLX-LM version and a model already present in a local directory, inspect the installed `generate_step` source and compute its SHA-256 with `python -c "import hashlib,inspect; from mlx_lm.generate import generate_step; print(hashlib.sha256(inspect.getsource(generate_step).encode()).hexdigest())"`. The server checks this hash and the installed version against both the supplied values and a reviewed source-hash allowlist; currently only mlx-lm 0.30.7 with the recorded `generate_step` hash is admitted. This detects drift but does not alone prove the behavior of model-specific code. The guarded profile records model revision `4dcb3d101c2a062e5c1d4bb173588c54ea6c4d25`; use that same snapshot and revision on the server, or update both sides together. Run the server separately:

```sh
~/.venvs/qwen-p3a/bin/python -m src.cli.guarded_mlx_server \
  --model-dir /path/to/snapshots/YOUR_PINNED_REVISION \
  --model-id mlx-community/Qwen3-4B-4bit \
  --revision YOUR_PINNED_REVISION \
  --expected-mlx-lm-version YOUR_INSTALLED_VERSION \
  --expected-generate-step-sha256 YOUR_AUDITED_SOURCE_HASH \
  --ledger artifacts/guarded-run.sqlite \
  --total-tokens 100000 --max-prompt-tokens 8192 --max-new-tokens 1024 \
  --thinking off --max-request-seconds 120
uv run python -m src.cli.experiment --config configs/p3a.local.guarded-mlx.yml --check
uv run python -m src.cli.experiment --config configs/p3a.local.guarded-mlx.yml
```

The model directory must be a local snapshot whose directory name equals the full revision; this is a consistency check, not a cryptographic verification of its files. The configured model ID, revision, temperature, thinking mode and token limits must match on both sides. The guarded profile sets thinking to `off` and rejects templates that do not expose `enable_thinking`. Start a new experiment with an empty ledger; preserve that ledger and the same server configuration for resume. The service binds only `127.0.0.1` and does not authenticate other local processes. Its dedicated process exits when a request deadline is exceeded; the committed token reservation remains consumed, and an interrupted request is not automatically retried. Model loading is explicit and accepts an existing local directory; the experiment CLI does not download weights or start this service. A timeout or failure consumes its reservation conservatively.
