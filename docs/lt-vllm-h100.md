# Qwen3.8 single-model run on Linux

This is a procedure for issue #11, not evidence of a completed H100 run. The
vLLM integration has only been exercised with fixtures; model loading,
compatibility, cost, speed and the 200-task result require later GPU validation.
The run is exploratory and separate from strict RSI and official evaluation.

Use an H100 with 80 GB, sufficient host RAM and disk for the BF16 checkpoint,
CUDA-compatible NVIDIA drivers, Python 3.11 or 3.12, and a trusted environment.
The official [model card](https://huggingface.co/Qwen/Qwen3.8-27B) and
[vLLM recipe](https://recipes.vllm.ai/Qwen/Qwen3.8-27B) describe compatibility;
the recipe states vLLM 0.17.0+ and Transformers 5.8.0+. These instructions use
vLLM 0.33.0 as a pinned starting point, not a validated combination. Record any
change as a new registration. The runner checks the registered vLLM version
before creating its task journal and checks it again in each worker. Do not start a frozen run before applicable
research gates and authorization are met.

```sh
# For the Ubuntu-based image; install a compiler before model download.
apt-get update && apt-get install -y build-essential
nvidia-smi
# Clone this repository and check out the reviewed commit containing #11.
git clone https://github.com/morshoto/qwen-meetup-2.git
cd qwen-meetup-2
python3 -m venv .venv-vllm
. .venv-vllm/bin/activate
python -m pip install 'vllm==0.33.0' 'transformers>=5.8.0' huggingface_hub
python -c 'import importlib.metadata as m; print("vllm", m.version("vllm"), "transformers", m.version("transformers"))'
python -m src.cli.lt_h100_preflight --vllm-version 0.33.0
# For an uv-created virtual environment without pip, record packages with:
uv pip freeze --python "$(command -v python)" > artifacts/runtime-packages.txt
```

Manually place `arc-agi_training_challenges.json` and
`arc-agi_training_solutions.json` under `data/raw/`. Use the same verified Phase
A data, including the allocation and audit files tracked in this repository.
Acquire a pinned official snapshot (the revision below is an example; verify it
against the model publisher before use). This downloads the full model and may
incur storage and network charges:

```sh
python -c "from huggingface_hub import snapshot_download; print(snapshot_download('Qwen/Qwen3.8-27B', revision='37561816e66b5b9153b1b64c10968c18c1e6f4a0'))"
```

Create a local config from that snapshot, hashing every file and the standalone
chat template. Inspect the identity and inventory before preparing the run;
these hashes bind the locally acquired files, but do not independently prove
publisher provenance. This command can take time because it reads all weights.

```sh
mkdir -p artifacts
python - <<'PY'
import hashlib, json
from pathlib import Path
model_id = 'Qwen/Qwen3.8-27B'
revision = '37561816e66b5b9153b1b64c10968c18c1e6f4a0'
root = Path.home() / '.cache/huggingface/hub' / ('models--' + model_id.replace('/', '--')) / 'snapshots' / revision
if not root.is_dir() or any(not p.is_file() for p in root.iterdir()):
    raise SystemExit('snapshot missing or not a flat file inventory')
def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()
config = {'model': {'model_id': model_id, 'revision': revision,
                    'files_sha256': {p.name: digest(p) for p in sorted(root.iterdir())},
                    'chat_template_sha256': digest(root / 'chat_template.jinja')},
          'backend': {'name': 'vllm', 'version': '0.33.0'}}
Path('artifacts/qwen38-vllm.json').write_text(json.dumps(config, indent=2) + '\n')
PY
```

Keep the virtual environment active for all commands. Preparation verifies the
200-task allocation and creates a new run directory; it performs no inference.
The runner reads the snapshot from the standard Hugging Face cache and does not
download weights. Each invocation runs up to 10 pending tasks; repeat until the
score reports status `complete`, with 200 attempted and 400 candidates.
The vLLM batch stops after an error or timeout and records that task as a
terminal failure. Inspect diagnostics and stop; rerunning will not repair that
result. Resolve the cause and prepare a new registration before a clean run.

```sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python -m src.cli.lt_single prepare --config artifacts/qwen38-vllm.json --directory artifacts/qwen38-run
python -m src.cli.lt_single run --directory artifacts/qwen38-run
python -m src.cli.lt_single score --directory artifacts/qwen38-run > artifacts/qwen38-score.json
```

Use `python -m src.cli.lt_single reconcile --directory artifacts/qwen38-run
--task TASK_ID` only after inspecting interruption evidence and confirming the
owned process group is gone. Save the manifest, journal and score before
terminating the instance. The manifest binds the task settings, model inventory,
backend version and code; the score includes these identities and incomplete
counts. The process-group observations do not prove full host or GPU cleanup.

The worker reloads the model per task to retain existing process isolation and
recovery. This may be expensive on rented hardware; measure a small authorized
development pilot, including loading, before budgeting the full run. The
30-minute task timeout includes startup. A timeout or error is a terminal result,
not a successful candidate. Requested seeds do not establish independent samples.

## Development-only compatibility pilot

The pilot is separate from the 200-task single-model runner. On the local
machine, construct the development handoff from the training challenges and
bounded exposure registry; only the resulting public export and receipt need to
be transferred. The export contains the three development tasks, including
public training pairs, but no held-out test answers. Do not transfer the
solutions file for this pilot.

```sh
mkdir -p artifacts/h100-pilot-input
python -m src.arc.lt_development_export --challenges data/raw/arc-agi_training_challenges.json --exposure research/rsi/bounded-exposure-proposal.json --manifest configs/lt.phase-a.development.json --part export > artifacts/h100-pilot-input/export.json
python -m src.arc.lt_development_export --challenges data/raw/arc-agi_training_challenges.json --exposure research/rsi/bounded-exposure-proposal.json --manifest configs/lt.phase-a.development.json --part receipt > artifacts/h100-pilot-input/receipt.json
```

On the H100 host, check out the exact reviewed commit, install the runtime and
prepare the pinned snapshot inventory as above. Copy the development export and
receipt into `artifacts/h100-pilot-input/`, and save the `model` object from
`artifacts/qwen38-vllm.json` as `artifacts/h100-pilot-input/model.json`. Then:

```sh
python -m src.cli.lt_vllm_pilot prepare --export artifacts/h100-pilot-input/export.json --receipt artifacts/h100-pilot-input/receipt.json --model artifacts/h100-pilot-input/model.json --vllm-version 0.33.0 --directory artifacts/h100-pilot
python -m src.cli.lt_vllm_pilot run --directory artifacts/h100-pilot
```

The pilot registers its development-manifest identity, model inventory, vLLM
version, settings and source hashes. It runs up to three tasks, each with two
candidates, and stops after the first error or timeout. Per-task journal rows
record inference time, worker wall time including startup, raw calls, and owned
process-group observations. Calculate startup/teardown overhead from worker wall
time minus inference time; this is not a precise standalone model-load timer.
An interrupted task requires manual inspection; the pilot does not automatically
retry it. Preserve the entire pilot directory and system/runtime evidence before
terminating the instance. This harness does not enforce a spending cap: monitor
cost and time externally, allow enough budget for evidence transfer and cleanup,
and destroy the instance and any billed storage at the deadline.

## Cost and teardown gate

Before renting, check current credit, exchange rate, offer price, requested disk,
download and upload rates, and applicable fees. Budget for the full model
download and retain a reserve for evidence transfer and teardown. If the
remaining funds cannot cover a bounded pilot and the reserve, do not rent.
Use an on-demand instance and record its exact ID and start time.

From the local control machine, start the watchdog immediately after creation,
using a positive delay calculated from the remaining budget and the offer's
total hourly cost (do not include the angle brackets):

```sh
nohup python -m src.cli.lt_vast_watchdog <INSTANCE_ID> --after-seconds <SECONDS> > artifacts/h100-watchdog.log 2>&1 < /dev/null &
```

Verify the process remains alive and the log contains the correct instance ID.
Keep the control machine awake and connected; the watchdog is not a provider-side
spending limit, and machine sleep, network loss, authentication failure or API
failure can prevent teardown. Its command has only been tested with a fake CLI,
not a real H100. Monitor elapsed time, credit and the watchdog log independently.
On completion or failure, copy evidence and destroy the named instance manually:

```sh
vastai destroy instance <INSTANCE_ID> --yes --raw
vastai show instances --raw
vastai show volumes --raw
```

Verify the target is absent, inspect and remove any pilot-owned billed volumes,
and record the final account balance. A failed watchdog or uncertain teardown
requires immediate manual inspection; do not assume the credit balance is a
hard spending cap. Stop or cancel the watchdog only after teardown is verified.
