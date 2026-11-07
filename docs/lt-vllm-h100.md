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
nvidia-smi
# Clone this repository and check out the reviewed commit containing #11.
git clone https://github.com/morshoto/qwen-meetup-2.git
cd qwen-meetup-2
python3 -m venv .venv-vllm
. .venv-vllm/bin/activate
python -m pip install 'vllm==0.33.0' 'transformers>=5.8.0' huggingface_hub
python -c 'import importlib.metadata as m; print("vllm", m.version("vllm"), "transformers", m.version("transformers"))'
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
