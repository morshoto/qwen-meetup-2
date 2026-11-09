"""Real-model inference canary; no ARC examples or solutions."""
import argparse
import json
import platform
import time
from pathlib import Path
from src.arc.qwen38_local_runner import Backend, atomic_json

parser = argparse.ArgumentParser()
parser.add_argument('--model-path', required=True)
parser.add_argument('--output', required=True)
args = parser.parse_args()
started = time.monotonic()
backend = Backend(args.model_path)
load_seconds = time.monotonic() - started
response = backend.call('Reply with exactly the word READY.', 20261111, max_tokens=32)
record = {'architecture': platform.machine(), 'model_path': args.model_path,
          'load_seconds': round(load_seconds, 3), 'response': response}
atomic_json(Path(args.output), record)
if not response['raw'].strip():
    raise RuntimeError('Model returned no text; inspect smoke record.')
print(json.dumps(record, indent=2), flush=True)
