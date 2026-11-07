"""Fixed synthetic, unscored canary reload prompt and inference settings."""
import hashlib
import json

from src.arc.lt_live_backend import PROMPT

_SAMPLE_TASK = {'train': [{'input': [[0]], 'output': [[1]]}], 'test': [{'input': [[2]]}]}
SAMPLE_PROMPT = json.dumps({'instruction': PROMPT, 'task': _SAMPLE_TASK}, sort_keys=True, separators=(',', ':'))
RELOAD_SETTINGS = {'seed': 20261111, 'max_tokens': 64, 'temperature': 0.6, 'thinking': 'off',
                   'sample_prompt_sha256': hashlib.sha256(SAMPLE_PROMPT.encode()).hexdigest()}
