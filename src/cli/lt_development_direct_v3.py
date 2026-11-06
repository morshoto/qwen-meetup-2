"""Run the existing direct ARC agent on the registered development tasks.

This exploratory runner generates grids directly; it never executes model code.
Score results only after this process has terminated.
"""
import json
import os
import time
from pathlib import Path

from src.arc.agent import direct
from src.arc.lt_direct_format import extract_final, normalize_direct
from src.arc.lt_development_export import build_development_export, load_development_export

MODEL = str(Path.home() / '.cache/huggingface/hub/models--mlx-community--Qwen3-14B-4bit/snapshots/a4d9b2df59d2c150bef02fcbe0d91046b7ca33a4')
IDS = ['84db8fc4', 'a64e4611', 'e5c44e8f']


def main():
    import mlx.core as mx
    from mlx_lm import load, stream_generate
    from mlx_lm.sample_utils import make_sampler

    root = Path(__file__).resolve().parents[2]
    manifest = json.loads((root / 'configs/lt.phase-a.development.json').read_text())
    registry = root / 'research/rsi/bounded-exposure-proposal.json'
    export, receipt = build_development_export((root / 'data/raw/arc-agi_training_challenges.json').read_bytes(), registry.read_bytes(), manifest)
    tasks = load_development_export(export, receipt)
    if sorted(tasks) != IDS:
        raise ValueError('development selection differs')
    out = Path(os.environ['LT_DEVELOPMENT_RUN_DIR'])
    out.mkdir(mode=0o700, parents=False, exist_ok=False)
    (out / 'receipt.json').write_text(json.dumps(receipt, sort_keys=True))
    thinking = os.environ.get('LT_THINKING', 'off')
    if thinking not in ('on', 'off'): raise ValueError('invalid thinking setting')
    max_tokens = 4096
    model, tokenizer = load(MODEL)

    class Backend:
        def generate(self, prompt, *, kind, candidate):
            self.stats = None
            text = ''
            payload = json.loads(prompt)
            count = len(payload['task']['test'])
            payload['instruction'] = ('Return only JSON: an array of grids, one per test input. '
                'Syntax-only example for one output: [[[0,1],[2,3]]]. '
                'Do not copy the example. Compute the output for every actual test input. '
                'Each grid is an array of integer rows. No object wrappers, markdown, or explanation.')
            prompt = json.dumps(payload, sort_keys=True, separators=(',', ':'))
            formatted = tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}], tokenize=False,
                                                       add_generation_prompt=True, enable_thinking=thinking == 'on')
            mx.random.seed(20261111 + candidate)
            for response in stream_generate(model, tokenizer, prompt=formatted, max_tokens=max_tokens,
                                            sampler=make_sampler(temp=0.6)):
                text += response.text
                self.stats = {'prompt_tokens': response.prompt_tokens, 'completion_tokens': response.generation_tokens,
                              'peak_memory_gb': response.peak_memory, 'finish_reason': response.finish_reason}
            self.raw = text
            try:
                return json.dumps(normalize_direct(extract_final(text, thinking == 'on'), count), separators=(',', ':'))
            except ValueError:
                return 'null'  # Keep malformed or ambiguous output invalid in the agent.

    backend = Backend()
    with (out / 'results.jsonl').open('x') as stream:
        for task_id in IDS:
            start = time.monotonic()
            candidates, calls = [], []
            for candidate in range(2):
                result = direct(tasks[task_id], backend, candidate)
                candidates.append(result.outputs if not result.error else None)
                calls.append({'candidate': candidate, 'error': result.error, 'error_kind': result.error_kind,
                              'raw': backend.raw, 'usage': backend.stats})
            usage = [c['usage'] for c in calls]
            tokens = sum(x['prompt_tokens'] + x['completion_tokens'] for x in usage) if all(usage) else None
            row = {'model': '14b', 'task_id': task_id, 'status': 'complete', 'candidates': candidates,
                   'development_manifest_sha256': receipt['development_manifest_sha256'], 'tokens': tokens,
                   'runtime_seconds': time.monotonic() - start,
                   'peak_memory_bytes': int(max(x['peak_memory_gb'] for x in usage) * 1e9) if all(usage) else None,
                   'peak_swap_bytes': None, 'calls': calls, 'thinking': thinking, 'max_tokens': max_tokens}
            stream.write(json.dumps(row, sort_keys=True) + '\n'); stream.flush(); os.fsync(stream.fileno())
            print(task_id, 'recorded', flush=True)


if __name__ == '__main__':
    main()
