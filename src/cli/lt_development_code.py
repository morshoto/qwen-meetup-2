"""LT development code-agent run with checked Docker execution."""
import json
import os
import time
from pathlib import Path
from src.arc.agent import code_agent
from src.arc.http_backend import BackendError
from src.arc.lt_checked_sandbox import CheckedSandbox
from src.arc.lt_development_export import build_development_export, load_development_export
from src.cli.lt_development_direct import IDS, MODEL


def main():
    import mlx.core as mx
    from mlx_lm import load, stream_generate
    from mlx_lm.sample_utils import make_sampler
    root = Path(__file__).resolve().parents[2]
    manifest = json.loads((root / 'configs/lt.phase-a.development.json').read_text())
    export, receipt = build_development_export((root / 'data/raw/arc-agi_training_challenges.json').read_bytes(),
        (root / 'research/rsi/bounded-exposure-proposal.json').read_bytes(), manifest)
    tasks = load_development_export(export, receipt)
    if sorted(tasks) != IDS: raise ValueError('development selection differs')
    out = Path(os.environ['LT_DEVELOPMENT_RUN_DIR']); out.mkdir(mode=0o700, parents=False, exist_ok=False)
    (out / 'receipt.json').write_text(json.dumps(receipt, sort_keys=True))
    model, tokenizer = load(MODEL)
    sandbox = CheckedSandbox(image='sha256:a3013c6339a68a667aa9a5f26377693a1f4357a7f7a12db6470a8f06732b91f1', timeout=5)
    class Backend:
        def __init__(self): self.calls = []
        def generate(self, prompt, *, kind, candidate):
            formatted = tokenizer.apply_chat_template([{'role':'user','content':prompt}], tokenize=False,
                                                       add_generation_prompt=True, enable_thinking=False)
            mx.random.seed(20261111 + candidate)
            text = ''; stats = None
            for response in stream_generate(model, tokenizer, prompt=formatted, max_tokens=2048,
                                            sampler=make_sampler(temp=0.6)):
                text += response.text
                stats = {'prompt_tokens':response.prompt_tokens, 'completion_tokens':response.generation_tokens,
                         'peak_memory_gb':response.peak_memory, 'finish_reason':response.finish_reason}
            self.calls.append({'kind':kind, 'candidate':candidate, 'raw':text, 'usage':stats})
            return text
    backend = Backend()
    with (out / 'results.jsonl').open('x') as stream:
        for task_id in IDS:
            start = time.monotonic(); before = len(backend.calls); containers = len(sandbox.observations)
            candidates = []; failures = []; fatal = None
            for candidate in range(2):
                try:
                    result = code_agent(tasks[task_id], backend, sandbox, 0, candidate)
                    candidates.append(result.outputs if not result.error else None)
                    failures.append({'candidate':candidate, 'error':result.error, 'error_kind':result.error_kind,
                                     'history':result.history})
                except BackendError as exc:
                    fatal = str(exc); break
            calls = backend.calls[before:]; usage = [c['usage'] for c in calls]
            row = {'model':'14b', 'task_id':task_id, 'status':'error' if fatal else 'complete',
                   'candidates':candidates, 'development_manifest_sha256':receipt['development_manifest_sha256'],
                   'tokens':sum(x['prompt_tokens']+x['completion_tokens'] for x in usage) if usage and all(usage) else None,
                   'runtime_seconds':time.monotonic()-start,
                   'peak_memory_bytes':int(max(x['peak_memory_gb'] for x in usage)*1e9) if usage and all(usage) else None,
                   'peak_swap_bytes':None, 'calls':calls, 'failures':failures, 'fatal':fatal,
                   'containers':sandbox.observations[containers:]}
            stream.write(json.dumps(row,sort_keys=True)+'\n');stream.flush();os.fsync(stream.fileno())
            print(task_id, 'recorded', flush=True)
            if fatal: raise RuntimeError('container cleanup or backend failure')


if __name__ == '__main__': main()
