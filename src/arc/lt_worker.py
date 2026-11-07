"""Run two direct-agent requests without access to held-out answers."""
import math
import time
from src.arc.agent import direct


def execute_task(task, backend, model):
    if model not in ('14b', '32b'): raise ValueError('invalid model')
    start = time.monotonic()
    candidates, calls = [], []
    for candidate in range(2):
        backend.raw = None
        backend.stats = None
        result = direct(task, backend, candidate)
        candidates.append(result.outputs if not result.error else ('timeout' if result.error_kind == 'timeout' else None))
        calls.append({'candidate': candidate, 'raw': backend.raw, 'usage': backend.stats,
                      'error': result.error, 'error_kind': result.error_kind})
    usage = [x['usage'] for x in calls]
    valid = all(isinstance(x, dict) and type(x.get('prompt_tokens')) is int and
                type(x.get('completion_tokens')) is int and x['prompt_tokens'] >= 0 and
                x['completion_tokens'] >= 0 for x in usage)
    memory = [x.get('peak_memory_gb') for x in usage if isinstance(x, dict)]
    valid_memory = (len(memory) == 2 and all(type(x) in (int, float) and math.isfinite(x) and x >= 0 for x in memory))
    return {'task_id': task.id, 'model': model, 'status': 'complete', 'candidates': candidates, 'calls': calls,
            'tokens': sum(x['prompt_tokens'] + x['completion_tokens'] for x in usage) if valid else None,
            'runtime_seconds': time.monotonic() - start,
            'peak_memory_bytes': int(max(memory) * 1e9) if valid_memory else None, 'peak_swap_bytes': None}
