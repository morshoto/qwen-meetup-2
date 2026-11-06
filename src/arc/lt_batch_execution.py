"""Execute caller-supplied public tasks into an exclusive task journal.

The caller owns process deadlines, model identity and cleanup. This module does
not start a model or authorize a benchmark.
"""
import time
from src.arc.agent import direct


def run_batch(journal, tasks, backend, model, limit):
    if model not in ('14b', '32b') or set(tasks) != set(journal.ids):
        raise ValueError('task or model identity differs')
    completed = 0
    for task_id in journal.pending(limit):
        journal.start(task_id)
        start = time.monotonic()
        candidates, calls = [], []
        for candidate in range(2):
            result = direct(tasks[task_id], backend, candidate)
            candidates.append(result.outputs if not result.error else ('timeout' if result.error_kind == 'timeout' else None))
            calls.append({'candidate': candidate, 'raw': getattr(backend, 'raw', None),
                          'usage': getattr(backend, 'stats', None), 'error': result.error,
                          'error_kind': result.error_kind})
        usage = [c['usage'] for c in calls]
        valid_usage = all(isinstance(x, dict) and type(x.get('prompt_tokens')) is int and
                          type(x.get('completion_tokens')) is int and x['prompt_tokens'] >= 0 and
                          x['completion_tokens'] >= 0 for x in usage)
        memory = [x.get('peak_memory_gb') for x in usage if isinstance(x, dict)]
        row = {'task_id': task_id, 'model': model, 'status': 'complete', 'candidates': candidates,
               'calls': calls, 'tokens': sum(x['prompt_tokens'] + x['completion_tokens'] for x in usage) if valid_usage else None,
               'runtime_seconds': time.monotonic() - start,
               'peak_memory_bytes': int(max(memory) * 1e9) if len(memory) == 2 and all(type(x) in (int, float) and x >= 0 for x in memory) else None,
               'peak_swap_bytes': None}
        journal.finish(task_id, row)
        completed += 1
    return completed
