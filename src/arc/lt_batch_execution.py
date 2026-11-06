"""Execute caller-supplied public tasks into an exclusive task journal.

The caller owns process deadlines, model identity and cleanup. This module does
not start a model or authorize a benchmark.
"""
from src.arc.lt_worker import execute_task


def run_batch(journal, tasks, backend, model, limit):
    if model not in ('14b', '32b') or set(tasks) != set(journal.ids):
        raise ValueError('task or model identity differs')
    completed = 0
    for task_id in journal.pending(limit):
        journal.start(task_id)
        journal.finish(task_id, execute_task(tasks[task_id], backend, model))
        completed += 1
    return completed
