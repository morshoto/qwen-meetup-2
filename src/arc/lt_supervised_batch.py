"""Commit supervised task results only after the owned process group exits."""
import json


def supervised_batch(journal, model, limit, worker):
    if model not in ('14b', '32b'): raise ValueError('invalid model')
    completed = 0
    for task_id in journal.pending(limit):
        journal.start(task_id)
        path = journal.root / (task_id + '.worker.json')
        if path.exists(): raise ValueError('existing worker output requires reconciliation')
        observation = worker(task_id, path)
        if observation.get('leader_reaped') is not True or observation.get('group_absent') is not True:
            raise ValueError('owned process cleanup unverified')
        status = observation.get('status')
        if status == 'complete':
            try: row = json.loads(path.read_bytes())
            except (OSError, ValueError) as exc: raise ValueError('missing or malformed worker result') from exc
            if (not isinstance(row, dict) or row.get('task_id') != task_id or row.get('model') != model or
                    row.get('status') != 'complete' or not isinstance(row.get('candidates'), list) or
                    len(row['candidates']) != 2):
                raise ValueError('worker result identity differs')
        elif status in ('timeout', 'error'):
            row = {'task_id':task_id, 'model':model, 'status':status, 'candidates':[],
                   'tokens':None, 'runtime_seconds':None, 'peak_memory_bytes':None,
                   'peak_swap_bytes':None, 'calls':[]}
        else: raise ValueError('unknown worker status')
        row['worker_diagnostics'] = {'stdout':observation.get('stdout'), 'stderr':observation.get('stderr')}
        row['cleanup'] = {'leader_reaped':True, 'group_absent':True, 'returncode':observation.get('returncode')}
        journal.finish(task_id, row)
        completed += 1
    return completed
