"""Commit supervised task results only after the owned process group exits."""
import json


def supervised_batch(journal, model, limit, worker, on_result=None):
    if model != 'single': raise ValueError('invalid model')
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
                   'tokens':None, 'runtime_seconds':observation.get('elapsed_seconds'), 'peak_memory_bytes':None,
                   'peak_swap_bytes':None, 'calls':[]}
        else: raise ValueError('unknown worker status')
        row['worker_diagnostics'] = {'stdout':observation.get('stdout'), 'stderr':observation.get('stderr')}
        row['cleanup'] = {'leader_reaped':True, 'group_absent':True, 'returncode':observation.get('returncode')}
        journal.finish(task_id, row)
        if on_result is not None: on_result(task_id, row['status'])
        completed += 1
    return completed


"""Reconcile a stopped task only when its recorded process group is absent."""
import os
import uuid


def reconcile(journal, task):
    if task not in journal.ids: raise ValueError('unknown task')
    root=journal.root; start=root/(task+'.started.json');process=root/(task+'.process.json');result=root/(task+'.result.json');worker=root/(task+'.worker.json')
    if result.exists(): raise ValueError('task already completed')
    try: marker=json.loads(start.read_bytes());record=json.loads(process.read_bytes())
    except (OSError, ValueError) as exc: raise ValueError('missing process evidence') from exc
    if marker != {'task_id':task,'identity':journal.identity} or set(record) != {'task_id','identity','pid'} or record['task_id']!=task or record['identity']!=journal.identity or type(record['pid']) is not int or record['pid']<=1:
        raise ValueError('process evidence differs')
    try: os.killpg(record['pid'],0)
    except ProcessLookupError: pass
    except OSError as exc: raise ValueError('process group cannot be checked') from exc
    else: raise ValueError('process group still exists')
    if worker.exists():
        try: row=json.loads(worker.read_bytes())
        except (OSError, ValueError): row=None
        model=journal.identity.rsplit(':',1)[-1]
        if (isinstance(row,dict) and model == 'single' and row.get('task_id')==task and
                row.get('model')==model and row.get('status')=='complete' and
                isinstance(row.get('candidates'),list) and len(row['candidates'])==2 and
                all(key in row for key in ('tokens','runtime_seconds','peak_memory_bytes','peak_swap_bytes'))):
            row['cleanup']={'leader_reaped':None,'group_absent':True,'returncode':None,'reconciled':True}
            journal.finish(task,row)
            return 'recovered_complete'
    suffix='.interrupted-'+uuid.uuid4().hex
    if worker.exists(): os.rename(worker,root/(task+suffix+'.worker.json'))
    os.rename(process,root/(task+suffix+'.process.json'))
    os.rename(start,root/(task+suffix+'.started.json'))
    fd=os.open(root,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)
    return 'retry_allowed'
