"""Reconcile a stopped task only when its recorded process group is absent."""
import json
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
    suffix='.interrupted-'+uuid.uuid4().hex
    if worker.exists(): os.rename(worker,root/(task+suffix+'.worker.json'))
    os.rename(process,root/(task+suffix+'.process.json'))
    os.rename(start,root/(task+suffix+'.started.json'))
    fd=os.open(root,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)
