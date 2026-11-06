"""Register and execute isolated LT exploratory direct-grid batches."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path

from src.arc.lt_batch_journal import BatchJournal, _create
from src.arc.lt_benchmark_manifest import build, canonical, sha, verify
from src.arc.lt_public import build_public_export, load_public_export
from src.arc.lt_supervised_batch import supervised_batch
from src.arc.lt_task_supervisor import run_worker
from src.arc.lt_batch_recovery import reconcile

ROOT = Path(__file__).resolve().parents[2]


def prepare(directory):
    manifest=build(ROOT)
    challenges=json.loads((ROOT/'data/raw/arc-agi_training_challenges.json').read_bytes())
    hashes={task:sha(canonical(challenges[task])) for task in manifest['task_ids']}
    receipt={'comparison_sha256':manifest['sha256'],'selected_task_sha256':hashes}
    export=build_public_export(challenges,receipt)
    directory.mkdir(mode=0o700, parents=False, exist_ok=False)
    _create(directory/'manifest.json',manifest)
    _create(directory/'receipt.json',receipt)
    _create(directory/'export.json',export)
    return manifest['sha256']


def verify_model(manifest, model):
    meta=manifest['models'][model]
    root=Path.home()/'.cache/huggingface/hub'/('models--'+meta['model_id'].replace('/','--'))/'snapshots'/meta['revision']
    files=meta['files_sha256']
    if not root.is_dir() or {p.name for p in root.iterdir()} != set(files): raise ValueError('model inventory differs')
    for name,expected in files.items():
        digest=hashlib.sha256()
        with (root/name).open('rb') as stream:
            for chunk in iter(lambda:stream.read(8*1024*1024),b''): digest.update(chunk)
        if digest.hexdigest()!=expected: raise ValueError('model file hash differs: '+name)
    return root


def run(directory, model):
    manifest=json.loads((directory/'manifest.json').read_bytes());verify(ROOT,manifest)
    receipt=json.loads((directory/'receipt.json').read_bytes())
    export=json.loads((directory/'export.json').read_bytes())
    challenges=json.loads((ROOT/'data/raw/arc-agi_training_challenges.json').read_bytes())
    expected={'comparison_sha256':manifest['sha256'],
              'selected_task_sha256':{task:sha(canonical(challenges[task])) for task in manifest['task_ids']}}
    if receipt != expected or export != build_public_export(challenges,expected):
        raise ValueError('export identity differs')
    tasks=load_public_export(export,receipt)
    if set(tasks)!=set(manifest['task_ids']): raise ValueError('task set differs')
    lockfd=os.open(directory/'run.lock',os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW,0o600)
    try:
        fcntl.flock(lockfd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if model=='32b':
            prior=BatchJournal(directory/'14b',manifest['task_ids'],manifest['sha256']+':14b')
            if prior.pending(1): raise ValueError('14b must finish first')
        model_path=verify_model(manifest,model)
        journal=BatchJournal(directory/model,manifest['task_ids'],manifest['sha256']+':'+model)
        def worker(task, output):
            argv=['uv','run','--offline','--no-project','--python','/opt/homebrew/bin/python3.12',
                  '--with','mlx-lm==0.30.7','python','-m','src.cli.lt_batch_worker',
                  '--export',str(directory/'export.json'),'--receipt',str(directory/'receipt.json'),
                  '--output',str(output),'--task',task,'--model',model,'--model-path',str(model_path)]
            def record_process(pid):
                _create(journal.root/(task+'.process.json'),{'task_id':task,'identity':journal.identity,'pid':pid})
            return run_worker(argv,timeout=manifest['task_timeout_seconds'],on_start=record_process)
        def progress(task,status):
            print(json.dumps({'model':model,'task_id':task,'status':status}),flush=True)
        return supervised_batch(journal,model,manifest['batch_size'],worker,on_result=progress)
    finally:
        os.close(lockfd)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation',choices=('prepare','run','reconcile'))
    parser.add_argument('--directory',required=True,type=Path)
    parser.add_argument('--model',choices=('14b','32b'))
    parser.add_argument('--task')
    args=parser.parse_args();directory=args.directory.resolve()
    if args.operation=='prepare': print(prepare(directory))
    elif args.operation=='run':
        if not args.model: parser.error('--model is required for run')
        print(json.dumps({'completed_this_batch':run(directory,args.model)}))
    else:
        if not args.model or not args.task: parser.error('--model and --task are required for reconcile')
        manifest=json.loads((directory/'manifest.json').read_bytes());verify(ROOT,manifest)
        lockfd=os.open(directory/'run.lock',os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW,0o600)
        try:
            fcntl.flock(lockfd,fcntl.LOCK_EX|fcntl.LOCK_NB)
            journal=BatchJournal(directory/args.model,manifest['task_ids'],manifest['sha256']+':'+args.model)
            print(json.dumps({'status':reconcile(journal,args.task)}))
        finally: os.close(lockfd)


if __name__=='__main__': main()
