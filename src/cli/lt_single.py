"""Prepare and run a registered single-model exploratory ARC experiment."""
import argparse
import fcntl
import hashlib
import json
import os
import sys
from pathlib import Path

from src.arc.data import load_arc
from src.arc.lt_single_score import score_single
from src.arc.lt_batch_journal import BatchJournal, _create
from src.arc.lt_benchmark_manifest import canonical, sha
from src.arc.lt_single_manifest import build, verify
from src.arc.lt_public import build_public_export, load_public_export
from src.arc.lt_single_supervision import supervised_batch, reconcile
from src.arc.lt_task_supervisor import run_worker
from src.cli.lt_benchmark import verify_model as verify_phase_a_model

ROOT = Path(__file__).resolve().parents[2]


def verify_model(manifest):
    root = verify_phase_a_model({'models': {'single': manifest['model']}}, 'single')
    template_file = root / 'chat_template.jinja'
    template = (template_file.read_text() if template_file.is_file() else
                json.loads((root / 'tokenizer_config.json').read_bytes()).get('chat_template'))
    if not isinstance(template, str) or not template or sha(template.encode()) != manifest['model']['chat_template_sha256']:
        raise ValueError('chat template identity differs')
    return root


def prepare(directory, config):
    manifest = build(ROOT, json.loads(config.read_bytes()))
    challenges = json.loads((ROOT / 'data/raw/arc-agi_training_challenges.json').read_bytes())
    receipt = {'comparison_sha256': manifest['sha256'],
               'selected_task_sha256': {task: sha(canonical(challenges[task])) for task in manifest['task_ids']}}
    export = build_public_export(challenges, receipt)
    directory.mkdir(mode=0o700, parents=False, exist_ok=False)
    for name, value in (('manifest.json', manifest), ('receipt.json', receipt), ('export.json', export)):
        _create(directory / name, value)
    return manifest['sha256']


def worker_command(manifest, directory, task, output, model_path):
    backend = manifest.get('backend')
    if backend is None:
        argv = ['uv', 'run', '--offline', '--no-project', '--python', '/opt/homebrew/bin/python3.12',
                '--with', 'mlx-lm==0.30.7', 'python']
    else:
        argv = [sys.executable]
    argv += ['-m', 'src.cli.lt_single_worker', '--export', str(directory / 'export.json'),
             '--receipt', str(directory / 'receipt.json'), '--output', str(output),
             '--task', task, '--model-path', str(model_path)]
    if backend is not None:
        argv += ['--backend', backend['name'], '--backend-version', backend['version']]
    return argv


def run(directory):
    manifest = json.loads((directory / 'manifest.json').read_bytes())
    verify(ROOT, manifest)
    receipt = json.loads((directory / 'receipt.json').read_bytes())
    export = json.loads((directory / 'export.json').read_bytes())
    challenges = json.loads((ROOT / 'data/raw/arc-agi_training_challenges.json').read_bytes())
    expected = {'comparison_sha256': manifest['sha256'],
                'selected_task_sha256': {task: sha(canonical(challenges[task])) for task in manifest['task_ids']}}
    if receipt != expected or export != build_public_export(challenges, expected):
        raise ValueError('export identity differs')
    if set(load_public_export(export, receipt)) != set(manifest['task_ids']):
        raise ValueError('task set differs')
    lockfd = os.open(directory / 'run.lock', os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(lockfd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        model_path = verify_model(manifest)
        journal = BatchJournal(directory / 'single', manifest['task_ids'], manifest['sha256'] + ':single')
        def worker(task, output):
            argv = worker_command(manifest, directory, task, output, model_path)
            def record_process(pid):
                _create(journal.root / (task + '.process.json'),
                        {'task_id': task, 'identity': journal.identity, 'pid': pid})
            return run_worker(argv, timeout=manifest['task_timeout_seconds'], on_start=record_process)
        return supervised_batch(journal, 'single', manifest['batch_size'], worker,
                                on_result=lambda task, status: print(json.dumps({'task_id': task, 'status': status}), flush=True))
    finally:
        os.close(lockfd)


def score_directory(directory):
    manifest = json.loads((directory / 'manifest.json').read_bytes())
    verify(ROOT, manifest)
    tasks, verifier = load_arc(ROOT / 'data/raw/arc-agi_training_challenges.json',
                               ROOT / 'data/raw/arc-agi_training_solutions.json')
    if not set(manifest['task_ids']) <= set(tasks): raise ValueError('missing task')
    path = directory / 'single'
    identity = manifest['sha256'] + ':single'
    rows, hashes, interrupted = [], {}, []
    if path.exists():
        if json.loads((path / 'identity.json').read_bytes()) != {'identity': identity, 'task_ids': manifest['task_ids']}:
            raise ValueError('journal identity differs')
        for task in manifest['task_ids']:
            start, result = path / (task + '.started.json'), path / (task + '.result.json')
            if start.exists() and json.loads(start.read_bytes()) != {'task_id': task, 'identity': identity}:
                raise ValueError('start identity differs')
            if result.exists():
                raw = result.read_bytes()
                row = json.loads(raw)
                if not start.exists() or row.get('model') != 'single' or row.get('task_id') != task or row.get('identity') != identity:
                    raise ValueError('result identity differs')
                rows.append(row)
                hashes[task] = hashlib.sha256(raw).hexdigest()
            elif start.exists():
                interrupted.append(task)
    report = score_single(manifest['task_ids'], rows, verifier)
    report['evidence'] = {'comparison_sha256': manifest['sha256'], 'results_sha256': hashes,
                          'solutions_sha256': manifest['solutions_sha256'], 'protocol': manifest['protocol'],
                          'model': manifest['model'], 'backend': (manifest['backend'] if 'backend' in manifest else {'name': 'mlx', 'version': manifest['mlx_lm_version']}),
                          'task_settings': {key: manifest[key] for key in ('task_ids', 'candidate_count', 'temperature', 'seed_base', 'max_tokens', 'thinking')},
                          'started_without_result': interrupted}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=('prepare', 'run', 'reconcile', 'score'))
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--task')
    args = parser.parse_args()
    directory = args.directory.resolve()
    if args.operation == 'prepare':
        if not args.config: parser.error('--config is required for prepare')
        print(prepare(directory, args.config))
    elif args.operation == 'run':
        print(json.dumps({'completed_this_batch': run(directory)}))
    elif args.operation == 'score':
        print(json.dumps(score_directory(directory), sort_keys=True, allow_nan=False))
    else:
        if not args.task: parser.error('--task is required for reconcile')
        manifest = json.loads((directory / 'manifest.json').read_bytes())
        verify(ROOT, manifest)
        lockfd = os.open(directory / 'run.lock', os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        try:
            fcntl.flock(lockfd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            journal = BatchJournal(directory / 'single', manifest['task_ids'], manifest['sha256'] + ':single')
            print(json.dumps({'status': reconcile(journal, args.task)}))
        finally:
            os.close(lockfd)


if __name__ == '__main__':
    main()
