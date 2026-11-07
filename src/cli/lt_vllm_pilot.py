"""Run supervised vLLM development tasks, without solutions or benchmark data."""
import argparse
import fcntl
import importlib.metadata
import json
import os
import re
import sys
from pathlib import Path

from src.arc.lt_batch_journal import BatchJournal, _create
from src.arc.lt_benchmark_manifest import canonical, sha
from src.arc.lt_development_export import load_development_export
from src.arc.lt_public import load_public_export
from src.arc.lt_vllm_backend import VLLM_SETTINGS
from src.arc.lt_single_supervision import supervised_batch
from src.arc.lt_task_supervisor import run_worker
from src.cli.lt_single import verify_model
from src.cli.lt_h100_preflight import check as check_h100

ROOT = Path(__file__).resolve().parents[2]
SOURCES = ('src/cli/lt_h100_preflight.py', 'src/cli/lt_vllm_pilot.py', 'src/cli/lt_single_worker.py', 'src/arc/lt_vllm_backend.py',
           'src/arc/lt_live_backend.py', 'src/arc/lt_worker.py', 'src/arc/lt_task_supervisor.py',
           'src/arc/lt_single_supervision.py', 'src/arc/lt_batch_journal.py', 'src/arc/lt_public.py',
           'src/arc/lt_direct_format.py', 'src/arc/agent.py')


def development_ids():
    return json.loads((ROOT / 'configs/lt.phase-a.development.json').read_bytes())['task_ids']


def development_manifest_sha():
    value = json.loads((ROOT / 'configs/lt.phase-a.development.json').read_bytes())
    return sha(canonical(value))


def _manifest(export, receipt, model, version):
    tasks = load_development_export(export, receipt)
    if receipt['development_manifest_sha256'] != development_manifest_sha():
        raise ValueError('development manifest differs')
    ids = development_ids()
    if set(tasks) != set(ids) or len(ids) != 3:
        raise ValueError('development task set differs')
    if not isinstance(model, dict) or set(model) != {'model_id', 'revision', 'files_sha256', 'chat_template_sha256'}:
        raise ValueError('invalid model identity')
    if (model['model_id'] != 'Qwen/Qwen3.8-27B' or not isinstance(model['revision'], str) or
        not re.fullmatch(r'[0-9a-f]{40}', model['revision']) or not isinstance(model['chat_template_sha256'], str) or
        not re.fullmatch(r'[0-9a-f]{64}', model['chat_template_sha256']) or not isinstance(model['files_sha256'], dict) or
        not model['files_sha256'] or any(not isinstance(k, str) or k in ('', '.', '..') or '/' in k or '\\' in k or
        not isinstance(v, str) or not re.fullmatch(r'[0-9a-f]{64}', v) for k, v in model['files_sha256'].items())):
        raise ValueError('invalid model identity')
    if not isinstance(version, str) or not re.fullmatch(r'\d+\.\d+\.\d+', version):
        raise ValueError('invalid runtime version')
    value = {'purpose': 'development_only', 'task_ids': ids, 'model': model, 'backend_version': version,
             'export_sha256': sha(canonical(export)), 'receipt_sha256': sha(canonical(receipt)),
             'sources_sha256': {p: sha((ROOT / p).read_bytes()) for p in SOURCES},
             'settings': {**VLLM_SETTINGS, 'temperature': 0.6,
                          'max_tokens': 4096, 'candidate_count': 2, 'seed_base': 20261111, 'thinking': 'off'}}
    return {**value, 'sha256': sha(canonical(value))}


def prepare(directory, export, receipt, model, version):
    manifest = _manifest(export, receipt, model, version)
    directory.mkdir(mode=0o700, parents=False, exist_ok=False)
    public = {'comparison_sha256': export['development_manifest_sha256'], 'tasks': export['tasks']}
    public_receipt = {'comparison_sha256': receipt['development_manifest_sha256'],
                      'selected_task_sha256': receipt['selected_task_sha256']}
    for name, value in (('manifest.json', manifest), ('dev-export.json', export), ('dev-receipt.json', receipt),
                        ('export.json', public), ('receipt.json', public_receipt)):
        _create(directory / name, value)
    return manifest['sha256']


def run(directory, *, max_tasks=3):
    if type(max_tasks) is not int or not 1 <= max_tasks <= 3:
        raise ValueError('max_tasks must be between 1 and 3')
    manifest = json.loads((directory / 'manifest.json').read_bytes())
    export = json.loads((directory / 'dev-export.json').read_bytes())
    receipt = json.loads((directory / 'dev-receipt.json').read_bytes())
    if manifest != _manifest(export, receipt, manifest['model'], manifest['backend_version']):
        raise ValueError('pilot registration differs')
    expected_export = {'comparison_sha256': export['development_manifest_sha256'], 'tasks': export['tasks']}
    expected_receipt = {'comparison_sha256': receipt['development_manifest_sha256'], 'selected_task_sha256': receipt['selected_task_sha256']}
    if (json.loads((directory / 'export.json').read_bytes()) != expected_export or
            json.loads((directory / 'receipt.json').read_bytes()) != expected_receipt):
        raise ValueError('pilot public export differs')
    load_public_export(expected_export, expected_receipt)
    fd = os.open(directory / 'run.lock', os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        check_h100(manifest['backend_version'])
        model_path = verify_model(manifest)
        if importlib.metadata.version('vllm') != manifest['backend_version']:
            raise ValueError('runtime version differs')
        journal = BatchJournal(directory / 'single', manifest['task_ids'], manifest['sha256'] + ':single')
        def worker(task, output):
            argv = [sys.executable, '-m', 'src.cli.lt_single_worker', '--export', str(directory / 'export.json'),
                    '--receipt', str(directory / 'receipt.json'), '--output', str(output), '--task', task,
                    '--model-path', str(model_path), '--backend', 'vllm', '--backend-version', manifest['backend_version']]
            def record(pid):
                _create(journal.root / (task + '.process.json'), {'task_id': task, 'identity': journal.identity, 'pid': pid})
            return run_worker(argv, timeout=1800, on_start=record)
        return supervised_batch(journal, 'single', max_tasks, worker, stop_on_failure=True,
                                on_result=lambda task, status: print(json.dumps({'task_id': task, 'status': status}), flush=True))
    finally:
        os.close(fd)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=('prepare', 'run'))
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--export', type=Path)
    parser.add_argument('--receipt', type=Path)
    parser.add_argument('--model', type=Path)
    parser.add_argument('--vllm-version')
    parser.add_argument('--max-tasks', type=int, choices=(1, 2, 3), default=3)
    args = parser.parse_args()
    directory = args.directory.resolve()
    if args.operation == 'prepare':
        if not all((args.export, args.receipt, args.model, args.vllm_version)):
            parser.error('prepare requires --export, --receipt, --model, and --vllm-version')
        print(prepare(directory, json.loads(args.export.read_bytes()), json.loads(args.receipt.read_bytes()),
                      json.loads(args.model.read_bytes()), args.vllm_version))
    else:
        print(json.dumps({'completed_this_batch': run(directory, max_tasks=args.max_tasks)}))


if __name__ == '__main__':
    main()
