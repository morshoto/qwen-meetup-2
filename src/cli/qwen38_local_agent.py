"""Model-authored task-analysis agent; code executes in an isolated container."""
from __future__ import annotations
import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from src.arc.lt_live_backend import PROMPT
from src.arc.qwen38_agent_sandbox import execute, IMAGE
from src.arc.qwen38_local_runner import Backend, SEED, atomic_json, clean_task, parse_prediction, prepare, proxy_task, run_tasks
from src.arc.qwen38_rsi_comparison import digest

BASE = 'def run(task):\n    return {"note": ""}\n'


def sources(backend, root, manifest, tasks):
    path = root / 'agent' / 'proposals.json'
    if path.exists():
        data = json.loads(path.read_text())
        if data['manifest_sha256'] != manifest['sha256']: raise ValueError('registration mismatch')
        return data['sources']
    feedback = []
    for task_id in manifest['allocation']['development'][:3]:
        row = json.loads((root / 'development/baseline' / (task_id + '.json')).read_text())
        task, expected = proxy_task(tasks[task_id])
        feedback.append({'task': task, 'expected': expected,
                         'attempts': [{'raw': c['raw'][:500], 'correct': c['correct']} for c in row['candidates']]})
    while len(json.dumps(feedback)) > 18000 and feedback: feedback.pop()
    instruction = ('Write one Python module for an ARC agent. Return ONLY Python code, no markdown. '
        'The module must define run(task) returning a JSON object with only the string key note. '
        'The task has train input/output examples and test inputs, all grids. The note will be appended '
        'as task-analysis metadata to a fixed instruction; use only this task, no guessed test answers, '
        'no external access, no model calls. Keep note below 1500 characters. '
        'Improve general task analysis based on these development attempts: ' + json.dumps(feedback, separators=(',', ':')))
    candidates, calls = [], []
    for i in range(2):
        query = instruction + ('\nProduce an alternative approach to the first module.' if i else '')
        response = backend.call(query, SEED + 300 + i, max_tokens=1536)
        code = response['raw'].strip()
        if code.startswith('```') and code.endswith('```'):
            code = '\n'.join(code.splitlines()[1:-1])
        calls.append({'request': query, **response})
        candidates.append(code)
    data = {'manifest_sha256': manifest['sha256'], 'calls': calls, 'sources': candidates}
    atomic_json(path, data)
    return candidates


def run_agent(backend, root, manifest, tasks, label, ids, source, proxy):
    source_sha = hashlib.sha256(source.encode()).hexdigest()
    runner_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    sandbox_path = Path(__file__).resolve().parents[1] / 'arc' / 'qwen38_agent_sandbox.py'
    sandbox_hash = hashlib.sha256(sandbox_path.read_bytes()).hexdigest()
    for index, task_id in enumerate(ids):
        path = root / label / (task_id + '.json')
        identity = digest({'manifest': manifest['sha256'], 'label': label, 'task_id': task_id, 'source_sha': source_sha, 'runner_sha': runner_hash, 'sandbox_sha': sandbox_hash, 'image': IMAGE})
        if path.exists():
            row = json.loads(path.read_text())
            if row.get('identity') != identity or len(row.get('candidates', [])) != 2: raise ValueError('checkpoint mismatch')
            continue
        task, expected = proxy_task(tasks[task_id]) if proxy else (clean_task(tasks[task_id]), None)
        outcome = execute(source, task, timeout=30)
        if not isinstance(outcome, dict) or set(outcome) != {'note'} or not isinstance(outcome['note'], str) or len(outcome['note']) > 1500:
            raise ValueError('agent violated note interface')
        payload = {'instruction': PROMPT, 'task': task}
        if outcome['note']: payload['agent_notes'] = outcome['note']
        content = json.dumps(payload, sort_keys=True, separators=(',', ':'))
        candidates = []
        print(f'{label}: {index}/{len(ids)} task {task_id}', flush=True)
        for i in range(2):
            response = backend.call(content, SEED + i)
            parsed, mode = parse_prediction(response['raw'], len(task['test']))
            response.update({'candidate_index': i, 'outputs': parsed, 'valid': parsed is not None, 'parse_mode': mode})
            if proxy: response['correct'] = parsed == expected
            candidates.append(response)
        atomic_json(path, {'identity': identity, 'manifest_sha256': manifest['sha256'], 'task_id': task_id,
                           'status': 'complete', 'agent_source_sha256': source_sha, 'runner_sha256': runner_hash,
                           'sandbox_sha256': sandbox_hash, 'sandbox_image': IMAGE, 'agent_output': outcome,
                           'candidates': candidates, 'runtime_seconds': sum(c['seconds'] for c in candidates)})
    return [json.loads((root / label / (t + '.json')).read_text()) for t in ids]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', default='.')
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--model-path', required=True)
    args = parser.parse_args()
    root = Path(args.run_dir).resolve()
    manifest, tasks = prepare(Path(args.repo).resolve(), root, args.model_path)
    backend = Backend(args.model_path)
    run_tasks(backend, root, manifest, tasks, 'development/baseline', manifest['allocation']['development'], PROMPT, proxy=True)
    choices = sources(backend, root, manifest, tasks)
    selection = manifest['allocation']['selection']
    existing = [root / 'selection' / 'baseline' / (task_id + '.json') for task_id in selection]
    if all(path.is_file() for path in existing):
        base = [json.loads(path.read_text()) for path in existing]
        if any(row.get('manifest_sha256') != manifest['sha256'] or row.get('status') != 'complete' for row in base):
            raise ValueError('selection baseline checkpoint mismatch')
    else:
        base = run_agent(backend, root, manifest, tasks, 'selection/agent_baseline', selection, BASE, True)
    def metrics(rows):
        return (sum(any(c['correct'] for c in row['candidates']) for row in rows),
                sum(c['valid'] for row in rows for c in row['candidates']))
    results = []
    for i, source in enumerate(choices):
        try:
            results.append(run_agent(backend, root, manifest, tasks, f'selection/agent{i}', selection, source, True))
        except (RuntimeError, TimeoutError, ValueError) as error:
            atomic_json(root / 'agent' / f'candidate{i}_error.json', {'error': str(error), 'source_sha256': hashlib.sha256(source.encode()).hexdigest()})
            results.append(None)
    eligible = [(metrics(rows), -i, i) for i, rows in enumerate(results) if rows is not None]
    eligible.sort(reverse=True)
    winner = eligible[0][2] if eligible and eligible[0][0][0] > metrics(base)[0] else None
    chosen = BASE if winner is None else choices[winner]
    frozen = {'manifest_sha256': manifest['sha256'], 'baseline': metrics(base),
              'candidates': [metrics(r) if r is not None else None for r in results],
              'selected_index': winner, 'source': chosen, 'source_sha256': hashlib.sha256(chosen.encode()).hexdigest()}
    path = root / 'agent' / 'selection.json'
    if path.exists() and json.loads(path.read_text()) != json.loads(json.dumps(frozen)):
        raise ValueError('agent selection changed')
    if not path.exists(): atomic_json(path, frozen)
    run_agent(backend, root, manifest, tasks, 'evaluation/agent', manifest['allocation']['orders']['agent'], chosen, False)

if __name__ == '__main__': main()
