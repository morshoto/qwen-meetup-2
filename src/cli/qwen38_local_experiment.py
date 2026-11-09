"""Run the local Qwen3.8 prompt-evolution arm and matched baseline.

Execution reads training challenges only, never the comparison solutions file.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import re
from pathlib import Path

from src.arc.lt_live_backend import PROMPT
from src.arc.qwen38_local_runner import (Backend, MODEL_REVISION, SEED, atomic_json,
                                         prepare, proxy_task, run_tasks)


def generate_prompts(backend, root, manifest, tasks, development):
    path = root / 'prompt_proposals.json'
    if path.exists():
        obj = json.loads(path.read_text())
        if obj.get('manifest_sha256') != manifest['sha256']:
            raise ValueError('prompt proposal registration mismatch')
        if obj['candidates'] is None:
            raise ValueError('prompt generation failed; inspect saved response')
        return obj['candidates']
    feedback = []
    for row in development:
        task, expected = proxy_task(tasks[row['task_id']])
        feedback.append({'task_id': row['task_id'], 'task': task, 'expected': expected,
                         'attempts': [{'raw': c['raw'], 'correct': c['correct']} for c in row['candidates']]})
    feedback = [f for f in feedback if not any(a['correct'] for a in f['attempts'])][:3]
    for item in feedback:
        for attempt in item['attempts']:
            attempt['raw'] = attempt['raw'][:800]
    while len(json.dumps(feedback)) > 20000 and feedback:
        feedback.pop()
    instruction = ('You are improving instructions for an ARC grid solver. Analyze the development examples and failed '
                   'responses below. Propose exactly two DISTINCT general instruction strings as a JSON array of two strings. '
                   'Each instruction must require one JSON array of output grids, one grid per test input, without markdown '
                   'or explanation. Do not include task-specific answers, examples, code, or references to hidden data. '
                   'Keep each instruction under 1500 characters. Original instruction: ' + PROMPT + '\nDevelopment feedback: ' +
                   json.dumps(feedback, separators=(',', ':')))
    raw_calls, candidates = [], None
    for attempt in range(2):
        query = instruction if attempt == 0 else ('Convert this prior response to ONLY a JSON array of exactly two '
                                                  'instruction strings, preserving the instruction content:\n' + raw_calls[0]['raw'])
        response = backend.call(query, SEED + 100 + attempt, max_tokens=1536)
        raw_calls.append({'request': query, **response})
        try:
            value = response['raw'].strip()
            if value.startswith('```') and value.endswith('```'):
                value = '\n'.join(value.splitlines()[1:-1])
            candidate = json.loads(value)
            if (not isinstance(candidate, list) or len(candidate) != 2 or
                any(not isinstance(p, str) or not 10 <= len(p) <= 1500 or re.search(r'```|\[\s*\[|\bdef\s+\w+\s*\(', p) for p in candidate) or candidate[0] == candidate[1]):
                raise ValueError('invalid candidate list')
            candidates = candidate
            break
        except (ValueError, TypeError):
            pass
    obj = {'manifest_sha256': manifest['sha256'], 'calls': raw_calls, 'candidates': candidates}
    atomic_json(path, obj)
    if candidates is None:
        raise ValueError('model failed to provide two valid prompt candidates; raw calls saved')
    return candidates


def select_prompt(root, manifest, originals, proposal_rows, candidates):
    def metrics(rows):
        return {'correct': sum(any(c['correct'] for c in r['candidates']) for r in rows),
                'valid': sum(c['valid'] for r in rows for c in r['candidates'])}
    base = metrics(originals)
    scores = [metrics(r) for r in proposal_rows]
    # Baseline wins a tie in correctness, as specified by the protocol.
    winner = max(range(2), key=lambda i: (scores[i]['correct'], scores[i]['valid'], -i))
    changed = scores[winner]['correct'] > base['correct']
    prompt = candidates[winner] if changed else PROMPT
    obj = {'manifest_sha256': manifest['sha256'], 'baseline': base, 'candidates': scores,
           'selected_index': winner if changed else None,
           'status': 'improved_selection' if changed else 'no_improvement_selected',
           'prompt': prompt, 'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest()}
    path = root / 'prompt_selection.json'
    if path.exists() and json.loads(path.read_text()) != obj:
        raise ValueError('selection differs from frozen selection')
    if not path.exists(): atomic_json(path, obj)
    return prompt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', default='.')
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--model-path', required=True)
    args = parser.parse_args()
    repo, root = Path(args.repo).resolve(), Path(args.run_dir).resolve()
    manifest, tasks = prepare(repo, root, args.model_path)
    backend = Backend(args.model_path)
    allocation = manifest['allocation']
    dev = run_tasks(backend, root, manifest, tasks, 'development/baseline', allocation['development'], PROMPT, proxy=True)
    chosen = None
    try:
        candidates = generate_prompts(backend, root, manifest, tasks, dev)
        original = run_tasks(backend, root, manifest, tasks, 'selection/baseline', allocation['selection'], PROMPT, proxy=True)
        proposals = [run_tasks(backend, root, manifest, tasks, f'selection/prompt{i}', allocation['selection'], p, proxy=True)
                     for i, p in enumerate(candidates)]
        chosen = select_prompt(root, manifest, original, proposals, candidates)
    except (ValueError, TypeError) as error:
        atomic_json(root / 'prompt_blocked.json', {'manifest_sha256': manifest['sha256'], 'error': str(error)})
        print('Prompt selection failed; continuing with local baseline:', error, flush=True)
    run_tasks(backend, root, manifest, tasks, 'evaluation/baseline', allocation['evaluation'], PROMPT)
    if chosen is not None:
        run_tasks(backend, root, manifest, tasks, 'evaluation/prompt', allocation['orders']['prompt'], chosen)
        print('Baseline and prompt arm completed; score offline.', flush=True)
    else:
        print('Baseline completed; prompt arm blocked, see prompt_blocked.json.', flush=True)



if __name__ == '__main__': main()
