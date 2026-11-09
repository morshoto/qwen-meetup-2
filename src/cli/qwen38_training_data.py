"""Collect only model-generated, verified development proxy responses for LoRA."""
import argparse
import hashlib
import json
from pathlib import Path
from src.arc.qwen38_local_runner import atomic_json, proxy_task, parse_prediction
from src.arc.lt_live_backend import PROMPT


def collect(repo, root):
    repo, root = Path(repo), Path(root)
    manifest = json.loads((root / 'manifest.json').read_text())
    raw = (repo / 'data/raw/arc-agi_training_challenges.json').read_bytes()
    if hashlib.sha256(raw).hexdigest() != manifest['challenge_sha256']:
        raise ValueError('challenge hash mismatch')
    tasks = json.loads(raw)
    ids = manifest['allocation']['development']
    splits = {'train': ids[:8], 'valid': ids[8:10]}
    result = {'train': [], 'valid': []}
    for split, split_ids in splits.items():
        for task_id in split_ids:
            path = root / 'development/baseline' / (task_id + '.json')
            if not path.exists():
                raise ValueError(f'missing development checkpoint {task_id}')
            row = json.loads(path.read_text())
            if row['manifest_sha256'] != manifest['sha256'] or row['task_id'] != task_id:
                raise ValueError('checkpoint identity mismatch')
            task, expected = proxy_task(tasks[task_id])
            content = json.dumps({'instruction': PROMPT, 'task': task}, sort_keys=True, separators=(',', ':'))
            records = [(path, candidate) for candidate in row['candidates']]
            for extra_path in sorted((root / 'training' / 'generated' / task_id).glob('*.json')):
                extra = json.loads(extra_path.read_text())
                registration_path = root / 'training' / 'generation_registration.json'
                if not registration_path.exists():
                    raise ValueError('missing training generation registration')
                registration = json.loads(registration_path.read_text())
                if (registration['manifest_sha256'] != manifest['sha256'] or
                    extra.get('manifest_sha256') != manifest['sha256'] or extra.get('task_id') != task_id or
                    extra.get('generator_sha256') != registration['generator_sha256'] or
                    extra.get('seed') != registration['seed_base'] + int(extra_path.stem)):
                    raise ValueError('extra generation identity mismatch')
                records.append((extra_path, extra['candidate']))
            seen_outputs = set()
            for source_path, candidate in records:
                parsed, _ = parse_prediction(candidate['raw'], 1)
                if parsed != candidate['outputs'] or (parsed is not None) != candidate['valid']:
                    raise ValueError('candidate parse differs from recorded output')
                key = json.dumps(parsed, sort_keys=True, separators=(',', ':'))
                if parsed == expected and candidate['valid'] and key not in seen_outputs:
                    seen_outputs.add(key)
                    result[split].append({'task_id': task_id, 'candidate_index': candidate['candidate_index'],
                                          'messages': [{'role': 'user', 'content': content},
                                                       {'role': 'assistant', 'content': candidate['raw']}],
                                          'checkpoint_sha256': hashlib.sha256(source_path.read_bytes()).hexdigest()})
    qualified = (len(result['train']) >= 5 and len({r['task_id'] for r in result['train']}) >= 2
                 and len(result['valid']) >= 1)
    return {'manifest_sha256': manifest['sha256'], 'splits': result,
            'status': 'eligible_for_canary' if qualified else 'insufficient_verified_data',
            'train_count': len(result['train']), 'train_parents': len({r['task_id'] for r in result['train']}),
            'valid_count': len(result['valid']),
            'duplicate_train_responses_across_tasks': len(result['train']) - len({r['messages'][1]['content'] for r in result['train']})}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', default='.')
    parser.add_argument('--run-dir', required=True)
    args = parser.parse_args()
    report = collect(args.repo, args.run_dir)
    path = Path(args.run_dir) / 'training' / 'verified_data.json'
    atomic_json(path, report)
    print(json.dumps({k:v for k,v in report.items() if k != 'splits'}, indent=2))

if __name__ == '__main__': main()
