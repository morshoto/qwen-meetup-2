"""Score completed exploratory checkpoints offline without exposing answers."""
from __future__ import annotations
import argparse
import hashlib
import json
import random
from pathlib import Path
from src.arc.data import load_arc
from src.arc.qwen38_rsi_comparison import summarize, compare, digest
from src.arc.qwen38_local_runner import parse_prediction


def report(repo, root):
    repo, root = Path(repo), Path(root)
    manifest = json.loads((root / 'manifest.json').read_text())
    identity = manifest.get('sha256')
    if identity != digest({k:v for k,v in manifest.items() if k != 'sha256'}):
        raise ValueError('manifest digest mismatch')
    challenge = repo / 'data/raw/arc-agi_training_challenges.json'
    solutions = repo / 'data/raw/arc-agi_training_solutions.json'
    if hashlib.sha256(challenge.read_bytes()).hexdigest() != manifest['challenge_sha256']:
        raise ValueError('challenge hash mismatch')
    _, verifier = load_arc(challenge, solutions)
    ids = manifest['allocation']['evaluation']
    all_results, result = {}, {}
    for arm in ('baseline', 'prompt', 'agent', 'weights'):
        rows = []
        for path in sorted((root / 'evaluation' / arm).glob('*.json')):
            row = json.loads(path.read_text())
            if row.get('manifest_sha256') != manifest['sha256']:
                raise ValueError('checkpoint manifest mismatch')
            if row.get('status') != 'complete' or len(row.get('candidates', [])) != 2:
                raise ValueError('checkpoint is not a complete two-candidate result')
            for candidate in row['candidates']:
                parsed, mode = parse_prediction(candidate['raw'], verifier.output_count(row['task_id']))
                if parsed != candidate['outputs'] or (parsed is not None) != candidate['valid'] or mode != candidate['parse_mode']:
                    raise ValueError('checkpoint output does not match raw response')
            label = 'evaluation/' + arm
            if arm == 'agent':
                expected_identity = digest({'manifest': identity, 'label': label, 'task_id': row['task_id'],
                                            'source_sha': row['agent_source_sha256'], 'runner_sha': row['runner_sha256'],
                                            'sandbox_sha': row['sandbox_sha256'], 'image': row['sandbox_image']})
            else:
                expected_identity = digest({'manifest': identity, 'label': label, 'task_id': row['task_id'],
                                            'prompt_sha256': row['prompt_sha256'], 'variant_sha256': row.get('variant_sha256')})
            if row.get('identity') != expected_identity:
                raise ValueError('checkpoint identity mismatch')
            if arm == 'baseline' and row.get('prompt_sha256') != manifest['prompt_sha256']:
                raise ValueError('baseline prompt mismatch')
            if arm == 'prompt':
                selection = json.loads((root / 'prompt_selection.json').read_text())
                if selection['manifest_sha256'] != identity or row.get('prompt_sha256') != selection['prompt_sha256']:
                    raise ValueError('prompt selection mismatch')
            if arm == 'agent':
                selection = json.loads((root / 'agent/selection.json').read_text())
                if selection['manifest_sha256'] != identity or row.get('agent_source_sha256') != selection['source_sha256']:
                    raise ValueError('agent selection mismatch')
            if arm == 'weights':
                verification = json.loads((root / 'training/adapter/verification.json').read_text())
                if row.get('prompt_sha256') != manifest['prompt_sha256'] or row.get('variant_sha256') != verification['adapter_sha256']:
                    raise ValueError('adapter identity mismatch')
            rows.append(row)
        scored = summarize(ids, rows, verifier)
        all_results[arm] = scored
        blocked_path = root / (arm + '_blocked.json')
        blocked = json.loads(blocked_path.read_text()) if blocked_path.exists() else None
        result[arm] = {'status': 'complete' if len(scored) == len(ids) else ('incomplete' if scored else ('blocked' if blocked else 'not_run')),
                       'blocked_reason': blocked.get('error') if blocked else None,
                       'completed': len(scored), 'planned': len(ids),
                       'first_correct': sum(r['pass1'] for r in scored.values()) if scored else None,
                       'either_correct': sum(r['pass2'] for r in scored.values()) if scored else None,
                       'valid_candidates': sum(r['valid'] for r in scored.values()) if scored else None,
                       'candidate_count': len(scored) * 2,
                       'duplicate_raw_pairs': sum(row['candidates'][0].get('raw') == row['candidates'][1].get('raw') for row in rows),
                       'prompt_tokens': sum(c.get('prompt_tokens', 0) or 0 for row in rows for c in row['candidates']),
                       'completion_tokens': sum(c.get('completion_tokens', 0) or 0 for row in rows for c in row['candidates']),
                       'task_seconds': sum(r['seconds'] or 0 for r in scored.values())}
    prompt_selection = root / 'prompt_selection.json'
    if prompt_selection.exists():
        selection = json.loads(prompt_selection.read_text())
        result['prompt']['selection_status'] = selection['status']
        result['prompt']['selected_prompt_sha256'] = selection['prompt_sha256']
    agent_selection = root / 'agent' / 'selection.json'
    if agent_selection.exists():
        selection = json.loads(agent_selection.read_text())
        result['agent']['selection_status'] = ('no_improvement_selected' if selection['selected_index'] is None else 'improved_selection')
        result['agent']['selected_source_sha256'] = selection['source_sha256']
    for arm in ('prompt', 'agent', 'weights'):
        result[arm]['paired_vs_baseline'] = compare(all_results['baseline'], all_results[arm])
        paired = sorted(all_results['baseline'].keys() & all_results[arm].keys())
        if not paired:
            result[arm]['paired_vs_baseline']['gained'] = None
            result[arm]['paired_vs_baseline']['lost'] = None
        if paired:
            differences = [int(all_results[arm][t]['pass2']) - int(all_results['baseline'][t]['pass2']) for t in paired]
            rng = random.Random(20261111)
            samples = sorted(sum(rng.choice(differences) for _ in paired) / len(paired) for _ in range(2000))
            result[arm]['paired_vs_baseline']['either_difference_percentage_points'] = 100 * sum(differences) / len(paired)
            result[arm]['paired_vs_baseline']['bootstrap_95_percent_points'] = [100 * samples[49], 100 * samples[1949]]
    return {'manifest_sha256': manifest['sha256'], 'comparison_previously_evaluated': True,
            'scope': 'exploratory; not an official evaluation', 'arms': result}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', default='.')
    parser.add_argument('--run-dir', required=True)
    args = parser.parse_args()
    print(json.dumps(report(args.repo, args.run_dir), indent=2))

if __name__ == '__main__': main()
