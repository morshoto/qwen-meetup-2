"""Offline paired accounting for the existing exploratory comparison set.

Callers must authenticate the comparison manifest, base model and adapter first.
Missing or incomplete task rows never become incorrect scores.
"""
from __future__ import annotations

import math
import re

from src.arc.qwen38_local_runner import parse_prediction
from src.arc.qwen38_rsi_comparison import digest


def paired_report(ids, baseline_rows, gen1_rows, verifier, *, manifest_sha256, prompt_sha256, adapter_sha256):
    if (not ids or len(ids) != len(set(ids)) or
        any(not isinstance(x, str) or re.fullmatch('[0-9a-f]{64}', x) is None
            for x in (manifest_sha256, prompt_sha256, adapter_sha256))):
        raise ValueError('invalid paired comparison identity')

    def summarize(rows, label, variant):
        index = {}; scored = {}
        summary = {'requested_tasks': len(ids), 'complete_tasks': 0, 'incomplete_tasks': 0,
                   'missing_tasks': 0, 'failed_tasks': 0, 'partial_tasks': 0,
                   'first_correct': 0, 'either_correct': 0, 'valid_candidates': 0,
                   'invalid_candidates': 0, 'duplicate_raw_pairs': 0, 'runtime_seconds': 0.0}
        for row in rows:
            task_id = row.get('task_id')
            if task_id not in ids or task_id in index:
                raise ValueError('unknown or duplicate comparison task')
            identity = digest({'manifest': manifest_sha256, 'label': label, 'task_id': task_id,
                               'prompt_sha256': prompt_sha256, 'variant_sha256': variant})
            if (row.get('identity') != identity or row.get('manifest_sha256') != manifest_sha256 or
                row.get('prompt_sha256') != prompt_sha256 or row.get('variant_sha256') != variant):
                raise ValueError('comparison identity mismatch')
            index[task_id] = row
        for task_id in ids:
            row = index.get(task_id)
            if row is None:
                summary['missing_tasks'] += 1
                continue
            if row.get('status') != 'complete':
                summary['failed_tasks'] += 1
                continue
            candidates = row.get('candidates')
            if not isinstance(candidates, list) or len(candidates) != 2:
                summary['partial_tasks'] += 1
                continue
            elapsed = row.get('runtime_seconds')
            if isinstance(elapsed, bool) or not isinstance(elapsed, (int, float)) or not math.isfinite(elapsed) or elapsed < 0:
                raise ValueError('invalid comparison runtime')
            correct = []; raw_values = []
            for candidate_index, candidate in enumerate(candidates):
                raw = candidate.get('raw')
                if candidate.get('candidate_index') != candidate_index or not isinstance(raw, str):
                    raise ValueError('invalid comparison candidate')
                parsed, mode = parse_prediction(raw, verifier.output_count(task_id))
                if (parsed != candidate.get('outputs') or (parsed is not None) != candidate.get('valid') or
                    mode != candidate.get('parse_mode')):
                    raise ValueError('comparison candidate parse mismatch')
                summary['valid_candidates' if parsed is not None else 'invalid_candidates'] += 1
                correct.append(parsed is not None and verifier.matches(task_id, parsed))
                raw_values.append(raw)
            first, either = bool(correct[0]), any(correct)
            summary['complete_tasks'] += 1; summary['first_correct'] += first; summary['either_correct'] += either
            summary['duplicate_raw_pairs'] += raw_values[0] == raw_values[1]
            summary['runtime_seconds'] += elapsed
            scored[task_id] = {'first': first, 'either': either}
        summary['incomplete_tasks'] = len(ids) - summary['complete_tasks']
        return summary, scored

    baseline, base_scores = summarize(baseline_rows, 'evaluation/baseline', None)
    gen1, gen1_scores = summarize(gen1_rows, 'evaluation/weights', adapter_sha256)
    common = base_scores.keys() & gen1_scores.keys()
    pairs = {'complete': len(common), 'gained': 0, 'lost': 0, 'both_correct': 0, 'both_incorrect': 0}
    for task_id in common:
        before, after = base_scores[task_id]['either'], gen1_scores[task_id]['either']
        key = 'both_correct' if before and after else 'lost' if before else 'gained' if after else 'both_incorrect'
        pairs[key] += 1
    return {'baseline': baseline, 'gen1': gen1, 'paired': pairs,
            'adapter_sha256': adapter_sha256, 'comparison': 'exploratory previously evaluated tasks',
            'two_candidate_metric': 'either recorded candidate; independence not assumed'}
