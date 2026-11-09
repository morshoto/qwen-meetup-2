import copy

import pytest

from src.arc.data import Verifier
from src.arc.qwen38_rsi_comparison import digest
from src.arc.qwen38_weight_paired import paired_report


def row(task_id, outputs, *, manifest='a' * 64, prompt='b' * 64, variant=None, label='evaluation/baseline'):
    import json
    candidates = []
    for index, output in enumerate(outputs):
        raw = json.dumps([output], separators=(',', ':')) if output is not None else 'invalid'
        candidates.append({'candidate_index': index, 'raw': raw, 'outputs': [output] if output is not None else None,
                           'valid': output is not None, 'parse_mode': 'strict' if output is not None else 'invalid'})
    return {'task_id': task_id, 'manifest_sha256': manifest, 'prompt_sha256': prompt, 'variant_sha256': variant,
            'identity': digest({'manifest': manifest, 'label': label, 'task_id': task_id, 'prompt_sha256': prompt,
                                'variant_sha256': variant}), 'status': 'complete', 'candidates': candidates,
            'runtime_seconds': 2.0}


def test_paired_report_distinguishes_gains_losses_and_incomplete():
    ids = ['a', 'b', 'c', 'd', 'e']; adapter = 'c' * 64
    verifier = Verifier({k: [[[1]]] for k in ids})
    base = [row('a', [[[1]], [[0]]]), row('b', [[[0]], [[0]]]), row('c', [[[1]], [[0]]]), row('d', [[[0]], [[0]]])]
    gen1 = [row('a', [[[0]], [[0]]], variant=adapter, label='evaluation/weights'),
            row('b', [[[1]], [[0]]], variant=adapter, label='evaluation/weights'),
            row('c', [[[1]], [[0]]], variant=adapter, label='evaluation/weights'),
            row('d', [[[0]], [None][0]], variant=adapter, label='evaluation/weights')]
    report = paired_report(ids, base, gen1, verifier, manifest_sha256='a' * 64, prompt_sha256='b' * 64, adapter_sha256=adapter)
    assert report['paired'] == {'complete': 4, 'gained': 1, 'lost': 1, 'both_correct': 1, 'both_incorrect': 1}
    assert report['baseline']['complete_tasks'] == report['gen1']['complete_tasks'] == 4
    assert report['baseline']['incomplete_tasks'] == report['gen1']['incomplete_tasks'] == 1
    assert report['gen1']['valid_candidates'] == 7 and report['gen1']['invalid_candidates'] == 1
    assert report['gen1']['first_correct'] == 2 and report['gen1']['either_correct'] == 2


def test_paired_report_rejects_changed_identity_duplicates_and_forged_parse():
    verifier = Verifier({'a': [[[1]]]}); base = row('a', [[[1]], [[0]]]); adapter = 'c' * 64
    other = row('a', [[[1]], [[0]]], variant=adapter, label='evaluation/weights')
    for change in ('identity', 'duplicate', 'parse', 'adapter'):
        b = [copy.deepcopy(base)]; g = [copy.deepcopy(other)]
        if change == 'identity': b[0]['identity'] = 'bad'
        elif change == 'duplicate': b.append(copy.deepcopy(b[0]))
        elif change == 'parse': g[0]['candidates'][0]['outputs'] = [[[9]]]
        else: g[0]['variant_sha256'] = None
        with pytest.raises(ValueError):
            paired_report(['a'], b, g, verifier, manifest_sha256='a' * 64, prompt_sha256='b' * 64, adapter_sha256=adapter)
