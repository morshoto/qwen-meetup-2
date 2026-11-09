import hashlib
import json
from src.cli.qwen38_local_score import report
from src.arc.qwen38_rsi_comparison import digest


def test_score_reports_missing_arms_and_paired_partial_results(tmp_path):
    (tmp_path / 'data/raw').mkdir(parents=True)
    run = tmp_path / 'run'
    (run / 'evaluation/baseline').mkdir(parents=True)
    (run / 'evaluation/prompt').mkdir(parents=True)
    challenges = {'a': {'train': [{'input': [[0]], 'output': [[1]]}], 'test': [{'input': [[2]]}]},
                  'b': {'train': [{'input': [[0]], 'output': [[1]]}], 'test': [{'input': [[2]]}]}}
    raw = json.dumps(challenges).encode()
    (tmp_path / 'data/raw/arc-agi_training_challenges.json').write_bytes(raw)
    (tmp_path / 'data/raw/arc-agi_training_solutions.json').write_text(json.dumps({'a': [[[1]]], 'b': [[[2]]]}))
    manifest={'challenge_sha256': hashlib.sha256(raw).hexdigest(), 'prompt_sha256':'p', 'allocation': {'evaluation': ['a','b']}}
    manifest['sha256']=digest(manifest)
    (run / 'manifest.json').write_text(json.dumps(manifest))
    (run / 'prompt_selection.json').write_text(json.dumps({'manifest_sha256':manifest['sha256'], 'prompt_sha256':'p', 'status':'improved_selection'}))
    def save(arm, task, grids):
        row = {'task_id': task, 'status':'complete', 'manifest_sha256':manifest['sha256'], 'prompt_sha256':'p', 'runtime_seconds':1,
               'candidates':[{'outputs': g, 'raw': json.dumps(g), 'valid':g is not None, 'parse_mode':'strict' if g is not None else 'invalid', 'prompt_tokens':1, 'completion_tokens':2} for g in grids]}
        row['identity']=digest({'manifest':manifest['sha256'],'label':'evaluation/'+arm,'task_id':task,'prompt_sha256':'p','variant_sha256':None})
        (run / 'evaluation' / arm / (task + '.json')).write_text(json.dumps(row))
    save('baseline', 'a', [[[[0]]], None])
    save('baseline', 'b', [[[[2]]], None])
    save('prompt', 'a', [[[[1]]], None])
    r = report(tmp_path, run)
    assert r['arms']['baseline']['first_correct'] == 1
    assert r['arms']['prompt']['status'] == 'incomplete'
    assert r['arms']['prompt']['paired_vs_baseline']['paired'] == 1
    assert r['arms']['prompt']['paired_vs_baseline']['gained'] == 1
    assert r['arms']['weights']['status'] == 'not_run'
    assert r['arms']['weights']['first_correct'] is None
    path = run / 'evaluation/prompt/a.json'
    row = json.loads(path.read_text()); row['candidates'][0]['raw'] = '[[[9]]]'
    path.write_text(json.dumps(row))
    import pytest
    with pytest.raises(ValueError, match='raw response'):
        report(tmp_path, run)
