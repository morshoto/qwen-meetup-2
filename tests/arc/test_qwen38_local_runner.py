import json
import pytest
from src.arc.qwen38_local_runner import parse_prediction, proxy_task, run_tasks


def test_parse_recovery_and_proxy_excludes_held_answer():
    task, answer = proxy_task({'train': [{'input': [[1]], 'output': [[2]]}, {'input': [[3]], 'output': [[4]]}], 'test': [{'input': [[0]], 'output': [[8]]}]})
    assert task == {'train': [{'input': [[1]], 'output': [[2]]}], 'test': [{'input': [[3]]}]}
    assert answer == [[[4]]]
    assert parse_prediction('```json\n[[4]]\n```', 1) == ([[[4]]], 'remove_code_fence+wrap_single_grid')
    assert parse_prediction('wrong', 1) == (None, 'invalid')


def test_resume_skips_matching_checkpoint_and_rejects_changed_prompt(tmp_path):
    class Backend:
        count = 0
        def call(self, content, seed):
            self.count += 1
            return {'raw': '[[[4]]]', 'seconds': 0}
    b = Backend(); manifest = {'sha256': 'test'}
    tasks = {'a': {'train': [{'input': [[1]], 'output': [[2]]}, {'input': [[3]], 'output': [[4]]}], 'test': [{'input': [[0]]}]}}
    rows = run_tasks(b, tmp_path, manifest, tasks, 'dev', ['a'], 'p', proxy=True)
    assert b.count == 2 and rows[0]['candidates'][0]['correct']
    run_tasks(b, tmp_path, manifest, tasks, 'dev', ['a'], 'p', proxy=True)
    assert b.count == 2
    with pytest.raises(ValueError): run_tasks(b, tmp_path, manifest, tasks, 'dev', ['a'], 'changed', proxy=True)


def test_prompt_selection_falls_back_when_candidate_does_not_improve(tmp_path):
    from src.cli.qwen38_local_experiment import select_prompt
    from src.arc.lt_live_backend import PROMPT
    def row(correct, valid=True):
        return {'candidates': [{'correct': correct, 'valid': valid}, {'correct': False, 'valid': True}]}
    selected = select_prompt(tmp_path, {'sha256':'m'}, [row(True)], [[row(True)], [row(False)]], ['first candidate', 'second candidate'])
    assert selected == PROMPT
    assert json.loads((tmp_path / 'prompt_selection.json').read_text())['status'] == 'no_improvement_selected'


def test_training_examples_must_match_generated_raw_and_parent_split(tmp_path):
    import hashlib
    from src.cli.qwen38_training_data import collect
    (tmp_path / 'data/raw').mkdir(parents=True)
    ids = [f't{i}' for i in range(10)]
    tasks = {t: {'train': [{'input': [[1]], 'output': [[2]]}, {'input': [[3]], 'output': [[4]]}], 'test': [{'input': [[0]]}]} for t in ids}
    raw = json.dumps(tasks).encode()
    (tmp_path / 'data/raw/arc-agi_training_challenges.json').write_bytes(raw)
    run = tmp_path / 'run'; (run / 'development/baseline').mkdir(parents=True)
    manifest = {'sha256': 'm', 'challenge_sha256': hashlib.sha256(raw).hexdigest(), 'allocation': {'development': ids}}
    (run / 'manifest.json').write_text(json.dumps(manifest))
    for task_id in ids:
        candidate = {'raw': '[[[4]]]', 'outputs': [[[4]]], 'valid': True, 'candidate_index': 0}
        (run / 'development/baseline' / (task_id + '.json')).write_text(json.dumps({'manifest_sha256':'m','task_id':task_id,'candidates':[candidate]}))
    data = collect(tmp_path, run)
    assert data['status'] == 'eligible_for_canary' and data['train_count'] == 8 and data['valid_count'] == 2
    extra = run / 'training/generated/t0/00.json'; extra.parent.mkdir(parents=True)
    (run / 'training/generation_registration.json').write_text(json.dumps({'manifest_sha256':'m','generator_sha256':'g','seed_base':1}))
    extra.write_text(json.dumps({'manifest_sha256':'m','task_id':'t0','seed':1,'generator_sha256':'g','candidate':candidate}))
    assert collect(tmp_path, run)['train_count'] == 8  # same held-out answer is not a new example
    bad = json.loads(extra.read_text()); bad['seed'] = 99; extra.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match='identity mismatch'): collect(tmp_path, run)
    bad['seed'] = 1; extra.write_text(json.dumps(bad))
    path = run / 'development/baseline/t0.json'; row=json.loads(path.read_text()); row['candidates'][0]['raw']='[[[7]]]'; path.write_text(json.dumps(row))
    with pytest.raises(ValueError): collect(tmp_path, run)


def test_evaluation_task_removes_test_outputs():
    from src.arc.qwen38_local_runner import clean_task
    task = clean_task({'train':[{'input':[[1]],'output':[[2]]}], 'test':[{'input':[[3]],'output':[[9]]}]})
    assert task['test'] == [{'input': [[3]]}]
    assert '9' not in json.dumps(task)
