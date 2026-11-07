import hashlib
import json
import unittest
from unittest.mock import patch
from pathlib import Path

from src.arc.lt_batch_journal import _create
from src.arc.lt_live_backend import PROMPT
from src.arc.lt_phase_b_comparison_dispatch import run_comparison
from src.arc.lt_phase_b_registered_worker import _digest
from src.arc.lt_public import build_public_export
from tests.arc import test_lt_phase_b_training_dispatch as fixtures
from tests.arc.test_rsi_process_monitor import state


class ComparisonDispatchTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.TrainingDispatchTests('test_one_shot_train_records_reload_pending_and_replays')
        self.fixture.setUp(); self.addCleanup(self.fixture.doCleanups)
        f = self.fixture; c = f.fixture; self.run = c.run
        self.challenges = {f'{i + 100:08x}': {'train': [{'input': [[i % 10]], 'output': [[1]]}],
                                                       'test': [{'input': [[2]]}]} for i in range(50)}
        self.tasks = {k: _digest(v) for k, v in self.challenges.items()}
        allocation = json.loads((self.run / 'allocation.json').read_text())
        allocation['comparison_task_sha256'] = self.tasks
        (self.run / 'allocation.json').write_text(json.dumps(allocation))
        c.fixture.challenge.write_text(json.dumps(self.challenges))
        self.spec = {'protocol': 'lt-phase-b-comparison-v1', 'status': 'frozen', 'run_id': c.spec['run_id'],
            'collection_sha256': c.collection_sha, 'training_sha256': f.training_sha, 'canary_sha256': c.canary_sha,
            'source_code_sha256': c.spec['source_code_sha256'], 'prompt_sha256': hashlib.sha256(PROMPT.encode()).hexdigest(),
            'selected_task_sha256': self.tasks, 'adapter_sha256': 'f' * 64, 'settings': c.spec['settings']}
        self.sha = _digest(self.spec)
        self.receipt = {'comparison_sha256': self.sha, 'selected_task_sha256': self.tasks}
        self.export = build_public_export(self.challenges, self.receipt)
        for name, value in (('comparison-spec', self.spec), ('comparison-receipt', self.receipt), ('comparison-export', self.export)):
            (self.run / (name + '.json')).write_text(json.dumps(value))
        self.preflight = f.preflight_value
        patches = {'preflight_training_data': self.preflight,
            'validate_canary_identity': (c.spec['settings'], c.spec['model_manifest']),
            'review_real_reload_attempt': {'status': 'real_adapter_reload_observed_unverified', 'reload_verified': False,
                'promotion_allowed': False, 'reload': {'adapter_sha256': 'f' * 64}}}
        self.mock = {}
        for name, value in patches.items():
            p = patch('src.arc.lt_phase_b_comparison_dispatch.' + name, return_value=value)
            self.mock[name] = p.start(); self.addCleanup(p.stop)
        self.calls = []
        def worker(argv, *, on_start, **kwargs):
            self.assertNotIn(str(f.solution), argv)
            task = argv[argv.index('--task') + 1]; adapter_state = argv[argv.index('--adapter-state') + 1]
            output = argv[argv.index('--output') + 1]
            self.calls.append((task, adapter_state)); on_start(1234)
            prompt = json.dumps({'instruction': PROMPT, 'task': self.challenges[task]}, sort_keys=True, separators=(',', ':'))
            _create(Path(output), {'task_id': task, 'model': '14b', 'status': 'complete',
                'candidates': [[[[1]]], [[[1]]]], 'adapter_state': adapter_state,
                'adapter_sha256': 'f'*64 if adapter_state == 'on' else None, 'run_sha256': self.sha,
                'source_code_sha256': c.spec['source_code_sha256'], 'public_task_sha256': self.tasks[task],
                'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest()})
            return {'status': 'complete', 'leader_reaped': True, 'group_absent': True, 'returncode': 0,
                'elapsed_seconds': 1., 'stdout': '', 'stderr': None, 'baseline': state(),
                'observation': {'stop_reason': 'process_exited', 'exit_code': 0, 'elapsed_seconds': 1.,
                    'stream_complete': True, 'post_reap_group_absent': True,
                    'cleanup_observation': {'group_cleanup_verified': False}, 'samples': [state()]}}
        p = patch('src.arc.lt_phase_b_comparison_dispatch.run_monitored_worker', side_effect=worker)
        self.worker = p.start(); self.addCleanup(p.stop)

    def invoke(self, limit=1):
        f = self.fixture; c = f.fixture
        return run_comparison(source_root=c.fixture.root, run_dir=c.run, expected_comparison_sha256=self.sha,
            expected_training_sha256=f.training_sha, expected_canary_sha256=c.canary_sha,
            expected_collection_sha256=c.collection_sha, challenge_path=c.fixture.challenge,
            solution_path=f.solution, official_archive=c.fixture.archive, python_executable='/fixture/python',
            model_path=c.model, limit=limit)

    def test_dispatches_public_worker_and_resumes_off_first(self):
        self.assertEqual(self.invoke(), {'completed': 1, 'stop_reason': 'batch_limit', 'state': 'off'})
        self.assertEqual(self.invoke(), {'completed': 1, 'stop_reason': 'batch_limit', 'state': 'off'})
        self.assertEqual([s for _, s in self.calls], ['off', 'off'])
        self.assertEqual(self.mock['preflight_training_data'].call_count, 4)

    def test_full_off_batch_precedes_on_batch(self):
        self.assertEqual(self.invoke(50), {'completed': 50, 'stop_reason': 'state_complete', 'state': 'off'})
        self.assertEqual(self.invoke(1), {'completed': 1, 'stop_reason': 'batch_limit', 'state': 'on'})
        self.assertEqual([state for _, state in self.calls], ['off'] * 50 + ['on'])

    def test_bad_export_and_preflight_fail_before_launch(self):
        path = self.run / 'comparison-export.json'; value = json.loads(path.read_text())
        value['tasks'][next(iter(value['tasks']))]['test'][0]['output'] = [[1]]
        path.write_text(json.dumps(value))
        with self.assertRaises(ValueError): self.invoke()
        self.worker.assert_not_called()
        path.write_text(json.dumps(self.export))
        self.mock['preflight_training_data'].side_effect = ValueError('unfrozen')
        with self.assertRaises(ValueError): self.invoke()
        self.worker.assert_not_called()

    def test_changed_inputs_after_batch_block_success(self):
        original = self.worker.side_effect
        def changed(*args, **kwargs):
            result = original(*args, **kwargs)
            self.fixture.solution.write_text('{"changed": true}')
            return result
        self.worker.side_effect = changed
        with self.assertRaises(ValueError): self.invoke()
        self.assertTrue(list((self.run / 'comparison-off').glob('*.result.json')))
        self.fixture.solution.write_text('{}')
        self.worker.side_effect = original
        with self.assertRaises(ValueError): self.invoke()
        self.assertEqual(len(self.calls), 1)


if __name__ == '__main__': unittest.main()
