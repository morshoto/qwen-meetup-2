import hashlib
import json
import unittest
from unittest.mock import patch

from src.arc.lt_batch_journal import _create
from src.arc.lt_phase_b_canary_sample import RELOAD_SETTINGS
from src.arc.lt_phase_b_registered_worker import _digest
from src.arc.lt_phase_b_real_reload_dispatch import run_real_reload_dispatch, review_real_reload_attempt
from tests.arc import test_lt_phase_b_training_dispatch as fixtures
from tests.arc.test_rsi_process_monitor import state


class RealReloadDispatchTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.TrainingDispatchTests('test_one_shot_train_records_reload_pending_and_replays')
        self.fixture.setUp(); self.addCleanup(self.fixture.doCleanups)
        f = self.fixture; c = f.fixture
        f.config.write_text(json.dumps({'model': str(c.model), 'adapter_path': str(c.run / 'gen1-adapter')}))
        f.config.chmod(0o600)
        f.spec['config_sha256'] = hashlib.sha256(f.config.read_bytes()).hexdigest()
        f.training_sha = _digest(f.spec)
        (c.run / 'training-spec.json').write_text(json.dumps(f.spec))
        f.preflight_value['training_sha256'] = f.training_sha
        f.mock['preflight_training_data'].return_value = f.preflight_value
        self.summary = f.invoke()
        (c.run / 'gen1-adapter').mkdir()
        patches = {'preflight_training_data': f.preflight_value,
            'validate_canary_identity': (c.spec['settings'], c.spec['model_manifest']),
            'review_reload_attempt': {'status': 'canary_reload_observed_unverified', 'reload_verified': False,
                                      'promotion_allowed': False},
            'review_real_training_attempt': self.summary}
        self.mock = {}
        for name, value in patches.items():
            p = patch('src.arc.lt_phase_b_real_reload_dispatch.' + name, return_value=value)
            self.mock[name] = p.start(); self.addCleanup(p.stop)
        def worker(argv, *, on_start, **kwargs):
            self.assertNotIn(str(f.solution), argv); on_start(1234)
            result = {'status': 'real_reload_observed_unverified', 'reload_verified': False, 'promotion_allowed': False,
                'training_sha256': f.training_sha, 'canary_sha256': c.canary_sha,
                'collection_sha256': c.collection_sha, 'adapter_sha256': 'f'*64,
                'sample_prompt_sha256': RELOAD_SETTINGS['sample_prompt_sha256'], 'sample_raw': '[[[1]]]',
                'sample_format_valid': True, 'usage': {'prompt_tokens': 20, 'completion_tokens': 4,
                  'peak_memory_gb': 1., 'finish_reason': 'stop'}}
            _create(c.run / 'real-reload-result.json', result)
            return {'status': 'complete', 'leader_reaped': True, 'group_absent': True, 'baseline': state(),
                'observation': {'stop_reason': 'process_exited', 'exit_code': 0, 'elapsed_seconds': 1.,
                  'stream_complete': True, 'post_reap_group_absent': True, 'samples': [state()]}}
        p = patch('src.arc.lt_phase_b_real_reload_dispatch.run_monitored_worker', side_effect=worker)
        self.worker = p.start(); self.addCleanup(p.stop)

    def invoke(self):
        f = self.fixture; c = f.fixture
        return run_real_reload_dispatch(source_root=c.fixture.root, run_dir=c.run,
            expected_training_sha256=f.training_sha, expected_canary_sha256=c.canary_sha,
            expected_collection_sha256=c.collection_sha, challenge_path=c.fixture.challenge,
            solution_path=f.solution, official_archive=c.fixture.archive,
            python_executable='/fixture/python', model_path=c.model)

    def test_records_one_shot_unverified_reload(self):
        result = self.invoke()
        self.assertEqual(result['status'], 'real_adapter_reload_observed_unverified')
        self.assertFalse(result['reload_verified']); self.assertFalse(result['promotion_allowed'])
        self.assertEqual(self.mock['preflight_training_data'].call_count, 2)
        self.assertTrue((self.fixture.fixture.run / 'real-reload-summary.json').exists())
        with self.assertRaises(ValueError): self.invoke()
        self.worker.assert_called_once()

    def test_replay_rejects_missing_or_changed_evidence(self):
        result = self.invoke()
        f = self.fixture; c = f.fixture
        def replay():
            return review_real_reload_attempt(c.run, training=f.spec, canary=c.canary,
                expected_training_sha256=f.training_sha, expected_canary_sha256=c.canary_sha,
                expected_collection_sha256=c.collection_sha)
        self.assertEqual(replay(), result)
        for name, field, value in (('real-reload-started.json', 'training_sha256', '0'*64),
                                   ('real-reload-process.json', 'pid', 0),
                                   ('real-reload-observation.json', 'group_absent', False),
                                   ('real-reload-result.json', 'sample_format_valid', False),
                                   ('real-reload-summary.json', 'reload_verified', True)):
            path = c.run / name
            original = path.read_text(); changed = json.loads(original); changed[field] = value
            path.write_text(json.dumps(changed))
            with self.subTest(name=name), self.assertRaises(ValueError): replay()
            path.write_text(original)
        (c.run / 'real-reload-summary.json').unlink()
        with self.assertRaises(ValueError): replay()

    def test_replay_detects_changed_training_and_canary(self):
        self.invoke()
        f = self.fixture; c = f.fixture
        self.mock['review_real_training_attempt'].return_value = {**self.summary, 'yield_status': 'overfitting_only'}
        with self.assertRaises(ValueError):
            review_real_reload_attempt(c.run, training=f.spec, canary=c.canary,
                expected_training_sha256=f.training_sha, expected_canary_sha256=c.canary_sha,
                expected_collection_sha256=c.collection_sha)

    def test_failed_preflight_does_not_start(self):
        self.mock['preflight_training_data'].side_effect = ValueError('mismatch')
        with self.assertRaises(ValueError): self.invoke()
        self.worker.assert_not_called()

    def test_failed_process_is_retained_without_retry(self):
        def fail(argv, *, on_start, **kwargs):
            on_start(1234)
            return {'status': 'error', 'observation': {'stop_reason': 'memory_pressure_limit'}}
        self.worker.side_effect = fail
        self.assertEqual(self.invoke()['status'], 'real_reload_failed')
        self.assertTrue((self.fixture.fixture.run / 'real-reload-observation.json').exists())
        with self.assertRaises(ValueError): self.invoke()

    def test_changed_configuration_fails_before_start(self):
        self.fixture.config.write_text('{}')
        with self.assertRaises(ValueError): self.invoke()
        self.worker.assert_not_called()

    def test_changed_inputs_after_reload_block_summary(self):
        original = self.worker.side_effect
        def change(*args, **kwargs):
            result = original(*args, **kwargs)
            self.fixture.solution.write_text('{"changed": true}')
            return result
        self.worker.side_effect = change
        with self.assertRaises(ValueError): self.invoke()
        self.assertFalse((self.fixture.fixture.run / 'real-reload-summary.json').exists())
        with self.assertRaises(ValueError): self.invoke()

    def test_result_adapter_mismatch_rejected(self):
        original = self.worker.side_effect
        def change(*args, **kwargs):
            result = original(*args, **kwargs)
            path = self.fixture.fixture.run / 'real-reload-result.json'
            value = json.loads(path.read_text()); value['adapter_sha256'] = '0'*64
            path.write_text(json.dumps(value))
            return result
        self.worker.side_effect = change
        with self.assertRaises(ValueError): self.invoke()
        self.assertFalse((self.fixture.fixture.run / 'real-reload-summary.json').exists())


if __name__ == '__main__': unittest.main()
