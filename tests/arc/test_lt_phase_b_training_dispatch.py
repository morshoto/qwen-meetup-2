import hashlib
import json
import unittest
from unittest.mock import patch

from src.arc.lt_batch_journal import _create
from src.arc.lt_phase_b_registered_worker import _digest
from src.arc.lt_phase_b_tensor_step import observe_training
from src.arc.lt_phase_b_training_dispatch import run_training, review_real_training_attempt
from tests.arc import test_lt_phase_b_canary_dispatch as fixtures
from tests.arc.test_lt_phase_b_tensor_evidence import Model
from tests.arc.test_rsi_process_monitor import state


class TrainingDispatchTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.CanaryDispatchTests('test_records_one_shot_observation_and_reload_pending')
        self.fixture.setUp(); self.addCleanup(self.fixture.doCleanups)
        f = self.fixture
        self.evidence = f.run / 'training-evidence'; self.evidence.mkdir(mode=0o700)
        self.config = f.run / 'training-config.json'; self.config.write_text('{}'); self.config.chmod(0o600)
        self.solution = f.fixture.root / 'solutions.json'; self.solution.write_text('{}')
        self.spec = {'protocol': 'lt-phase-b-real-training-v1', 'status': 'frozen', 'run_id': f.spec['run_id'],
            'collection_sha256': f.collection_sha, 'canary_sha256': f.canary_sha, 'source_code_sha256': 'a' * 64,
            'dataset_manifest_sha256': 'd' * 64, 'config_sha256': hashlib.sha256(self.config.read_bytes()).hexdigest(),
            'solution_sha256': hashlib.sha256(self.solution.read_bytes()).hexdigest(), 'yield_status': 'disjoint_split',
            'checkpoint_policy': 'final_step_only', 'timeout_seconds': 300,
            'resource_limits': f.spec['settings']['resource_limits']}
        self.training_sha = _digest(self.spec)
        (f.run / 'training-spec.json').write_text(json.dumps(self.spec))
        self.preflight_value = {'status': 'training_data_consistent_unverified', 'training_allowed': False,
            'training_sha256': self.training_sha, 'solution_sha256': self.spec['solution_sha256'],
            'dataset_manifest_sha256': self.spec['dataset_manifest_sha256'], 'attempted_tasks': 100,
            'eligible_tasks': 3, 'yield_status': 'disjoint_split'}
        patches = {'preflight_training_data': self.preflight_value,
            'validate_canary_identity': (f.spec['settings'], f.spec['model_manifest']),
            'review_reload_attempt': {'status': 'canary_reload_observed_unverified', 'reload_verified': False,
                                      'promotion_allowed': False},
            'inspect_adapter_artifact': {'status': 'adapter_files_consistent_unverified', 'adapter_sha256': 'f'*64,
                'weights_sha256': 'e'*64, 'config_sha256': 'd'*64, 'matched_tensor_count': 2,
                'reload_verified': False, 'promotion_allowed': False}}
        self.mock = {}
        for name, value in patches.items():
            p = patch('src.arc.lt_phase_b_training_dispatch.' + name, return_value=value)
            self.mock[name] = p.start(); self.addCleanup(p.stop)
        log = ''.join(f'Iter {i}: Train loss 0.125, x\n' for i in range(1, 21))
        def worker(argv, *, on_start, training, expected_training_steps, **kwargs):
            self.assertTrue(training); self.assertEqual(expected_training_steps, 20)
            self.assertNotIn(str(self.solution), argv)
            on_start(1234)
            model = Model()
            evidence = observe_training(model, output_dir=self.evidence,
                run_step=lambda: model.params['layer.lora_b'].__setitem__((0, 0), .25),
                flatten=lambda x: list(x.items()), evaluate=lambda _: None)
            _create(f.run / 'training-result.json', {**evidence, 'training_sha256': self.training_sha,
                'collection_sha256': f.collection_sha, 'canary_sha256': f.canary_sha,
                'training_run_id': f.spec['run_id'], 'yield_status': 'disjoint_split', 'promotion_allowed': False})
            return {'status': 'complete', 'leader_reaped': True, 'group_absent': True, 'returncode': 0,
                'baseline': state(), 'observation': {'stop_reason': 'process_exited', 'exit_code': 0,
                'elapsed_seconds': 1., 'stream_complete': True, 'post_reap_group_absent': True, 'samples': [state()],
                'log': log, 'log_sha256': hashlib.sha256(log.encode()).hexdigest(),
                'loss': {'status': 'ok', 'last_step': 20, 'last_loss': .125}}}
        p = patch('src.arc.lt_phase_b_training_dispatch.run_monitored_worker', side_effect=worker)
        self.mock['worker'] = p.start(); self.addCleanup(p.stop)

    def invoke(self):
        f = self.fixture
        return run_training(source_root=f.fixture.root, run_dir=f.run, expected_training_sha256=self.training_sha,
            expected_collection_sha256=f.collection_sha, expected_canary_sha256=f.canary_sha,
            challenge_path=f.fixture.challenge, solution_path=self.solution, official_archive=f.fixture.archive,
            python_executable='/fixture/python', model_path=f.model)

    def test_one_shot_train_records_reload_pending_and_replays(self):
        result = self.invoke()
        self.assertEqual(result['status'], 'real_training_observed_reload_pending')
        self.assertFalse(result['training_verified']); self.assertFalse(result['promotion_allowed'])
        self.assertEqual(self.mock['preflight_training_data'].call_count, 2)
        self.assertEqual(review_real_training_attempt(self.fixture.run, training=self.spec,
            expected_training_sha256=self.training_sha), result)
        with self.assertRaises(ValueError): self.invoke()
        self.mock['worker'].assert_called_once()

    def test_preflight_or_canary_failure_does_not_start(self):
        self.mock['preflight_training_data'].side_effect = ValueError('fixture')
        with self.assertRaises(ValueError): self.invoke()
        self.mock['worker'].assert_not_called()
        self.assertFalse((self.fixture.run / 'training-started.json').exists())

    def test_worker_failure_retained_and_no_retry(self):
        def fail(argv, *, on_start, **kwargs):
            on_start(1234)
            return {'status': 'error', 'observation': {'stop_reason': 'memory_pressure_limit'}}
        self.mock['worker'].side_effect = fail
        self.assertEqual(self.invoke()['status'], 'training_failed')
        self.assertTrue((self.fixture.run / 'training-observation.json').exists())
        with self.assertRaises(ValueError): self.invoke()

    def test_log_tampering_rejected(self):
        self.invoke()
        path = self.fixture.run / 'training-observation.json'
        data = json.loads(path.read_text()); data['observation']['log_sha256'] = '0' * 64
        path.write_text(json.dumps(data))
        with self.assertRaises(ValueError):
            review_real_training_attempt(self.fixture.run, training=self.spec, expected_training_sha256=self.training_sha)


if __name__ == '__main__': unittest.main()
