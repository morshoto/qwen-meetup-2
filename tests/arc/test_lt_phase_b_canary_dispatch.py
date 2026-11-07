import hashlib
import json
import unittest
from pathlib import Path
from unittest.mock import patch

from src.arc.lt_batch_journal import _create
from src.arc.lt_phase_b_canary_config import build_canary_config
from src.arc.lt_phase_b_canary_dispatch import run_canary
from src.arc.lt_phase_b_tensor_step import observe_one_step
from tests.arc import test_lt_phase_b_dispatch as fixtures
from tests.arc.test_lt_phase_b_tensor_evidence import Model
from tests.arc.test_rsi_process_monitor import state


class CanaryDispatchTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.DispatchTests('test_preflight_then_dispatch_public_only_worker')
        self.fixture.setUp(); self.addCleanup(self.fixture.doCleanups)
        self.run = self.fixture.run
        self.spec = self.fixture.fixture.spec
        self.collection_sha = self.fixture.sha
        self.model = self.fixture.root / 'a4d9b2df59d2c150bef02fcbe0d91046b7ca33a4'; self.model.mkdir()
        self.data = self.run / 'canary-data'; self.data.mkdir(mode=0o700)
        self.evidence = self.run / 'canary-evidence'; self.evidence.mkdir(mode=0o700)
        self.adapter = self.run / 'canary-adapter'
        plan = build_canary_config(model_dir=self.model, data_dir=self.data, adapter_dir=self.adapter)
        (self.data / 'train.jsonl').write_text(plan['dataset_jsonl']); (self.data / 'train.jsonl').chmod(0o600)
        self.config = self.run / 'canary-config.json'
        self.config.write_text(json.dumps(plan['trainer_config'])); self.config.chmod(0o600)
        self.canary = {'protocol': 'lt-phase-b-canary-v1', 'status': 'frozen', 'run_id': 'fixture-canary',
                       'collection_sha256': self.collection_sha, 'source_code_sha256': 'a' * 64,
                       'runtime_manifest': self.spec['runtime_manifest'], 'environment_manifest': self.spec['environment_manifest'],
                       'model_manifest': self.spec['model_manifest'], 'config_sha256': hashlib.sha256(self.config.read_bytes()).hexdigest(),
                       'dataset_sha256': plan['dataset_sha256'], 'thinking_mode': 'off', 'timeout_seconds': 300,
                       'resource_limits': self.spec['settings']['resource_limits']}
        self.canary_sha = fixtures.fixtures.sha(self.canary)
        (self.run / 'canary-spec.json').write_text(json.dumps(self.canary))
        self.preflight = patch('src.arc.lt_phase_b_canary_dispatch.preflight_collection', return_value={
            'status': 'preflight_consistent_unverified', 'execution_ready': False, 'run_sha256': self.collection_sha,
            'selected_tasks': 100, 'evidence_type': 'local_files_and_caller_supplied_registration'})
        self.preflight_check = self.preflight.start(); self.addCleanup(self.preflight.stop)
        self.adapter_check = patch('src.arc.lt_phase_b_canary_dispatch.inspect_adapter_artifact', return_value={
            'status': 'adapter_files_consistent_unverified', 'adapter_sha256': 'f' * 64,
            'weights_sha256': 'e' * 64, 'config_sha256': 'd' * 64,
            'reload_verified': False, 'promotion_allowed': False})
        self.adapter_check.start(); self.addCleanup(self.adapter_check.stop)
        self.shapes = patch('src.arc.lt_phase_b_tensor_evidence._expected_shapes',
                            return_value={'layer.lora_a': [2, 1], 'layer.lora_b': [1, 2]})
        self.shapes.start(); self.addCleanup(self.shapes.stop)
        self.calls = []
        def worker(argv, *, on_start, training, **kwargs):
            self.calls.append(argv); self.assertTrue(training); on_start(1234)
            model = Model()
            result = observe_one_step(model, output_dir=self.evidence,
                run_step=lambda: model.params['layer.lora_b'].__setitem__((0, 0), .25),
                flatten=lambda x: list(x.items()), evaluate=lambda _: None)
            _create(self.run / 'canary-result.json', {**result, 'canary_sha256': self.canary_sha,
                'collection_sha256': self.collection_sha, 'canary_run_id': 'fixture-canary', 'promotion_allowed': False})
            return {'status': 'complete', 'leader_reaped': True, 'group_absent': True, 'returncode': 0,
                    'baseline': state(), 'elapsed_seconds': 1., 'stdout': 'Iter 1: Train loss 0.125, x\n', 'stderr': None,
                    'observation': {'stop_reason': 'process_exited', 'exit_code': 0, 'elapsed_seconds': 1.,
                        'stream_complete': True, 'post_reap_group_absent': True, 'samples': [state()],
                        'log': 'Iter 1: Train loss 0.125, x\n',
                        'loss': {'status': 'ok', 'last_step': 1, 'last_loss': .125},
                        'cleanup_observation': {'group_cleanup_verified': False}, 'escaped_processes_excluded': False}}
        self.worker = patch('src.arc.lt_phase_b_canary_dispatch.run_monitored_worker', side_effect=worker)
        self.process = self.worker.start(); self.addCleanup(self.worker.stop)

    def invoke(self):
        return run_canary(source_root=self.fixture.root, run_dir=self.run,
            expected_collection_sha256=self.collection_sha, expected_canary_sha256=self.canary_sha,
            challenge_path=self.fixture.challenge, official_archive=self.fixture.archive,
            python_executable='/fixture/python', model_path=self.model)

    def test_records_one_shot_observation_and_reload_pending(self):
        result = self.invoke()
        self.assertEqual(result['status'], 'training_observed_reload_pending')
        self.assertFalse(result['training_verified'])
        self.assertFalse(result['promotion_allowed'])
        self.assertEqual(result['adapter']['adapter_sha256'], 'f' * 64)
        self.assertTrue((self.run / 'canary-started.json').exists())
        self.assertTrue((self.run / 'canary-process.json').exists())
        self.assertTrue((self.run / 'canary-observation.json').exists())
        self.assertTrue((self.run / 'canary-training-summary.json').exists())
        with self.assertRaises(ValueError): self.invoke()
        self.assertEqual(len(self.calls), 1)

    def test_false_success_with_mismatched_loss_never_promotes(self):
        original = self.process.side_effect
        def forged(*args, **kwargs):
            result = original(*args, **kwargs)
            result['observation']['loss']['last_loss'] = .5
            return result
        self.process.side_effect = forged
        with self.assertRaises(ValueError): self.invoke()
        self.assertTrue((self.run / 'canary-observation.json').exists())
        self.assertFalse((self.run / 'canary-training-summary.json').exists())
        with self.assertRaises(ValueError): self.invoke()

    def test_preflight_failure_never_starts(self):
        self.preflight_check.side_effect = ValueError('fixture mismatch')
        with self.assertRaises(ValueError): self.invoke()
        self.assertEqual(self.calls, [])
        self.assertFalse((self.run / 'canary-started.json').exists())

    def test_failed_worker_is_recorded_and_never_retried(self):
        def fail(argv, *, on_start, **kwargs):
            on_start(1234)
            return {'status': 'error', 'leader_reaped': True, 'group_absent': True,
                    'baseline': state(), 'observation': {'stop_reason': 'nonfinite_loss'}}
        self.process.side_effect = fail
        result = self.invoke()
        self.assertEqual(result['status'], 'training_failed')
        self.assertTrue((self.run / 'canary-observation.json').exists())
        self.assertFalse((self.run / 'canary-training-summary.json').exists())
        with self.assertRaises(ValueError): self.invoke()
        self.assertEqual(self.process.call_count, 1)


if __name__ == '__main__': unittest.main()
