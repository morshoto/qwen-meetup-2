import json
import unittest
from unittest.mock import patch

from src.arc.lt_batch_journal import _create
from src.arc.lt_phase_b_canary_sample import RELOAD_SETTINGS
from src.arc.lt_phase_b_reload_dispatch import run_reload, review_reload_attempt
from tests.arc import test_lt_phase_b_canary_dispatch as fixtures
from tests.arc.test_rsi_process_monitor import state


class ReloadDispatchTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.CanaryDispatchTests('test_records_one_shot_observation_and_reload_pending')
        self.fixture.setUp(); self.addCleanup(self.fixture.doCleanups)
        self.training_summary = self.fixture.invoke()
        self.fixture.adapter.mkdir()
        self.preflight = patch('src.arc.lt_phase_b_reload_dispatch.preflight_collection', return_value={
            'status': 'preflight_consistent_unverified', 'execution_ready': False,
            'run_sha256': self.fixture.collection_sha, 'selected_tasks': 100,
            'evidence_type': 'local_files_and_caller_supplied_registration'})
        self.preflight.start(); self.addCleanup(self.preflight.stop)
        self.inspect = patch('src.arc.lt_phase_b_reload_dispatch.inspect_adapter_artifact',
                             return_value=self.training_summary['adapter'])
        self.inspect_mock = self.inspect.start(); self.addCleanup(self.inspect.stop)
        for name in ('verify_source_tree', 'verify_runtime_identity', 'verify_environment', 'verify_model_manifest'):
            p = patch('src.arc.lt_phase_b_reload_dispatch.' + name)
            p.start(); self.addCleanup(p.stop)
        self.calls = []
        def worker(argv, *, on_start, training=False, **kwargs):
            self.calls.append(argv); self.assertFalse(training); on_start(1234)
            result = {'status': 'reload_observed_unverified', 'reload_verified': False, 'promotion_allowed': False,
                      'canary_sha256': self.fixture.canary_sha, 'collection_sha256': self.fixture.collection_sha,
                      'adapter_sha256': 'f' * 64, 'sample_prompt_sha256': RELOAD_SETTINGS['sample_prompt_sha256'],
                      'sample_raw': '[[[2]]]', 'sample_format_valid': True,
                      'usage': {'prompt_tokens': 20, 'completion_tokens': 4, 'peak_memory_gb': 1., 'finish_reason': 'stop'}}
            _create(self.fixture.run / 'canary-reload-result.json', result)
            return {'status': 'complete', 'leader_reaped': True, 'group_absent': True, 'returncode': 0,
                    'baseline': state(), 'elapsed_seconds': 1., 'stdout': '', 'stderr': None,
                    'observation': {'stop_reason': 'process_exited', 'exit_code': 0, 'elapsed_seconds': 1.,
                        'stream_complete': True, 'post_reap_group_absent': True, 'samples': [state()],
                        'cleanup_observation': {'group_cleanup_verified': False}, 'escaped_processes_excluded': False}}
        self.worker = patch('src.arc.lt_phase_b_reload_dispatch.run_monitored_worker', side_effect=worker)
        self.process = self.worker.start(); self.addCleanup(self.worker.stop)

    def invoke(self):
        f = self.fixture
        return run_reload(source_root=f.fixture.root, run_dir=f.run,
            expected_collection_sha256=f.collection_sha, expected_canary_sha256=f.canary_sha,
            challenge_path=f.fixture.challenge, official_archive=f.fixture.archive,
            python_executable='/fixture/python', model_path=f.model)

    def test_replays_training_and_records_reload_without_promotion(self):
        result = self.invoke()
        self.assertEqual(result['status'], 'canary_reload_observed_unverified')
        self.assertFalse(result['reload_verified'])
        self.assertFalse(result['promotion_allowed'])
        self.assertEqual(result['reload']['adapter_sha256'], 'f' * 64)
        self.assertTrue((self.fixture.run / 'canary-reload-observation.json').exists())
        self.assertTrue((self.fixture.run / 'canary-reload-summary.json').exists())
        with self.assertRaises(ValueError): self.invoke()
        self.assertEqual(len(self.calls), 1)

    def test_reload_replay_rechecks_persisted_evidence(self):
        expected = self.invoke()
        f = self.fixture
        def replay():
            return review_reload_attempt(f.run, canary=f.canary, expected_canary_sha256=f.canary_sha,
                expected_collection_sha256=f.collection_sha)
        self.assertEqual(replay(), expected)
        for name, field, value in (('canary-reload-started.json', 'canary_sha256', '0' * 64),
                                   ('canary-reload-process.json', 'pid', 0),
                                   ('canary-reload-observation.json', 'group_absent', False),
                                   ('canary-reload-result.json', 'sample_format_valid', False),
                                   ('canary-reload-summary.json', 'reload_verified', True)):
            path = f.run / name
            original = path.read_text()
            changed = json.loads(original); changed[field] = value
            path.write_text(json.dumps(changed))
            with self.subTest(name=name), self.assertRaises(ValueError): replay()
            path.write_text(original)

    def test_reload_replay_rejects_changed_training_and_adapter(self):
        self.invoke()
        f = self.fixture
        self.inspect_mock.return_value = dict(self.training_summary['adapter'], adapter_sha256='0' * 64)
        with self.assertRaises(ValueError):
            review_reload_attempt(f.run, canary=f.canary, expected_canary_sha256=f.canary_sha,
                                  expected_collection_sha256=f.collection_sha)

    def test_tampered_training_summary_prevents_reload(self):
        path = self.fixture.run / 'canary-training-summary.json'
        summary = json.loads(path.read_text()); summary['loss']['last_loss'] = 99
        path.write_text(json.dumps(summary))
        with self.assertRaises(ValueError): self.invoke()
        self.assertEqual(self.calls, [])
        self.assertFalse((self.fixture.run / 'canary-reload-started.json').exists())

    def test_failed_reload_is_recorded_and_not_retried(self):
        def fail(argv, *, on_start, **kwargs):
            on_start(1234)
            return {'status': 'error', 'leader_reaped': True, 'group_absent': True,
                    'baseline': state(), 'observation': {'stop_reason': 'memory_pressure_limit'}}
        self.process.side_effect = fail
        result = self.invoke()
        self.assertEqual(result['status'], 'reload_failed')
        self.assertFalse(result['reload_verified'])
        self.assertTrue((self.fixture.run / 'canary-reload-observation.json').exists())
        with self.assertRaises(ValueError): self.invoke()
        self.assertEqual(self.process.call_count, 1)


if __name__ == '__main__': unittest.main()
