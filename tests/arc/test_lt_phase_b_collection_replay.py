import json
import unittest

from src.arc.lt_batch_journal import BatchJournal, _create
from src.arc.lt_phase_b_collection import collect_batch
from src.arc.lt_phase_b_collection_replay import replay_collection_dataset, reconstruct_collection_dataset
from tests.arc import test_lt_phase_b_sft_materialize as fixtures
from tests.arc import test_lt_phase_b_dataset as examples
from tests.arc.test_rsi_process_monitor import state


class CollectionReplayTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.SFTMaterializeTests('test_writes_private_exact_jsonl_once')
        self.fixture.setUp(); self.addCleanup(self.fixture.doCleanups)
        self.manifest = self.fixture.materialize()
        self.f = self.fixture.fixture.fixture
        self.source = self.f.source
        self.root = self.fixture.run / 'collection'
        self.journal = BatchJournal(self.root, sorted(self.f.tasks), self.source['run_sha256'] + ':14b:collection')
        self.limits = {'timeout_seconds': 300, 'minimum_disk_gib': 20, 'minimum_free_memory_percent': 20,
                       'maximum_swap_growth_mib': 512, 'maximum_process_rss_gib': 32, 'sample_interval_seconds': 5}
        rows = {'a': [self.f.good, self.f.good], 'b': [self.f.bad, self.f.good],
                'c': [self.f.good, self.f.bad], 'd': [self.f.bad, None]}
        def worker(task, output, on_start):
            on_start(1234)
            _create(output, examples.row(task, rows[task]))
            return {'status': 'complete', 'leader_reaped': True, 'group_absent': True, 'returncode': 0,
                'elapsed_seconds': 1., 'stdout': '', 'stderr': None, 'baseline': state(),
                'observation': {'stop_reason': 'process_exited', 'exit_code': 0, 'elapsed_seconds': 1.,
                  'stream_complete': True, 'cleanup_observation': {'group_cleanup_verified': False},
                  'post_reap_group_absent': True, 'samples': [state()]}}
        collect_batch(self.journal, limit=4, worker=worker, resource_limits=self.limits)

    def replay(self, **kwargs):
        args = {'journal_dir': self.root, 'tasks': self.f.tasks, 'comparison_tasks': {'x': examples.task('x')},
                'verifier': self.f.verifier, 'source': self.source, 'manifest': self.manifest,
                'resource_limits': self.limits}
        args.update(kwargs)
        return replay_collection_dataset(**args)

    def test_rebuilds_verified_split_and_matches_materialized_hashes(self):
        result = self.replay()
        self.assertEqual(result['status'], 'collection_dataset_consistent_unverified')
        self.assertEqual(result['attempted_tasks'], 4)
        self.assertEqual(result['eligible_tasks'], 3)
        self.assertFalse(result['training_allowed'])
        self.assertNotIn('journal', result)

    def test_reconstructs_private_dataset_before_materialization(self):
        result = reconstruct_collection_dataset(journal_dir=self.root, tasks=self.f.tasks,
            comparison_tasks={'x': examples.task('x')}, verifier=self.f.verifier,
            source=self.source, resource_limits=self.limits)
        self.assertEqual(result['yield_status'], 'disjoint_split')
        self.assertEqual(result['eligible_task_count'], 3)
        self.assertEqual(result['split_task_ids'], self.manifest['split_task_ids'])

    def test_zero_self_solutions_remains_insufficient(self):
        from src.arc.data import Verifier
        result = reconstruct_collection_dataset(journal_dir=self.root, tasks=self.f.tasks,
            comparison_tasks={'x': examples.task('x')}, verifier=Verifier({k: [[[9]]] for k in self.f.tasks}),
            source=self.source, resource_limits=self.limits)
        self.assertEqual(result['yield_status'], 'insufficient_verified_self_solutions')
        self.assertEqual(result['eligible_task_count'], 0)
        self.assertEqual(result['train'], [])

    def test_missing_and_modified_evidence_fail(self):
        for name, change in (('a.process.json', {'pid': 0}), ('b.result.json', {'candidates': []}),
                             ('c.resources.json', {'group_absent': False}), ('d.started.json', {'identity': 'other'})):
            path = self.root / name; old = path.read_text(); data = json.loads(old); data.update(change)
            path.write_text(json.dumps(data)); path.chmod(0o600)
            with self.subTest(name=name), self.assertRaises(ValueError): self.replay()
            path.write_text(old)
        (self.root / 'a.worker.json').unlink()
        with self.assertRaises(ValueError): self.replay()

    def test_resource_and_manifest_changes_fail(self):
        path = self.root / 'a.resources.json'; value = json.loads(path.read_text())
        value['observation']['samples'][0]['memory_free_percent'] = 5
        path.write_text(json.dumps(value)); path.chmod(0o600)
        with self.assertRaises(ValueError): self.replay()
        path.unlink()
        self.assertRaises(ValueError, self.replay)

    def test_invalid_manifest_counts_are_rejected(self):
        with self.assertRaises(ValueError): self.replay(manifest={**self.manifest, 'valid_count': True})
        with self.assertRaises(ValueError): self.replay(manifest={**self.manifest, 'train_jsonl_sha256': '0' * 64})

    def test_incorrect_verifier_and_extra_files_fail(self):
        from src.arc.data import Verifier
        with self.assertRaises(ValueError): self.replay(verifier=Verifier({k: [[[9]]] for k in self.f.tasks}))
        _create(self.root / 'unknown.json', {})
        with self.assertRaises(ValueError): self.replay()


if __name__ == '__main__': unittest.main()
