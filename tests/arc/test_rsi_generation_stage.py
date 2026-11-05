import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from src.arc.data import Task
from src.arc.e6_curriculum import _quality_grids
from src.arc.e6_fake import FAKE_PROGRAMS, FakeE6Sandbox
from src.arc.rsi_collection_budget import CollectionBudget
from src.arc.rsi_generation_stage import collect_generation
from src.arc.rsi_lineage_audit import probes_sha256
from src.arc.rsi_reference_pool import build_reference_pool
from src.arc.rsi_slot_audit import manifest_sha256, task_sha256


class GenerationStageTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.ids = [f'task-{i:02}' for i in range(20)]
        self.tasks = {key: Task(key, (([[1, 2]], [[2, 1]]),), ([[3, 4]],)) for key in self.ids}
        self.hashes = {key: task_sha256(self.tasks[key]) for key in self.ids[:10]}
        self.public, self.private = _quality_grids(42, 5), _quality_grids(43, 5)
        self.budget = CollectionBudget.create(self.root, registration_sha256='a' * 64, generation=0)
        self.transport = Mock(return_value=FAKE_PROGRAMS[1])
        self.pool = []

    def run_stage(self, **changes):
        pool_hash = hashlib.sha256(json.dumps(self.pool, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        args = dict(root=self.root, registration_sha256='a' * 64, generation=0, parent_adapter='none',
            task_ids=self.ids, tasks=self.tasks, expected_manifest_sha256=manifest_sha256(self.ids),
            expected_task_hashes=self.hashes, public_probes=self.public, private_probes=self.private,
            expected_public_probes_sha256=probes_sha256(self.public), expected_private_probes_sha256=probes_sha256(self.private),
            reference_pool=self.pool, expected_reference_pool_sha256=pool_hash, transport=self.transport,
            sandbox=FakeE6Sandbox(), budget=self.budget)
        args.update(changes)
        return collect_generation(**args)

    def test_ten_slots_screen_against_previous_novel_slots(self):
        result = self.run_stage()
        self.assertEqual(result['status'], 'generation_complete_unverified')
        self.assertEqual(result['counts'], {'novel_unverified': 1, 'repeated_code': 9})
        self.assertEqual(self.transport.call_count, 19)
        self.assertEqual(self.budget.snapshot()['model_calls'], 19)
        self.assertEqual(result['budget_usage']['model_calls'], 19)
        self.assertGreater(result['budget_usage']['reserved_tokens'], 0)
        self.assertGreater(result['budget_usage']['sandbox_runs'], 0)
        self.assertEqual(self.pool, [])
        self.assertFalse(result['training_allowed'])
        self.assertTrue((self.root / 'g0.stage.completed.json').exists())
        start = json.loads((self.root / 'g0.stage.started.json').read_text())
        self.assertEqual(start['task_hashes'], self.hashes)
        self.assertEqual(start['public_probe_sha256'], probes_sha256(self.public))
        self.assertEqual(start['private_probe_sha256'], probes_sha256(self.private))
        with self.assertRaises(FileExistsError):
            self.run_stage()

    def test_historical_reference_is_duplicate_only(self):
        code = FAKE_PROGRAMS[1]
        entry = {'record_id': 'historical', 'source': code,
                 'source_sha256': hashlib.sha256(code.strip().encode()).hexdigest(),
                 'origin_sha256': 'b' * 64, 'ancestor_task_ids': [self.ids[0]]}
        reference = build_reference_pool(entries=[entry], private_probes=self.private,
            expected_probe_sha256=probes_sha256(self.private), generation_task_ids=set(self.ids[:10]),
            development_task_ids=set(self.ids[10:]), sandbox=FakeE6Sandbox(), max_sandbox_starts=10)
        self.pool = reference['pool']
        result = self.run_stage()
        self.assertEqual(result['counts'], {'repeated_code': 10})
        self.assertFalse(reference['eligible_for_training'])
        self.assertFalse(result['training_allowed'])

    def test_failure_is_incomplete_and_cannot_replay(self):
        self.transport.side_effect = [FAKE_PROGRAMS[1], FAKE_PROGRAMS[1], TimeoutError('uncertain')]
        result = self.run_stage()
        self.assertEqual(result['status'], 'generation_incomplete')
        self.assertEqual(result['completed_slots'], 1)
        self.assertEqual(result['budget_usage']['model_calls'], 3)
        self.assertGreater(result['budget_usage']['reserved_tokens'], 0)
        self.assertFalse(result['training_allowed'])
        self.assertTrue((self.root / 'g0.stage.incomplete.json').exists())
        self.assertFalse((self.root / 'g0.stage.completed.json').exists())
        with self.assertRaises(FileExistsError):
            self.run_stage()
        self.assertEqual(self.transport.call_count, 3)

    def test_mismatched_inputs_fail_before_dispatch(self):
        for changes in ({'expected_reference_pool_sha256': 'f' * 64},
                        {'expected_task_hashes': {}}, {'parent_adapter': 'f' * 64}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.run_stage(**changes)
        self.transport.assert_not_called()
        self.assertFalse((self.root / 'g0.stage.started.json').exists())
