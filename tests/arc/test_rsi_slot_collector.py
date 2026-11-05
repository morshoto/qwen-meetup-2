import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from src.arc.data import Task
from src.arc.e6_curriculum import _quality_grids
from src.arc.e6_fake import FAKE_PROGRAMS, FakeE6Sandbox
from src.arc.rsi_collection_budget import CollectionBudget
from src.arc.rsi_lineage_audit import probes_sha256
from src.arc.rsi_slot_audit import manifest_sha256, task_sha256
from src.arc.rsi_slot_collector import collect_slot
from src.arc.rsi_slot_lifecycle import observe_slot

REG = 'a' * 64


class SlotCollectorTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.ids = [f'task-{i:02}' for i in range(20)]
        self.task = Task(self.ids[0], (([[1, 2]], [[2, 1]]),), ([[3, 4]],))
        self.public, self.private = _quality_grids(42, 5), _quality_grids(43, 5)
        self.budget = CollectionBudget.create(self.root, registration_sha256=REG, generation=0)
        self.transport = Mock(side_effect=[FAKE_PROGRAMS[2], FAKE_PROGRAMS[1]])

    def collect(self, **overrides):
        args = dict(root=self.root, registration_sha256=REG, generation=0, slot=0,
                    task_ids=self.ids, task=self.task, expected_manifest_sha256=manifest_sha256(self.ids),
                    expected_task_sha256=task_sha256(self.task), public_probes=self.public,
                    private_probes=self.private, expected_public_probes_sha256=probes_sha256(self.public),
                    expected_private_probes_sha256=probes_sha256(self.private), reference_pool=[],
                    transport=self.transport, sandbox=FakeE6Sandbox(), budget=self.budget)
        args.update(overrides)
        return collect_slot(**args)

    def test_repair_is_scheduled_and_durable(self):
        result = self.collect()
        self.assertEqual(result['status'], 'novel_unverified')
        self.assertFalse(result['eligible_for_training'])
        self.assertEqual([call.kwargs['seed'] for call in self.transport.call_args_list], [12000, 12001])
        self.assertIn('visible_demonstration_mismatch', self.transport.call_args_list[1].args[0])
        self.assertNotIn(str(self.private[0]), repr(self.transport.call_args_list))
        self.assertEqual(self.budget.snapshot()['model_calls'], 2)
        self.assertGreater(self.budget.snapshot()['sandbox_runs'], 0)
        self.assertEqual(observe_slot(self.root, registration_sha256=REG, generation=0, slot=0)['status'], 'completion_recorded_unverified')
        evidence = json.loads((self.root / 'g0-slot-00.collection.json').read_text())
        self.assertEqual(evidence['status'], 'novel_unverified')
        self.assertNotIn(str(self.private[0]), repr(evidence))

    def test_wrong_task_stops_before_claim_or_dispatch(self):
        with self.assertRaises(ValueError):
            self.collect(expected_task_sha256='b' * 64)
        self.transport.assert_not_called()
        self.assertEqual(observe_slot(self.root, registration_sha256=REG, generation=0, slot=0)['status'], 'unstarted')

    def test_private_failure_is_terminal(self):
        self.transport.side_effect = [FAKE_PROGRAMS[1], FAKE_PROGRAMS[1]]
        private = [[[7, 8]]] * 5
        result = self.collect(private_probes=private, expected_private_probes_sha256=probes_sha256(private))
        self.assertEqual(result['status'], 'private_acceptance_rejected')
        self.assertEqual(self.transport.call_count, 1)

    def test_failed_dispatch_is_consumed_and_charged(self):
        self.transport.side_effect = TimeoutError('unknown dispatch outcome')
        with self.assertRaises(TimeoutError):
            self.collect()
        self.assertEqual(self.budget.snapshot()['model_calls'], 1)
        self.assertEqual(observe_slot(self.root, registration_sha256=REG, generation=0, slot=0)['status'], 'started_outcome_uncertain')
        with self.assertRaises(FileExistsError):
            self.collect()
        self.assertEqual(self.transport.call_count, 1)
