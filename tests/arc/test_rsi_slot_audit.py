"""Bind captured request consistency to a frozen schedule and source task."""
import hashlib
import json
import unittest

from src.arc.data import Task
from src.arc.rsi_task_generation import task_prompt
from src.arc.rsi_slot_audit import audit_scheduled_attempt, manifest_sha256, task_sha256
from tests.arc.test_rsi_transport_audit import build_evidence, CODE


class SlotAuditTests(unittest.TestCase):
    def setUp(self):
        self.ids = [f't{i:02}' for i in range(20)]
        self.task = Task('t00', (([[1, 2]], [[2, 1]]),), ([[3, 4]],))
        raw, self.response, self.ledger = build_evidence()
        req = json.loads(raw)
        req['messages'][1]['content'] = task_prompt(self.task)
        self.request = json.dumps(req).encode()
        self.ledger['request_body_sha256'] = hashlib.sha256(self.request).hexdigest()

    def check(self, *, ids=None, task=None, slot=0, generation=0, manifest=None, task_hash=None):
        return audit_scheduled_attempt(self.ids if ids is None else ids, self.task if task is None else task,
            generation=generation, slot=slot, attempt=0, request_body=self.request,
            response_body=self.response, ledger=self.ledger, expected_code=CODE,
            expected_manifest_sha256=manifest or manifest_sha256(self.ids),
            expected_task_sha256=task_hash or task_sha256(self.task))

    def test_first_attempt_matches_schedule_and_cannot_authorize_training(self):
        result = self.check()
        self.assertEqual(result['status'], 'slot_transport_consistent')
        self.assertEqual(result['task_id'], 't00')
        self.assertEqual(result['seed_requested'], 12000)
        self.assertFalse(result['eligible_for_training'])

    def test_changed_manifest_task_or_slot_fails(self):
        with self.assertRaises(ValueError):
            self.check(ids=self.ids[:-1] + ['different'])
        with self.assertRaises(ValueError):
            self.check(task=Task('t00', (([[1, 2]], [[9, 9]]),), ([[3, 4]],)))
        with self.assertRaises(ValueError):
            self.check(slot=1)

    def test_wrong_generation_seed_and_repair_attempt_fail(self):
        with self.assertRaises(ValueError):
            self.check(generation=1)
        with self.assertRaises(ValueError):
            audit_scheduled_attempt(self.ids, self.task, generation=0, slot=0, attempt=1,
                request_body=self.request, response_body=self.response, ledger=self.ledger,
                expected_code=CODE, expected_manifest_sha256=manifest_sha256(self.ids),
                expected_task_sha256=task_sha256(self.task))


if __name__ == '__main__':
    unittest.main()
