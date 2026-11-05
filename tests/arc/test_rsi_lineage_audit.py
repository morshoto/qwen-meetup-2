"""Replay repair lineage while keeping all conclusions provisional."""
import hashlib
import json
import unittest

from src.arc.data import Task
from src.arc.e6_curriculum import _quality_grids
from src.arc.e6_fake import FAKE_PROGRAMS, FakeE6Sandbox
from src.arc.rsi_task_generation import task_prompt, task_repair_prompt
from src.arc.rsi_slot_audit import manifest_sha256, task_sha256
from src.arc.rsi_lineage_audit import inspect_lineage, probes_sha256
from tests.arc.test_rsi_transport_audit import build_evidence


def attempt(prompt, code, seed, request_id):
    raw_request, raw_response, ledger = build_evidence()
    request = json.loads(raw_request)
    response = json.loads(raw_response)
    request['messages'][1]['content'] = prompt
    request['seed'] = seed
    response['choices'][0]['message']['content'] = code
    raw_request, raw_response = json.dumps(request).encode(), json.dumps(response).encode()
    ledger.update(request_id=request_id, seed_requested=seed,
                  request_body_sha256=hashlib.sha256(raw_request).hexdigest(),
                  response_sha256=hashlib.sha256(raw_response).hexdigest())
    return {'request_body': raw_request, 'response_body': raw_response, 'ledger': ledger, 'code': code}


class LineageTests(unittest.TestCase):
    def setUp(self):
        self.ids = [f't{i:02}' for i in range(20)]
        self.task = Task('t00', (([[1, 2]], [[2, 1]]),), ([[3, 4]],))
        self.public, self.private = _quality_grids(42, 5), _quality_grids(43, 5)
        self.first = attempt(task_prompt(self.task), FAKE_PROGRAMS[2], 12000, 'r0')
        second_prompt = task_repair_prompt(self.task, FAKE_PROGRAMS[2], 'visible_demonstration_mismatch')
        self.second = attempt(second_prompt, FAKE_PROGRAMS[1], 12001, 'r1')

    def check(self, rows, *, private=None, expected_private=None, sandbox=None):
        selected_private = self.private if private is None else private
        return inspect_lineage(self.ids, self.task, generation=0, slot=0, attempts=rows,
                               expected_manifest_sha256=manifest_sha256(self.ids),
                               expected_task_sha256=task_sha256(self.task),
                               public_probes=self.public,
                               private_probes=selected_private,
                               expected_public_probes_sha256=probes_sha256(self.public),
                               expected_private_probes_sha256=expected_private or probes_sha256(selected_private),
                               sandbox=sandbox or FakeE6Sandbox())

    def test_replays_visible_failure_then_model_repair_without_claiming_training(self):
        result = self.check([self.first, self.second])
        self.assertEqual(result['attempt_statuses'], ['visible_demonstration_mismatch', 'verified_candidate'])
        self.assertEqual(result['status'], 'lineage_consistent')
        self.assertFalse(result['eligible_for_training'])
        self.assertNotIn(str(self.private[0]), repr(result))

    def test_rejects_tampered_repair_prompt_or_duplicate_request(self):
        tampered = attempt(task_prompt(self.task), FAKE_PROGRAMS[1], 12001, 'r1')
        with self.assertRaises(ValueError):
            self.check([self.first, tampered])
        duplicate = attempt(task_repair_prompt(self.task, FAKE_PROGRAMS[2],
                            'visible_demonstration_mismatch'), FAKE_PROGRAMS[1], 12001, 'r0')
        with self.assertRaisesRegex(ValueError, 'request'):
            self.check([self.first, duplicate])

    def test_private_failure_or_success_cannot_be_followed_by_repair(self):
        success = attempt(task_prompt(self.task), FAKE_PROGRAMS[1], 12000, 'r0')
        with self.assertRaisesRegex(ValueError, 'after terminal'):
            self.check([success, self.second])
        with self.assertRaisesRegex(ValueError, 'after terminal'):
            self.check([success, self.second], private=[[[7, 8]]] * 5)

    def test_rejects_more_than_three_attempts(self):
        with self.assertRaises(ValueError):
            self.check([self.first] * 4)

    def test_changed_probe_set_is_rejected_before_replay(self):
        with self.assertRaisesRegex(ValueError, 'probe'):
            self.check([self.first], private=_quality_grids(44, 5),
                       expected_private=probes_sha256(self.private))

    def test_syntax_failure_can_be_repaired_using_safe_category(self):
        bad = 'def transform(grid):\n    return ('
        first = attempt(task_prompt(self.task), bad, 12000, 'r0')
        repair = attempt(task_repair_prompt(self.task, bad, 'syntax_error'), FAKE_PROGRAMS[1], 12001, 'r1')
        result = self.check([first, repair])
        self.assertEqual(result['attempt_statuses'], ['syntax_error', 'verified_candidate'])

    def test_runtime_failure_can_be_repaired_without_exception_details(self):
        class BrokenFirst(FakeE6Sandbox):
            def run(self, code, grid):
                if code == FAKE_PROGRAMS[2]:
                    raise RuntimeError('sandbox execution failed:\nIndexError: private detail')
                return super().run(code, grid)
        prompt = task_repair_prompt(self.task, FAKE_PROGRAMS[2], 'runtime_exception')
        second = attempt(prompt, FAKE_PROGRAMS[1], 12001, 'r1')
        result = self.check([self.first, second], sandbox=BrokenFirst())
        self.assertEqual(result['attempt_statuses'], ['runtime_exception', 'verified_candidate'])
        self.assertNotIn('private detail', repr(result))

    def test_prohibited_code_cannot_trigger_a_repair(self):
        bad = 'import os\ndef transform(grid): return grid'
        first = attempt(task_prompt(self.task), bad, 12000, 'r0')
        with self.assertRaisesRegex(ValueError, 'after terminal'):
            self.check([first, self.second])


if __name__ == '__main__':
    unittest.main()
