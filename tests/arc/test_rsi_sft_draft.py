"""Construct quarantined SFT drafts without exposing generated query answers."""
import json
import unittest

from src.arc.data import Task
from src.arc.e6_curriculum import _quality_grids
from src.arc.e6_fake import FAKE_PROGRAMS, FakeE6Sandbox
from src.arc.rsi_candidate import inspect_candidate
from src.arc.rsi_novelty import screen_novelty
from src.arc.rsi_sft_draft import make_sft_draft


class SftDraftTests(unittest.TestCase):
    def setUp(self):
        self.task = Task('parent', (([[1, 2]], [[2, 1]]),), ([[3, 4]],))
        self.code = FAKE_PROGRAMS[1]
        self.candidate = inspect_candidate(self.task, self.code,
            public_probes=_quality_grids(42, 5), private_probes=_quality_grids(43, 5), sandbox=FakeE6Sandbox())
        self.novelty = screen_novelty(self.code, self.candidate['private_behavior_sha256'],
                                      self.candidate['private_probe_set_sha256'], [])
        self.provenance = {'source_type': 'generator_distillation', 'model_id': 'mlx-community/Qwen3-14B-4bit',
                           'model_revision': 'a4d9b2df59d2c150bef02fcbe0d91046b7ca33a4',
                           'request_id': 'fixture', 'raw_response_sha256': 'a' * 64,
                           'execution_mode': 'fake_fixture'}

    def test_draft_contains_no_query_outputs_or_private_probe_payload(self):
        draft = make_sft_draft(self.task, self.code, self.candidate, self.novelty,
                               self.provenance, generation=0, parent_adapter='none')
        self.assertFalse(draft['eligible_for_training'])
        self.assertEqual(draft['status'], 'pending_independent_audit')
        self.assertEqual(draft['source_type'], 'generator_distillation')
        self.assertTrue(draft['mask_prompt'])
        self.assertEqual(draft['messages'][-1]['content'], self.code)
        self.assertNotIn('query_outputs', json.dumps(draft))
        self.assertNotIn('[[4,3]]', json.dumps(draft, separators=(',', ':')))
        self.assertNotIn('private_probes', json.dumps(draft))

    def test_rejects_source_substitution_or_duplicate_novelty(self):
        with self.assertRaisesRegex(ValueError, 'source'):
            make_sft_draft(self.task, FAKE_PROGRAMS[2], self.candidate, self.novelty,
                           self.provenance, generation=0, parent_adapter='none')
        with self.assertRaisesRegex(ValueError, 'novel'):
            make_sft_draft(self.task, self.code, self.candidate, {'status': 'duplicate_ast'},
                           self.provenance, generation=0, parent_adapter='none')

    def test_rejects_wrong_model_or_generation_parent(self):
        with self.assertRaisesRegex(ValueError, 'provenance'):
            make_sft_draft(self.task, self.code, self.candidate, self.novelty,
                           dict(self.provenance, model_revision='other'), generation=0, parent_adapter='none')
        with self.assertRaisesRegex(ValueError, 'parent'):
            make_sft_draft(self.task, self.code, self.candidate, self.novelty,
                           self.provenance, generation=1, parent_adapter='none')


if __name__ == '__main__':
    unittest.main()
