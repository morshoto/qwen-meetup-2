import copy
import hashlib
import json
import unittest
from unittest.mock import Mock

from src.arc.e6_curriculum import _quality_grids
from src.arc.e6_fake import FAKE_PROGRAMS, FakeE6Sandbox
from src.arc.rsi_lineage_audit import probes_sha256
from src.arc.rsi_novelty import screen_novelty
from src.arc.rsi_reference_pool import build_reference_pool


def entry(index):
    code = FAKE_PROGRAMS[index]
    return {'record_id': f'r{index}', 'source': code,
            'source_sha256': hashlib.sha256(code.strip().encode()).hexdigest(),
            'origin_sha256': str(index) * 64, 'ancestor_task_ids': [f't{index}']}


class ReferencePoolTests(unittest.TestCase):
    def setUp(self):
        self.probes = _quality_grids(43, 5)
        self.entries = [entry(1), entry(4)]
        self.sandbox = FakeE6Sandbox()

    def build(self, **changes):
        args = dict(entries=self.entries, private_probes=self.probes,
                    expected_probe_sha256=probes_sha256(self.probes), generation_task_ids={'t1', 't4'},
                    development_task_ids={'dev'}, sandbox=self.sandbox, max_sandbox_starts=20)
        args.update(changes)
        return build_reference_pool(**args)

    def test_recomputed_fingerprints_are_duplicate_only(self):
        result = self.build()
        self.assertEqual(result['status'], 'reference_pool_unverified')
        self.assertEqual(result['sandbox_starts'], 20)
        self.assertFalse(result['eligible_for_training'])
        self.assertEqual(len(result['pool']), 2)
        first = result['pool'][0]
        novelty = screen_novelty(self.entries[0]['source'], first['behavior_sha256'], first['probe_set_sha256'], result['pool'])
        self.assertEqual(novelty['status'], 'duplicate_source')
        self.assertNotIn('private_probes', json.dumps(result))
        self.assertNotIn('query_outputs', json.dumps(result))

    def test_unknown_ancestry_development_overlap_or_source_mismatch_stops_before_dispatch(self):
        for change in ('missing', 'overlap', 'hash', 'duplicate'):
            entries = copy.deepcopy(self.entries)
            if change == 'missing': entries[0]['ancestor_task_ids'] = []
            if change == 'overlap': entries[0]['ancestor_task_ids'] = ['dev']
            if change == 'hash': entries[0]['source_sha256'] = 'f' * 64
            if change == 'duplicate': entries[1]['record_id'] = entries[0]['record_id']
            sandbox = Mock()
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.build(entries=entries, sandbox=sandbox)
            sandbox.run.assert_not_called()

    def test_insufficient_start_budget_or_wrong_probes_stops_before_dispatch(self):
        sandbox = Mock()
        with self.assertRaises(ValueError):
            self.build(max_sandbox_starts=19, sandbox=sandbox)
        with self.assertRaises(ValueError):
            self.build(expected_probe_sha256='f' * 64, sandbox=sandbox)
        sandbox.run.assert_not_called()

    def test_sandbox_failure_aborts_instead_of_omitting_reference(self):
        sandbox = Mock()
        sandbox.run.side_effect = TimeoutError('unavailable')
        with self.assertRaises(TimeoutError):
            self.build(sandbox=sandbox)
        self.assertEqual(sandbox.run.call_count, 1)
