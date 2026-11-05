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

class LegacyReferencePoolTests(unittest.TestCase):
    def setUp(self):
        from src.arc.ast_fingerprint import stable_ast_sha256
        self.code=FAKE_PROGRAMS[1];self.probes=_quality_grids(43,5)
        self.metadata={'status':'historical_static_reference_unverified','model_key':'14b',
            'model_id':'mlx-community/Qwen3-14B-4bit','model_revision':'a4d9b2df59d2c150bef02fcbe0d91046b7ca33a4',
            'record_task_id':'e6f-'+'a'*24,'source_sha256':hashlib.sha256(self.code.strip().encode()).hexdigest(),
            'ast_sha256':stable_ast_sha256(self.code),'candidate_index_sha256':'b'*64,
            'e8_slot_result_sha256':'c'*64,'e9_existing_oracles_sha256':'d'*64,
            'eligible_for_training':False,'independently_verified_for_new_protocol':False,
            'candidate_code_executed':False,'private_behavior_for_proposed_probes':'not_evaluated'}
        self.raw=json.dumps(self.metadata,sort_keys=True).encode()
    def args(self):
        return dict(model_key='14b',source=self.code,metadata_bytes=self.raw,
                    expected_metadata_sha256=hashlib.sha256(self.raw).hexdigest(),
                    private_probes=self.probes,expected_probe_sha256=probes_sha256(self.probes),
                    sandbox=FakeE6Sandbox(),max_sandbox_starts=10)
    def test_legacy_id_is_kept_separate_and_not_training_eligible(self):
        from src.arc.rsi_reference_pool import build_legacy_duplicate_reference
        result=build_legacy_duplicate_reference(**self.args())
        self.assertEqual(result['sandbox_starts'],10)
        self.assertEqual(result['historical_task_id'],self.metadata['record_task_id'])
        self.assertFalse(result['eligible_for_training']);self.assertFalse(result['source_origins_verified'])
        self.assertNotIn('private_probes',json.dumps(result));self.assertNotIn('query_outputs',json.dumps(result))
        self.assertEqual(screen_novelty(self.code,result['pool'][0]['behavior_sha256'],
                         probes_sha256(self.probes),result['pool'])['status'],'duplicate_source')
    def test_cross_model_hash_probe_and_budget_fail_before_sandbox(self):
        from src.arc.rsi_reference_pool import build_legacy_duplicate_reference
        for change in ({'model_key':'32b'},{'source':FAKE_PROGRAMS[4]},
                       {'expected_metadata_sha256':'0'*64},{'expected_probe_sha256':'0'*64},
                       {'max_sandbox_starts':9}):
            sandbox=Mock();args=self.args();args.update(change,sandbox=sandbox)
            with self.subTest(change=change),self.assertRaises(ValueError): build_legacy_duplicate_reference(**args)
            sandbox.run.assert_not_called()
