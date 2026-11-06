"""Reserve exploratory tasks without contaminating later evaluation."""
import hashlib
import json
import unittest

from src.arc.rsi_exploratory_split import propose_exploratory_split


class ExploratorySplitTests(unittest.TestCase):
    def setUp(self):
        self.generation=[f'g{i:03d}' for i in range(800)]
        self.development=[f'd{i:03d}' for i in range(200)]
        self.value={'status':'bounded_exposure_proposed_unverified','execution_ready':False,
                    'known_exposed_task_ids':self.generation[20:100],
                    'partitions':{'generation':self.generation,'development':self.development}}
    def propose(self,v=None):
        raw=json.dumps(v or self.value,sort_keys=True).encode()
        return propose_exploratory_split(raw,expected_sha256=hashlib.sha256(raw).hexdigest())
    def test_reserves_known_unexposed_tasks_away_from_generation_and_later_200(self):
        r=self.propose(); chosen=set(r['exploratory_task_ids'])
        self.assertEqual(len(chosen),50); self.assertEqual(len(r['generation_task_ids']),750)
        self.assertFalse(chosen & set(self.value['known_exposed_task_ids']))
        self.assertFalse(chosen & set(self.generation[:20])); self.assertFalse(chosen & set(self.development))
        self.assertFalse(chosen & set(r['generation_task_ids'])); self.assertFalse(r['execution_ready'])
    def test_rejects_insufficient_candidates_and_overlapping_manifests(self):
        v=dict(self.value,known_exposed_task_ids=self.generation[20:])
        with self.assertRaisesRegex(ValueError,'eligible'): self.propose(v)
        v=json.loads(json.dumps(self.value)); v['partitions']['development'][0]=self.generation[0]
        with self.assertRaises(ValueError): self.propose(v)
    def test_rejects_changed_digest(self):
        with self.assertRaises(ValueError): propose_exploratory_split(b'{}',expected_sha256='a'*64)
