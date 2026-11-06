"""Check prompt review replays without claiming independent provenance."""
import copy
import unittest

from src.arc.rsi_prompt_reconciliation import reconcile_prompt_reviews


class ReconciliationTests(unittest.TestCase):
    def setUp(self):
        self.proposal = {'status':'bounded_exposure_proposed_unverified', 'execution_ready':False,
                         'task_source_sha256':'a'*64, 'partitions':{'development':['a','b']}}
        self.split = {'partitions':{'development':['a','b']}}
        self.reviews = {name:{'status':status, 'dataset_sha256':'a'*64, 'split_sha256':'b'*64,
                              'inventory_sha256':'c'*64, 'file_count':3, 'matches':{}, 'independence_established':False}
                        for name,status in [('exact','bounded_exact_prompt_review_only'),
                                            ('hash','bounded_historical_prompt_hash_review_only'),
                                            ('text','bounded_historical_task_text_review_only')]}
        self.replay = {name:{'file_count':3,'inventory_sha256':'c'*64,'matches':{}} for name in self.reviews}

    def run_review(self):
        return reconcile_prompt_reviews(self.proposal,self.split,'b'*64,self.reviews,self.replay)

    def test_accepts_matching_replay_but_does_not_authorize(self):
        result=self.run_review()
        self.assertFalse(result['independent_review_completed'])
        self.assertFalse(result['execution_ready'])
        self.assertEqual(result['development_count'],2)

    def test_rejects_changed_split_dataset_or_replay(self):
        for target,key,value in [(self.split['partitions'],'development',['b','a']),
                                 (self.reviews['exact'],'dataset_sha256','d'*64),
                                 (self.replay['hash'],'inventory_sha256','d'*64),
                                 (self.replay['text'],'matches',{'a':['x']})]:
            old=copy.deepcopy(target[key]); target[key]=value
            with self.subTest(key=key), self.assertRaises(ValueError): self.run_review()
            target[key]=old

    def test_rejects_matched_development_even_if_replay_agrees(self):
        self.reviews['hash']['matches']={'a/direct':['x']}; self.replay['hash']['matches']={'a/direct':['x']}
        with self.assertRaisesRegex(ValueError,'exposure'): self.run_review()
