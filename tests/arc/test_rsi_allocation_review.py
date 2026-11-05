"""Review candidate IDs and exact content without freezing the split."""
import hashlib
import json
import unittest
from pathlib import Path

from src.arc.rsi_allocation_review import review_task_allocation
from src.arc.rsi_exploratory_split import propose_exploratory_split


class AllocationReviewTests(unittest.TestCase):
    def setUp(self):
        self.gen=[f'g{i:03d}' for i in range(800)]; self.dev=[f'd{i:03d}' for i in range(200)]
        self.tasks={k:{'train':[{'input':[[i//100,(i//10)%10,i%10]],'output':[[0]]}],
                       'test':[{'input':[[1]]}]} for i,k in enumerate(self.gen+self.dev)}
        self.dataset=json.dumps(self.tasks,sort_keys=True).encode()
        self.bounded={'status':'bounded_exposure_proposed_unverified','execution_ready':False,
                      'task_source_sha256':hashlib.sha256(self.dataset).hexdigest(),
                      'known_exposed_task_ids':self.gen[:317],
                      'partitions':{'generation':self.gen,'development':self.dev}}
        self.raw=json.dumps(self.bounded,sort_keys=True).encode(); self.sha=hashlib.sha256(self.raw).hexdigest()
        self.selected=propose_exploratory_split(self.raw,expected_sha256=self.sha)
        self.selected['selection_code_sha256']=hashlib.sha256(Path('src/arc/rsi_exploratory_split.py').read_bytes()).hexdigest()
    def review(self):
        return review_task_allocation(self.dataset,self.raw,self.selected,expected_bounded_sha256=self.sha)
    def test_checks_all_known_exposures_and_no_content_overlap(self):
        result=self.review()
        self.assertEqual(result['counts'],{'generation':750,'exploratory':50,'development':200})
        self.assertEqual(result['known_exposed_counts'],{'generation':317,'exploratory':0,'development':0})
        self.assertFalse(result['independent_review_completed']); self.assertFalse(result['execution_ready'])
    def test_rejects_changed_selection_or_dataset(self):
        self.selected['exploratory_task_ids'][0]=self.gen[0]
        with self.assertRaises(ValueError):self.review()
        with self.assertRaises(ValueError):review_task_allocation(b'{}',self.raw,self.selected,expected_bounded_sha256=self.sha)
    def test_rejects_duplicate_content_and_hidden_labels(self):
        for mode in ('duplicate','label'):
            values=json.loads(self.dataset)
            if mode=='duplicate': values[self.dev[0]]=values[self.selected['exploratory_task_ids'][0]]
            else: values[self.dev[0]]['test'][0]['output']=[[1]]
            data=json.dumps(values,sort_keys=True).encode(); bounded=dict(self.bounded,task_source_sha256=hashlib.sha256(data).hexdigest())
            raw=json.dumps(bounded,sort_keys=True).encode(); digest=hashlib.sha256(raw).hexdigest()
            selection=propose_exploratory_split(raw,expected_sha256=digest); selection['selection_code_sha256']=self.selected['selection_code_sha256']
            with self.subTest(mode=mode),self.assertRaises(ValueError):review_task_allocation(data,raw,selection,expected_bounded_sha256=digest)
