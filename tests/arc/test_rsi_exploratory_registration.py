"""Binding three task partitions does not authorize a model run."""
import hashlib
import json
import unittest
from pathlib import Path
from unittest.mock import patch

from src.arc.rsi_exploratory_registration import prepare_exploratory_registration
from src.arc.rsi_exploratory_split import propose_exploratory_split
from src.arc.rsi_task_generation import generation_slots


class ExploratoryRegistrationTests(unittest.TestCase):
    def setUp(self):
        self.generation=[f'g{i:03d}' for i in range(800)]
        self.development=[f'd{i:03d}' for i in range(200)]
        self.bounded={'status':'bounded_exposure_proposed_unverified','execution_ready':False,
                      'known_exposed_task_ids':self.generation[20:100], 'task_source_sha256':'b'*64,
                      'partitions':{'generation':self.generation,'development':self.development}}
        self.raw=json.dumps(self.bounded,sort_keys=True).encode()
        self.proposal=propose_exploratory_split(self.raw,expected_sha256=hashlib.sha256(self.raw).hexdigest())
        self.proposal['selection_code_sha256']=hashlib.sha256(Path('src/arc/rsi_exploratory_split.py').read_bytes()).hexdigest()
        self.base={'generation':0,'slots':generation_slots(self.generation,0),'input_paths':{},
                   'input_sha256':{'split':'a'*64,'challenges':'b'*64},'status':'draft_not_execution_ready','execution_ready':False}
        challenges={key:{'train':[{'input':[[i//100,(i//10)%10,i%10]],'output':[[0]]}],
                         'test':[{'input':[[1]]}]} for i,key in enumerate(self.generation+self.development)}
        self.inputs={'challenges':challenges, 'split':{'partitions':{'generation':self.generation,'development':self.development}},
                     'design_registration':{'supersedes':'research/rsi/design-registration-v2.json',
                                             'exposure_policy':'bounded_review_unknown_project_exposure'}}
        self.verify=patch('src.arc.rsi_exploratory_registration.verify_draft_registration').start()
        self.addCleanup(patch.stopall)
        patch('src.arc.rsi_exploratory_registration._read_inputs',return_value=(self.inputs,self.base['input_sha256'])).start()

    def prepare(self):
        return prepare_exploratory_registration('.',registration=self.base,bounded_bytes=self.raw,
            bounded_sha256=hashlib.sha256(self.raw).hexdigest(),exploratory=self.proposal)

    def test_binds_disjoint_manifests_without_authorization(self):
        result=self.prepare()
        self.assertFalse(result['execution_ready']); self.assertFalse(result['independent_review_completed'])
        self.assertEqual(result['counts'],{'generation':750,'exploratory':50,'development':200})
        self.assertEqual(len(result['generation_manifest_sha256']),64)
        self.assertEqual(len(result['generation_content_sha256']),64)
        self.assertEqual(len(result['exploratory_content_sha256']),64)
        self.assertEqual(self.verify.call_count,2)

    def test_rejects_changed_selection_and_base_split(self):
        self.proposal['exploratory_task_ids'][0]=self.generation[0]
        with self.assertRaises(ValueError):self.prepare()
        self.proposal=propose_exploratory_split(self.raw,expected_sha256=hashlib.sha256(self.raw).hexdigest())
        self.proposal['selection_code_sha256']=hashlib.sha256(Path('src/arc/rsi_exploratory_split.py').read_bytes()).hexdigest()
        self.inputs['split']['partitions']['development'][0]='other'
        with self.assertRaises(ValueError):self.prepare()

    def test_rejects_identical_content_under_different_ids(self):
        key=self.proposal['exploratory_task_ids'][0]
        self.inputs['challenges'][key]=self.inputs['challenges'][self.development[0]]
        with self.assertRaisesRegex(ValueError,'content'):
            self.prepare()

    def test_rejects_hidden_labels_in_challenge_content(self):
        key=self.proposal['exploratory_task_ids'][0]
        self.inputs['challenges'][key]['test'][0]['output']=[[1]]
        with self.assertRaisesRegex(ValueError,'hidden'):
            self.prepare()

    def test_rejects_changed_scheduled_slots(self):
        self.base['slots']=generation_slots(self.generation,1)
        with self.assertRaises(ValueError):self.prepare()
