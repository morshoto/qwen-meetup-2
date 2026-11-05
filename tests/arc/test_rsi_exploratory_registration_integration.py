"""Bind a three-part proposal to a real draft registration using local fixtures."""
import hashlib
import json
import shutil
import unittest
from pathlib import Path

from src.arc.rsi_exploratory_registration import prepare_exploratory_registration
from src.arc.rsi_exploratory_split import propose_exploratory_split
from src.arc.rsi_registration import prepare_draft_registration


class ExploratoryRegistrationIntegrationTests(unittest.TestCase):
    def test_real_draft_registration_and_distinct_task_content(self):
        from tests.arc.test_rsi_registration import DraftRegistrationTests
        fixture=DraftRegistrationTests(methodName='test_draft_binds_inputs_schedule_and_remains_non_authorizing')
        fixture.setUp(); self.addCleanup(fixture.doCleanups)
        root=fixture.root; paths=fixture.paths
        ids=[f't{i:04}' for i in range(1000)]
        challenges={key:{'train':[{'input':[[i//100,(i//10)%10,i%10]],'output':[[0]]}],
                         'test':[{'input':[[1]]}]} for i,key in enumerate(ids)}
        (root/paths['challenges']).write_text(json.dumps(challenges))
        code=root/'src/arc/rsi_exploratory_split.py'; code.parent.mkdir(parents=True)
        shutil.copyfile('src/arc/rsi_exploratory_split.py',code)
        (root/'research/rsi/design-v3.md').write_text('v3 design')
        sha=lambda p:hashlib.sha256((root/p).read_bytes()).hexdigest()
        v3={'kind':'design_only','execution_ready':False,'competitive_target_percent':42.4,
            'supersedes':'research/rsi/design-registration-v2.json',
            'exposure_policy':'bounded_review_unknown_project_exposure',
            'files':{p:sha(p) for p in ('research/rsi/design-registration-v2.json','research/rsi/design-v3.md')}}
        (root/'research/rsi/design-registration-v3.json').write_text(json.dumps(v3))
        paths['design_registration']='research/rsi/design-registration-v3.json'
        base=prepare_draft_registration(root,input_paths=paths,generation=0)
        bounded={'status':'bounded_exposure_proposed_unverified','execution_ready':False,
                 'task_source_sha256':sha(paths['challenges']),'known_exposed_task_ids':ids[20:100],
                 'partitions':{'generation':ids[:800],'development':ids[800:]}}
        raw=json.dumps(bounded,sort_keys=True).encode(); digest=hashlib.sha256(raw).hexdigest()
        selection=propose_exploratory_split(raw,expected_sha256=digest); selection['selection_code_sha256']=sha('src/arc/rsi_exploratory_split.py')
        result=prepare_exploratory_registration(root,registration=base,bounded_bytes=raw,
                                                bounded_sha256=digest,exploratory=selection)
        self.assertEqual(result['counts'],{'generation':750,'exploratory':50,'development':200})
        self.assertFalse(result['execution_ready'])
        self.assertFalse(result['independent_review_completed'])
