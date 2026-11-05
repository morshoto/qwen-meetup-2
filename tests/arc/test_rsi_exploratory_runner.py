"""Exercise the exploratory runner only with caller-supplied fixtures."""
import itertools
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from src.arc.data import Task, Verifier
from src.arc.e6_fake import FAKE_PROGRAMS, FakeE6Sandbox
from src.arc.rsi_exploratory_budget import ExploratoryBudget
from src.arc.rsi_exploratory_runner import run_exploratory
from src.arc.rsi_exploratory_incomplete import read_incomplete_exploratory
from src.arc.rsi_slot_audit import manifest_sha256, task_sha256


class ExploratoryRunnerTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup); self.root=Path(self.tmp.name)
        self.ids=[f't{i:03d}' for i in range(50)]; self.final=[f'c{i:03d}' for i in range(200)]
        self.tasks={k:Task(k,(([[1,2]],[[2,1]]),),([[3,4]],)) for k in self.ids}
        self.verifier=Verifier({k:[[[4,3]]] for k in self.ids})
        self.seeds={k:[2*i+100,2*i+101] for i,k in enumerate(self.ids)}
        self.types={k:None for k in self.ids}; self.counter=itertools.count()
        self.reg={'status':'exploratory_registration_draft_unverified','execution_ready':False,
                  'exploratory_manifest_sha256':manifest_sha256(self.ids),
                  'development_manifest_sha256':manifest_sha256(self.final)}
        self.reg_sha=hashlib.sha256(json.dumps(self.reg,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
        self.budget=ExploratoryBudget.create(self.root,registration_sha256=self.reg_sha,generation=0)
        self.generate=Mock(side_effect=lambda *a,**kw:{'code':FAKE_PROGRAMS[1],
            'request_id':f'{next(self.counter):032x}','usage':{'prompt_tokens':10,'completion_tokens':5}})
    def run_trial(self):
        return run_exploratory(root=self.root,registration_sha256=self.reg_sha,generation=0,registration=self.reg,
            task_ids=self.ids,confirmatory_task_ids=self.final,tasks=self.tasks,
            task_hashes={k:task_sha256(v) for k,v in self.tasks.items()},seeds=self.seeds,task_types=self.types,
            candidate_adapter='b'*64,generate=self.generate,sandbox=FakeE6Sandbox(),verifier=self.verifier,budget=self.budget)
    def test_matched_fixture_records_and_coverage(self):
        result=self.run_trial()
        self.assertEqual(result['status'],'exploratory_complete_unverified')
        self.assertFalse(result['claim_eligible']); self.assertEqual(self.budget.snapshot()['model_calls'],200)
        self.assertEqual(result['summary']['arms']['base']['correct'],50)
        calls=self.generate.call_args_list[:4]
        self.assertEqual([c.kwargs['seed'] for c in calls],[100,101,100,101])
        self.assertTrue(all(c.args[0]==calls[0].args[0] for c in calls))
        self.assertNotIn('[[4, 3]]',calls[0].args[0])
        self.assertEqual(result['summary']['arms']['base']['total_tokens'],1500)
        marker=json.loads((self.root/'exploratory.completed.json').read_text())
        self.assertEqual(marker['result_sha256'],result['sha256'])
        self.assertEqual(marker['budget_after']['model_calls'],200)
        with self.assertRaisesRegex(ValueError, "fresh"): self.run_trial()
    def test_interruption_is_incomplete_and_never_replayed(self):
        self.generate.side_effect=TimeoutError('unavailable')
        result=self.run_trial()
        self.assertEqual(result['status'],'exploratory_incomplete')
        self.assertEqual(self.budget.snapshot()['model_calls'],1)
        self.assertFalse(result['claim_eligible'])
        self.assertFalse((self.root/'exploratory.completed.json').exists())
        with self.assertRaisesRegex(ValueError, "fresh"): self.run_trial()
    def test_incomplete_completed_rows_can_be_read_without_fabricating_scores(self):
        calls = self.generate.side_effect
        def interrupted(*args, **kwargs):
            if self.generate.call_count > 4: raise TimeoutError('unavailable')
            return calls(*args, **kwargs)
        self.generate.side_effect = interrupted
        result = self.run_trial()
        self.assertEqual(result['completed'], {'base':1, 'trained':1})
        view = read_incomplete_exploratory(root=self.root, registration=self.reg, registration_sha256=self.reg_sha,
            task_ids=self.ids, confirmatory_task_ids=self.final, candidate_adapter='b'*64)
        self.assertEqual(view['status'], 'exploratory_partial_view_unverified')
        self.assertFalse(view['claim_eligible'])
        self.assertFalse(view['summary']['complete'])
        for arm in ('base', 'trained'):
            self.assertEqual(view['summary']['arms'][arm]['evaluated'],1)
            self.assertIsNone(view['summary']['arms'][arm]['total_tokens'])
        self.assertFalse((self.root/'exploratory-results.json').exists())
        with self.assertRaises(ValueError):
            read_incomplete_exploratory(root=self.root,registration=self.reg,registration_sha256='0'*64,
                task_ids=self.ids,confirmatory_task_ids=self.final,candidate_adapter='b'*64)

    def test_full_rows_from_interrupted_run_remain_incomplete(self):
        with patch('src.arc.rsi_exploratory_runner.summarize_exploratory',side_effect=ValueError('fixture')):
            result=self.run_trial()
        self.assertEqual(result['completed'],{'base':50,'trained':50})
        view=read_incomplete_exploratory(root=self.root,registration=self.reg,registration_sha256=self.reg_sha,
            task_ids=self.ids,confirmatory_task_ids=self.final,candidate_adapter='b'*64)
        self.assertTrue(view['row_coverage_complete'])
        self.assertFalse(view['summary']['complete'])
        self.assertFalse(view['claim_eligible'])

    def test_results_written_but_completion_failed_stays_incomplete(self):
        from src.arc.rsi_slot_lifecycle import _create as real_create
        def fail_completion(directory, name, value):
            if name == 'exploratory.completed.json': raise OSError('fixture failure')
            return real_create(directory, name, value)
        with patch('src.arc.rsi_exploratory_runner._create',side_effect=fail_completion):
            result = self.run_trial()
        self.assertEqual(result['status'],'exploratory_incomplete')
        self.assertTrue((self.root/'exploratory-results.json').exists())
        args=dict(root=self.root,registration=self.reg,registration_sha256=self.reg_sha,
                  task_ids=self.ids,confirmatory_task_ids=self.final,candidate_adapter='b'*64)
        view=read_incomplete_exploratory(**args)
        self.assertTrue(view['row_coverage_complete'])
        self.assertFalse(view['summary']['complete'])
        self.assertFalse(view['claim_eligible'])
        path=self.root/'exploratory-results.json'
        changed=json.loads(path.read_text());changed['arms']['base']['results'][0]['solved']=False
        path.write_text(json.dumps(changed))
        with self.assertRaises(ValueError): read_incomplete_exploratory(**args)

    def test_changed_manifest_fails_before_dispatch(self):
        self.final[0]=self.ids[0]
        with self.assertRaises(ValueError): self.run_trial()
        self.generate.assert_not_called()

    def test_changed_registration_rejected_before_dispatch(self):
        self.reg['unbound']='changed'
        with self.assertRaises(ValueError): self.run_trial()
        self.generate.assert_not_called()
