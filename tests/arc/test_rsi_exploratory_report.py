"""Summaries for a separate, non-confirmatory 50-task comparison."""
import copy
import unittest

from src.arc.rsi_exploratory_report import summarize_exploratory


class ExploratoryReportTests(unittest.TestCase):
    def setUp(self):
        ids = [f't{i:03d}' for i in range(50)]
        self.value = {'kind':'exploratory_50', 'model':'mlx-community/Qwen3-14B-4bit',
                      'revision':'a4d9b2df59d2c150bef02fcbe0d91046b7ca33a4',
                      'task_ids':ids, 'confirmatory_task_ids':[f'c{i:03d}' for i in range(200)],
                      'arms':{name:{'adapter':adapter,'settings':{'temperature':0.0,'samples':2},'results':[]}
                              for name,adapter in [('base','none'),('trained','a'*64)]}}
        for i,tid in enumerate(ids):
            for name in ('base','trained'):
                self.value['arms'][name]['results'].append({'task_id':tid,'task_type':'shape' if i<10 else None,
                    'solved':i==0,'failure_reason':None if i==0 else 'incorrect_output',
                    'elapsed_seconds':1.0,'prompt_tokens':2,'completion_tokens':3,'seeds':[i*2,i*2+1]})

    def test_reports_counts_types_failures_and_usage(self):
        report=summarize_exploratory(self.value)
        self.assertTrue(report['complete']); self.assertFalse(report['claim_eligible'])
        self.assertEqual(report['arms']['base']['correct'],1)
        self.assertEqual(report['arms']['base']['types']['shape']['correct'],1)
        self.assertEqual(report['arms']['base']['failure_reasons'],{'incorrect_output':49})
        self.assertEqual(report['arms']['base']['elapsed_seconds_total'],50)
        self.assertEqual(report['arms']['base']['total_tokens'],250)

    def test_incomplete_usage_and_results_stay_unknown(self):
        self.value['arms']['trained']['results'].pop()
        self.value['arms']['base']['results'][0]['prompt_tokens']=None
        report=summarize_exploratory(self.value)
        self.assertFalse(report['complete']); self.assertEqual(report['arms']['trained']['evaluated'],49)
        self.assertIsNone(report['arms']['base']['total_tokens'])
        self.assertEqual(report['arms']['base']['token_coverage'],49)

    def test_rejects_overlap_mismatched_settings_or_seeds(self):
        for mutate in (lambda v:v['confirmatory_task_ids'].__setitem__(0,'t000'),
                       lambda v:v['arms']['trained']['settings'].__setitem__('temperature',0.5),
                       lambda v:v['arms']['trained']['results'][0].__setitem__('seeds',[10,11])):
            value=copy.deepcopy(self.value); mutate(value)
            with self.assertRaises(ValueError): summarize_exploratory(value)

    def test_rejects_duplicate_and_inconsistent_results(self):
        for mutate in (lambda v:v['arms']['base']['results'].append(v['arms']['base']['results'][0]),
                       lambda v:v['arms']['base']['results'][0].__setitem__('failure_reason','bad'),
                       lambda v:v['arms']['base']['results'][1].__setitem__('elapsed_seconds',float('nan'))):
            value=copy.deepcopy(self.value); mutate(value)
            with self.assertRaises(ValueError): summarize_exploratory(value)
