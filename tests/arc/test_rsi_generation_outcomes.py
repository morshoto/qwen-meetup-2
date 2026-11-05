"""Outcome reporting never turns a replay into an independent Oracle gate."""
import unittest
from unittest.mock import patch
from src.arc.rsi_generation_outcomes import summarize_generation_outcomes


class GenerationOutcomeTests(unittest.TestCase):
    def test_provisional_counts_are_separate_from_strict_oracles(self):
        replay={'status':'generation_replay_consistent','checked_slots':10,'training_allowed':False,
                'counts':{'novel_unverified':4,'duplicate_source':2,'sampled_behavior_match':1,
                          'prohibited_operation':2,'private_acceptance_rejected':1},
                'attempt_counts':{'novel_unverified':4,'duplicate_source':3,'sampled_behavior_match':1,
                                  'prohibited_operation':2,'private_acceptance_rejected':1,'syntax_error':2}}
        with patch('src.arc.rsi_generation_outcomes.audit_generation',return_value=replay):
            result=summarize_generation_outcomes(model_key='14b',root='fixture')
        self.assertEqual(result['slot_outcomes'],{'provisional_new':4,'duplicate_or_behavior_match':3,'invalid':3})
        self.assertEqual(result['attempt_outcomes'],{'provisional_new':4,'duplicate_or_behavior_match':4,'invalid':5})
        self.assertIsNone(result['independently_verified_new_oracles'])
        self.assertEqual(result['next_step'],'stop_before_training')
        self.assertFalse(result['training_allowed'])

    def test_five_provisional_candidates_still_require_independent_audit(self):
        replay={'status':'generation_replay_consistent','checked_slots':10,'training_allowed':False,
                'counts':{'novel_unverified':5,'repeated_code':5},
                'attempt_counts':{'novel_unverified':5,'repeated_code':5}}
        with patch('src.arc.rsi_generation_outcomes.audit_generation',return_value=replay):
            result=summarize_generation_outcomes(model_key='32b',root='fixture')
            self.assertEqual(result['next_step'],'independent_audit_required')
            self.assertFalse(result['training_allowed'])
            replay['counts']['unknown']=1
            with self.assertRaises(ValueError): summarize_generation_outcomes(model_key='32b',root='fixture')

    def test_incomplete_report_retains_completed_counts_and_unattributed_reservations(self):
        from src.arc.rsi_generation_outcomes import summarize_incomplete_generation
        report={'status':'generation_incomplete','completed_slots':2,
                'slots':[{'slot':0,'task_id':'task-a','status':'novel_unverified','evidence_sha256':'a'*64},
                         {'slot':1,'task_id':'task-b','status':'duplicate_ast','evidence_sha256':'b'*64}],
                'counts':{'novel_unverified':1,'duplicate_ast':1},
                'attempt_counts':{'novel_unverified':1,'duplicate_ast':1,'syntax_error':1},
                'failure_type':'TimeoutError','training_allowed':False,
                'budget_after':{'model_calls':4,'reserved_tokens':400,'sandbox_runs':10},
                'budget_usage':{'model_calls':4,'reserved_tokens':400,'sandbox_runs':10},
                'evidence_type':'caller_supplied_components_unverified'}
        result=summarize_incomplete_generation(model_key='14b',report=report)
        self.assertEqual(result['slot_outcomes'],{'provisional_new':1,'duplicate_or_behavior_match':1,'invalid':0})
        self.assertEqual(result['attempt_outcomes'],{'provisional_new':1,'duplicate_or_behavior_match':1,'invalid':1})
        self.assertEqual(result['unattributed_model_reservations'],1)
        self.assertEqual(result['uncompleted_slots'],8)
        self.assertFalse(result['training_allowed'])
        self.assertFalse(result['independently_verified'])
        bad=dict(report,counts={'novel_unverified':2})
        with self.assertRaises(ValueError): summarize_incomplete_generation(model_key='14b',report=bad)
        bad=dict(report,budget_usage=dict(report['budget_usage'],model_calls=2))
        with self.assertRaises(ValueError): summarize_incomplete_generation(model_key='14b',report=bad)
        bad=dict(report,slots=[dict(report['slots'][0],slot=1),report['slots'][1]])
        with self.assertRaises(ValueError): summarize_incomplete_generation(model_key='14b',report=bad)

    def test_incomplete_empty_evidence_never_permits_training(self):
        from src.arc.rsi_generation_outcomes import summarize_incomplete_generation
        report={'status':'generation_incomplete','completed_slots':0,'slots':[],'counts':{},'attempt_counts':{},
                'failure_type':'RuntimeError','training_allowed':False,'budget_after':None,'budget_usage':None,
                'evidence_type':'caller_supplied_components_unverified'}
        result=summarize_incomplete_generation(model_key='32b',report=report)
        self.assertEqual(result['uncompleted_slots'],10)
        self.assertIsNone(result['unattributed_model_reservations'])
        self.assertEqual(result['next_step'],'stop_before_training')
