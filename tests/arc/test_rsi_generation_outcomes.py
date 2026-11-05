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
