"""Count historical failure categories without exposing private outputs."""
import unittest
from src.arc.rsi_historical_failures import summarize_historical_slots


class HistoricalFailuresTests(unittest.TestCase):
    def setUp(self):
        self.rows=[{'slot':0,'completion':'completed','status':'static_safety_rejected',
                    'attempts':[{'category':'static_safety_rejected','private_output':'do not copy'}],
                    'oracle_valid':False,'verified_novel':False,'raw_duplicate':False,'ast_duplicate':False},
                   {'slot':1,'completion':'completed','status':'passed','attempts':[{'category':'passed'}],
                    'oracle_valid':True,'verified_novel':False,'raw_duplicate':True,'ast_duplicate':True}]
    def test_counts_overlapping_flags_without_private_details(self):
        result=summarize_historical_slots(self.rows)
        self.assertEqual(result['slots'],2); self.assertEqual(result['strict_new'],0)
        self.assertEqual(result['slot_statuses'],{'passed':1,'static_safety_rejected':1})
        self.assertEqual(result['raw_duplicate_slots'],1); self.assertEqual(result['ast_duplicate_slots'],1)
        self.assertNotIn('do not copy',str(result)); self.assertFalse(result['training_eligible'])
    def test_rejects_duplicate_incomplete_and_invalid_rows(self):
        for rows in ([],[self.rows[0],self.rows[0]], [dict(self.rows[0],completion='interrupted')],
                     [dict(self.rows[0],verified_novel=True)], [dict(self.rows[0],attempts=[{}])]):
            with self.subTest(rows=rows),self.assertRaises(ValueError): summarize_historical_slots(rows)
