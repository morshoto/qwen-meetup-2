import unittest
from dataclasses import replace
from src.arc.e7_config import load_e7_config
from src.arc.e7_protocol import ARMS, arm_metrics, feasibility, schedule, selection


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.s, _ = load_e7_config('configs/e7.cpu.fake.yml')

    def test_schedule_is_paired_and_balanced(self):
        entries = schedule(self.s)
        self.assertEqual([e['arm'] for e in entries[:4]], [ARMS[0], ARMS[1], ARMS[1], ARMS[0]])
        for i in range(self.s.slots):
            pair = [e for e in entries if e['slot'] == i]
            self.assertEqual(pair[0]['attempt_seeds'], pair[1]['attempt_seeds'])

    def test_incomplete_never_counted_as_failure(self):
        rows = [{'arm': x['arm'], 'slot': x['slot'], 'completion': 'completed',
                 'verified_novel': False} for x in schedule(self.s)]
        self.assertEqual(selection(rows, self.s)['decision'], 'no_gain_observed')
        rows[0]['completion'] = 'incomplete'
        self.assertEqual(selection(rows, self.s)['decision'], 'inconclusive_budget_or_missing_data')
        self.assertEqual(selection(rows, self.s)['paired_completed_slots'], 2)
        rows.append(rows[0])
        self.assertEqual(selection(rows, self.s)['decision'], 'protocol_invalid')

    def test_strict_gate_and_evidence_precedence(self):
        self.assertEqual(feasibility(strict_count=0, new_strict_count=0,
            selection_decision='inconclusive_budget_or_missing_data', resource_limited=True),
            'inconclusive_resource_limited')
        self.assertEqual(feasibility(strict_count=5, new_strict_count=5,
            selection_decision='no_gain_observed'), 'ready_for_gen0_to_gen1')
        self.assertEqual(feasibility(strict_count=5, new_strict_count=5,
            selection_decision='protocol_invalid'), 'inconclusive_protocol_or_evidence')
        with self.assertRaises(ValueError):
            feasibility(strict_count=0, new_strict_count=1, selection_decision='no_gain_observed')

    def test_behavior_matches_include_source_duplicates(self):
        row = {'arm': ARMS[1], 'completion': 'completed', 'oracle_valid': True,
               'novelty': 'duplicate_source', 'behavior_sha256': 'x', 'behavior_match': True}
        metrics = arm_metrics([row], ARMS[1], 1)
        self.assertEqual(metrics['sampled_behavior_suspects'], 1)
        self.assertEqual(metrics['behavior_fingerprint_denominator'], 1)

    def test_cannot_undercut_verifier(self):
        self.assertEqual(replace(self.s, candidate_sandbox=20).candidate_sandbox, 20)


if __name__ == '__main__':
    unittest.main()
