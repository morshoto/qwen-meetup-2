import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


class ProtocolDraftTests(unittest.TestCase):
    def test_draft_cannot_be_mistaken_for_execution_registration(self):
        protocol = json.loads((ROOT / 'configs/lt.phase-b.protocol.json').read_bytes())
        self.assertEqual(protocol['status'], 'draft_not_frozen')
        self.assertFalse(protocol['execution_ready'])
        self.assertFalse(protocol['training_authorized'])
        self.assertIn('scheduled_generation_slot_overlap', protocol['open_gates'])
        self.assertIn('source_runtime_and_model_reverification', protocol['open_gates'])
        self.assertEqual(protocol['model']['revision'], 'a4d9b2df59d2c150bef02fcbe0d91046b7ca33a4')
        self.assertEqual(protocol['model']['model_id'], 'mlx-community/Qwen3-14B-4bit')
        self.assertEqual(protocol['evaluation']['official_public_evaluation'], 'reserved_for_phase_c')
        self.assertEqual(protocol['evaluation']['comparison_status'], 'proposal_not_frozen')
        self.assertEqual(protocol['data']['training_pool_status'], 'conflict_unresolved')
        self.assertEqual(protocol['canary']['purpose'], 'synthetic_technical_only')

    def test_yield_gate_and_exploratory_limits_are_explicit(self):
        protocol = json.loads((ROOT / 'configs/lt.phase-b.protocol.json').read_bytes())
        self.assertEqual(protocol['training']['yield_policy']['zero'], 'insufficient_verified_self_solutions')
        self.assertEqual(protocol['training']['yield_policy']['one_or_two'], 'overfitting_only')
        self.assertEqual(protocol['training']['yield_policy']['three_or_more'], 'task_disjoint_train_validation')
        self.assertEqual(protocol['training']['maximum_optimizer_steps'], 20)
        self.assertEqual(protocol['inference']['candidate_count'], 2)
        self.assertEqual(protocol['inference']['thinking'], 'off')
        self.assertIsNone(protocol['data']['observed_eligible_task_count'])


if __name__ == '__main__': unittest.main()
