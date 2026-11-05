from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.cli import e6g_audit
from src.pipelines.e6f import _verify_final

ROOT = Path(__file__).resolve().parents[2]


class SmokeAuditTests(unittest.TestCase):
    def test_sealed_smoke_provenance_and_budgets_are_audited_without_calls(self):
        run = ROOT/'artifacts/e6g/live-smoke'
        image = json.loads((run/'experiment_protocol.json').read_text())['sandbox_image_id']
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)/'audit'
            with patch.object(e6g_audit, 'OUTPUT', output), \
                 patch.object(e6g_audit, '_docker_image_id', return_value=image):
                e6g_audit.main()
            summary = json.loads((output/'summary.json').read_text())
            self.assertEqual(summary['model_calls'], 0)
            self.assertEqual(summary['sandbox_starts'], 0)
            self.assertEqual(summary['eligible_sft_records'], 2)
            self.assertEqual(summary['conditions']['G0_baseline']['requests_verified_against_http_bodies'], 6)
            self.assertEqual(summary['conditions']['G1_novelty_feedback']['requests_verified_against_http_bodies'], 1)
            self.assertEqual(summary['conditions']['G0_baseline']['corrected_behavior_evaluated_candidates'], 0)
            self.assertIsNone(summary['conditions']['G0_baseline']['corrected_sampled_behavior_suspect_rate'])
            self.assertEqual(summary['conditions']['G1_novelty_feedback']['corrected_sampled_behavior_suspect_rate'], 1.0)
            self.assertTrue(summary['current_pipeline_differs_from_historical'])
            self.assertFalse(summary['training_invoked'])
            self.assertNotIn('def transform', (output/'report.md').read_text())
            _verify_final(output)


if __name__ == '__main__':
    unittest.main()
