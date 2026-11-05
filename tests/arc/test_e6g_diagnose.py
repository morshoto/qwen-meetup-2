from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.cli import e6g_diagnose
from src.pipelines.e6f import _verify_final

ROOT = Path(__file__).resolve().parents[2]


class HistoricalAuditTests(unittest.TestCase):
    def test_historical_logs_are_classified_without_model_calls_or_raw_output(self):
        class FixtureSandbox:
            def __init__(self, **kwargs): pass
            def run(self, code, grid): return grid

        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / 'diagnosis'
            with patch.object(e6g_diagnose, 'OUTPUT', output), \
                 patch.object(e6g_diagnose, 'DockerSandbox', FixtureSandbox):
                e6g_diagnose.main()
            summary = json.loads((output/'summary.json').read_text())
            self.assertEqual(summary['model_calls'], 0)
            self.assertEqual(summary['duplicate_collection_candidates'], 4)
            self.assertEqual(summary['duplicate_collection_candidates_matching_seed_ast'], 4)
            self.assertGreater(summary['observations'], 4)
            self.assertIn('e6f_collection', summary['legacy_identity_incomplete_runs'])
            self.assertNotIn('generated_text\"', (output/'summary.json').read_text())
            self.assertEqual((output/'duplicates.csv').stat().st_mode & 0o777, 0o600)
            _verify_final(output)

    def test_seed_and_stage_review_preserve_legacy_limit(self):
        pool = ROOT/'artifacts/e6g/seed-pool'
        pool_v2 = ROOT/'artifacts/e6g/seed-pool-v2'
        review = ROOT/'artifacts/e6g/diagnosis-review'
        _verify_final(pool); _verify_final(pool_v2); _verify_final(review)
        self.assertEqual((pool/'private/pool.jsonl').read_bytes(), (pool_v2/'private/pool.jsonl').read_bytes())
        manifest = json.loads((pool/'pool_manifest.json').read_text())
        self.assertEqual(manifest['eligible_seed_oracles'], 2)
        self.assertTrue(manifest['legacy_identity_incomplete'])
        self.assertEqual(sum(x['historical_duplicate_observed'] for x in manifest['source_status']), 1)
        summary = json.loads((review/'summary.json').read_text())
        self.assertEqual(summary['classification_counts'],
                         {'same_model_text': 3, 'ast_normalization_collapsed_difference': 1})
        provenance = ROOT/'artifacts/e6g/provenance-audit-v2'
        _verify_final(provenance)
        binding = json.loads((provenance/'summary.json').read_text())
        self.assertEqual(binding['response_counts'], {'verified': 66})
        self.assertEqual(binding['seed_counts'], {'verified': 2})

    def test_stage_audit_distinguishes_sandbox_and_registration(self):
        stage = ROOT/'artifacts/e6g/stage-audit'
        _verify_final(stage)
        with (stage/'stages.csv').open(newline='') as stream:
            rows = list(csv.DictReader(stream))
        passed = next(r for r in rows if r['run'] == 'e6f_collection' and
                      r['candidate_id'] == '5' and r['generation_attempt'] == '15')
        self.assertEqual(passed['sandbox_stage'], 'passed')
        self.assertEqual(passed['registration_stage'], 'duplicate_oracle')
        repeat = next(r for r in rows if r['run'] == 'e6f_collection' and
                      r['candidate_id'] == '0' and r['generation_attempt'] == '2')
        self.assertEqual(repeat['sandbox_stage'], 'not_executed_for_repeated_code')
        summary = json.loads((stage/'summary.json').read_text())
        self.assertEqual(summary['observations'], 66)
        self.assertEqual(summary['model_calls'], 0)
        self.assertEqual(summary['sandbox_starts'], 0)


if __name__ == '__main__':
    unittest.main()
