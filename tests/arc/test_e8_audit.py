import tempfile
import json
import shutil
import unittest
from unittest.mock import patch
from pathlib import Path
from src.cli.e8_audit import audit
from src.cli.e8_report import report
from src.arc.e8_config import ORDER, load_e8_config
from src.pipelines.e8 import run_stage
from src.pipelines.e6g import _seal_g, _verify_g
from src.pipelines.e5 import _seal_mutable_file


class E8AuditTests(unittest.TestCase):
    def test_live_evidence_when_available(self):
        campaign = Path('artifacts/e8/live-smoke-20261101')
        if not (campaign / '8b' / 'seal.json').exists():
            self.skipTest('local sealed Live evidence unavailable')
        with tempfile.TemporaryDirectory(dir='artifacts/e8') as tmp:
            out = Path(tmp) / 'audit'
            with self.assertRaisesRegex(ValueError, 'dependency differs'):
                audit(campaign, out)
            historical = Path('artifacts/e8/audit-live-32b-20261101')
            _verify_g(historical)
            self.assertEqual(json.loads((historical / 'summary.json').read_text())['stages']['8b']['generation_requests_verified'], 6)

    def test_pool_tampering_fails_even_with_new_sidecar(self):
        with tempfile.TemporaryDirectory(dir='artifacts/e8') as tmp:
            root = Path(tmp) / 'fake'
            settings = load_e8_config('configs/e8.fake.yml')
            for stage in ORDER: run_stage(settings, stage, root)
            pool = root / 'initial_pool.json'
            data = json.loads(pool.read_text())
            data[0]['origin'] = 'tampered'
            pool.write_text(json.dumps(data))
            _seal_mutable_file(pool)
            with self.assertRaisesRegex(ValueError, 'seed pool'):
                audit(root, Path(tmp) / 'audit')

    def test_resealed_wrong_stage_summary_is_rejected(self):
        original = Path('artifacts/e8/live-smoke-20261101')
        if not (original / '8b/seal.json').exists():
            self.skipTest('local sealed Live evidence unavailable')
        with tempfile.TemporaryDirectory(dir='artifacts/e8') as tmp:
            root = Path(tmp) / 'campaign'
            shutil.copytree(original, root)
            result_path = root / '8b/result.json'
            result = json.loads(result_path.read_text())
            manifest = {k: v for k, v in result['model_manifest'].items() if k != 'model_id'}
            result['valid_oracles'] += 1
            result_path.write_text(json.dumps(result))
            _seal_g(root / '8b')
            from src.pipelines.e6f import _sha
            pinned = json.loads((root / 'protocol.lock.yml').read_text())['dependencies']
            def fixture_hash(path):
                return pinned[str(path)] if str(path) in pinned else _sha(path)
            with patch('src.cli.e8_audit._model_manifest', return_value=manifest), \
                 patch('src.cli.e8_audit._sha', side_effect=fixture_hash):
                with self.assertRaisesRegex(ValueError, 'stage summary'):
                    audit(root, Path(tmp) / 'audit')

    def test_fake_report_and_source_seal(self):
        with tempfile.TemporaryDirectory(dir='artifacts/e8') as tmp:
            root = Path(tmp) / 'fake'
            settings = load_e8_config('configs/e8.fake.yml')
            for stage in ORDER: run_stage(settings, stage, root)
            out = Path(tmp) / 'report'
            result = report(root, out)
            self.assertEqual(len(result['models']), 3)
            self.assertTrue(all(r['status'] == 'completed' for r in result['models']))
            _verify_g(out)


if __name__ == '__main__': unittest.main()
