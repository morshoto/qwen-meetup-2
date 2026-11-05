import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from src.cli.e8_retry_prepare import prepare
from src.pipelines.e6g import _verify_g
from src.pipelines.e5 import _seal_mutable_file
from src.pipelines.e6f import _sha


def fixture_hash_for(source):
    pinned = json.loads((source / 'protocol.lock.yml').read_text())['dependencies']
    return lambda path: pinned[str(path)] if str(path) in pinned else _sha(path)


class RetryTests(unittest.TestCase):
    def test_requires_explicit_approval(self):
        with tempfile.TemporaryDirectory(dir='artifacts/e8') as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(ValueError, 'requires'):
                prepare(root / 'source', root / 'attempt', root / 'new',
                        live=False, seconds=1200, approve_retry=False)
            self.assertFalse((root / 'attempt').exists())

    def test_resealed_pool_tampering_is_rejected_before_download(self):
        original = Path('artifacts/e8/live-smoke-20261101')
        if not (original / '14b/seal.json').exists():
            self.skipTest('local 14B timeout evidence unavailable')
        with tempfile.TemporaryDirectory(dir='artifacts/e8') as tmp:
            root = Path(tmp)
            source = root / 'source'
            shutil.copytree(original, source)
            pool = source / 'initial_pool.json'
            value = json.loads(pool.read_text())
            value[0]['origin'] = 'tampered'
            pool.write_text(json.dumps(value))
            _seal_mutable_file(pool)
            with patch('src.cli.e8_retry_prepare.subprocess.Popen') as process:
                with self.assertRaisesRegex(ValueError, 'pool or verifier'):
                    prepare(source, root / 'attempt', root / 'campaign',
                            live=True, seconds=1200, approve_retry=True)
                process.assert_not_called()

    def test_fake_acquisition_clones_only_verified_8b(self):
        source = Path('artifacts/e8/live-smoke-20261101')
        if not (source / '14b/seal.json').exists():
            self.skipTest('local 14B timeout evidence unavailable')
        class Process:
            returncode = 0
            def poll(self): return 0
        state = {'memory_free_percent': 80, 'swap_used_mib': 0, 'disk_free_bytes': 100 * 1024**3}
        with tempfile.TemporaryDirectory(dir='artifacts/e8') as tmp:
            attempt, campaign = Path(tmp) / 'attempt', Path(tmp) / 'campaign'
            with patch('src.cli.e8_retry_prepare._check_no_other_model_server'), \
                 patch('src.cli.e8_retry_prepare.snapshot', return_value=state), \
                 patch('src.cli.e8_retry_prepare.subprocess.Popen', return_value=Process()), \
                 patch('src.cli.e8_retry_prepare._sha', side_effect=fixture_hash_for(source)), \
                 patch('src.cli.e8_retry_prepare._model_manifest', return_value={'fixture': True}):
                result = prepare(source, attempt, campaign, live=True, seconds=1200, approve_retry=True)
            self.assertEqual(result['status'], 'download_complete')
            self.assertFalse(result['inference_started'])
            self.assertFalse((campaign / '14b').exists())
            _verify_g(campaign / '8b'); _verify_g(attempt)
            self.assertEqual((source / 'protocol.sha256').read_text(), (campaign / 'protocol.sha256').read_text())

    def test_32b_retry_preserves_both_completed_stages(self):
        source = Path('artifacts/e8/live-smoke-retry-20261101')
        if not (source / '32b/seal.json').exists():
            self.skipTest('local 32B timeout evidence unavailable')
        class Process:
            returncode = 0
            def poll(self): return 0
        state = {'memory_free_percent': 80, 'swap_used_mib': 0, 'disk_free_bytes': 100 * 1024**3}
        with tempfile.TemporaryDirectory(dir='artifacts/e8') as tmp:
            attempt, campaign = Path(tmp) / 'attempt', Path(tmp) / 'campaign'
            with patch('src.cli.e8_retry_prepare._check_no_other_model_server'), \
                 patch('src.cli.e8_retry_prepare.snapshot', return_value=state), \
                 patch('src.cli.e8_retry_prepare.subprocess.Popen', return_value=Process()) as process, \
                 patch('src.cli.e8_retry_prepare._sha', side_effect=fixture_hash_for(source)), \
                 patch('src.cli.e8_retry_prepare._model_manifest', return_value={'fixture': True}):
                result = prepare(source, attempt, campaign, live=True, seconds=1200,
                                 approve_retry=True, stage='32b')
            self.assertEqual(result['status'], 'download_complete')
            self.assertIn('mlx-community/Qwen3-32B-4bit', process.call_args.args[0])
            for stage in ('8b', '14b'):
                _verify_g(campaign / stage)
                self.assertEqual((source / stage / 'seal.json').read_bytes(), (campaign / stage / 'seal.json').read_bytes())
            self.assertFalse((campaign / '32b').exists())
            _verify_g(attempt)


if __name__ == '__main__': unittest.main()
