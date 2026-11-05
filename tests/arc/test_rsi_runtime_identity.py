import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src.arc.guarded_contract import AUDITED_GENERATE_STEP
from src.arc.rsi_runtime_identity import snapshot_runtime_identity, verify_runtime_identity


class RuntimeIdentityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.python = Path(self.tmp.name) / 'python'
        self.python.write_bytes(b'fixture interpreter')
        self.python.chmod(0o700)
        self.digest = hashlib.sha256(self.python.read_bytes()).hexdigest()
        self.report = {'machine': 'arm64', 'python_version': '3.12.1',
            'interpreter_realpath': str(self.python.resolve()), 'interpreter_sha256': self.digest,
            'packages': {name: '1.0' for name in ('mlx', 'mlx-lm', 'transformers', 'tokenizers', 'numpy', 'huggingface-hub', 'jinja2')},
            'generate_step_sha256': AUDITED_GENERATE_STEP['0.30.7'], 'generate_module_sha256': 'a' * 64}
        self.report['packages']['mlx-lm'] = '0.30.7'

    def run_probe(self, **changes):
        report = {**self.report, **changes}
        with patch('src.arc.rsi_runtime_identity.subprocess.run', return_value=SimpleNamespace(returncode=0, stdout=json.dumps(report), stderr='')) as process:
            value = snapshot_runtime_identity(self.python)
            self.assertIn('-I', process.call_args.args[0])
            return value

    def test_reviewed_runtime_is_an_observation_not_authorization(self):
        result = self.run_probe()
        self.assertEqual(result['generate_step_sha256'], AUDITED_GENERATE_STEP['0.30.7'])
        with patch('src.arc.rsi_runtime_identity.subprocess.run', return_value=SimpleNamespace(returncode=0, stdout=json.dumps(self.report), stderr='')):
            self.assertFalse(verify_runtime_identity(self.python, result)['execution_ready'])

    def test_changed_source_version_or_machine_fails(self):
        for change in ({'generate_step_sha256': 'b' * 64}, {'machine': 'x86_64'},
                       {'packages': {**self.report['packages'], 'mlx-lm': 'other'}},
                       {'interpreter_sha256': 'f' * 64}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.run_probe(**change)

    def test_expected_manifest_drift_and_failed_probe_are_rejected(self):
        with patch('src.arc.rsi_runtime_identity.subprocess.run', return_value=SimpleNamespace(returncode=0, stdout=json.dumps(self.report), stderr='')):
            with self.assertRaises(ValueError):
                verify_runtime_identity(self.python, {**self.report, 'python_version': 'other'})
        with patch('src.arc.rsi_runtime_identity.subprocess.run', return_value=SimpleNamespace(returncode=1, stdout='', stderr='fixture')):
            with self.assertRaises(ValueError):
                snapshot_runtime_identity(self.python)
