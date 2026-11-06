import hashlib
import json
import unittest
from unittest.mock import patch

import yaml

from src.arc.rsi_preflight_bundle import prepare_preflight_bundle
from tests.arc.test_rsi_preflight_identity import PreflightIdentityTests


class PreflightBundleTests(unittest.TestCase):
    def setUp(self):
        self.fixture = PreflightIdentityTests('test_local_checks_bind_all_manifests_and_remain_unverified')
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root
        self.python = self.root / 'python'
        self.python.write_bytes(b'fixture interpreter')
        self.python.chmod(0o700)
        cli = self.root / 'src' / 'cli'
        cli.mkdir(parents=True)
        self.wrapper = cli / 'p3b_train.py'
        self.wrapper.write_bytes(b'fixture wrapper')
        self.args = dict(root=self.root, expected_spec_sha256=self.fixture.digest,
            manifest_bytes=self.fixture.raw, source_root=self.root, python_executable=self.python,
            **{key: self.fixture.fixture.args[key] for key in
               ('synthetic_tasks', 'forbidden_tasks', 'tokenizer', 'render', 'load_local_dataset')})

    @patch('src.arc.rsi_preflight_bundle.verify_preflight_local_identities')
    def test_bundle_binds_config_wrapper_and_offline_command(self, verify):
        verify.return_value = {'status': 'preflight_local_identities_consistent', 'execution_ready': False,
                               'independently_verified': False, 'proposal_sha256': self.fixture.digest}
        result = prepare_preflight_bundle(**self.args)
        directory = self.root / 'preflight.launch.quarantine'
        config_raw = (directory / 'mlx_lora.yml').read_bytes()
        launch = json.loads((directory / 'launch.json').read_text())
        spec = json.loads((self.root / 'preflight.proposal.json').read_text())
        self.assertEqual(yaml.safe_load(config_raw), spec['config'])
        self.assertEqual(launch['config_sha256'], hashlib.sha256(config_raw).hexdigest())
        self.assertEqual(launch['wrapper_sha256'], hashlib.sha256(b'fixture wrapper').hexdigest())
        self.assertEqual(launch['python_sha256'], hashlib.sha256(b'fixture interpreter').hexdigest())
        self.assertEqual(launch['argv'][1:5], ['-u', '-B', '-m', 'src.cli.p3b_train'])
        self.assertEqual(launch['argv'][-2:], ['--thinking-mode', 'off'])
        self.assertEqual(launch['environment']['HF_HUB_OFFLINE'], '1')
        self.assertFalse(result['execution_ready'])
        with self.assertRaises(FileExistsError):
            prepare_preflight_bundle(**self.args)

    @patch('src.arc.rsi_preflight_bundle.verify_preflight_local_identities')
    def test_invalid_identity_or_changed_proposal_fails_before_bundle(self, verify):
        verify.return_value = {'status': 'wrong'}
        with self.assertRaises(ValueError):
            prepare_preflight_bundle(**self.args)
        verify.return_value = {'status': 'preflight_local_identities_consistent', 'execution_ready': False,
                               'independently_verified': False, 'proposal_sha256': self.fixture.digest}
        (self.root / 'preflight.proposal.json').write_text('{}')
        with self.assertRaises(ValueError):
            prepare_preflight_bundle(**self.args)
        self.assertFalse((self.root / 'preflight.launch.quarantine').exists())

    @patch('src.arc.rsi_preflight_bundle.verify_preflight_local_identities')
    def test_changed_datasets_and_rehashed_settings_are_rejected(self, verify):
        verify.return_value = {'status': 'preflight_local_identities_consistent', 'execution_ready': False,
                               'independently_verified': False, 'proposal_sha256': self.fixture.digest}
        for relative in ('preflight-canary.quarantine/train.jsonl', 'generation-sft.quarantine/train.jsonl'):
            path = self.root / relative
            original = path.read_bytes()
            path.write_bytes(original + b' ')
            with self.assertRaises(ValueError):
                prepare_preflight_bundle(**self.args)
            path.write_bytes(original)
        path = self.root / 'preflight.proposal.json'
        spec = json.loads(path.read_text())
        spec['config']['learning_rate'] = 0.01
        path.write_text(json.dumps(spec, sort_keys=True, separators=(',', ':')) + '\n')
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        verify.return_value['proposal_sha256'] = digest
        with self.assertRaises(ValueError):
            prepare_preflight_bundle(**{**self.args, 'expected_spec_sha256': digest})
        self.assertFalse((self.root / 'preflight.launch.quarantine').exists())


if __name__ == '__main__':
    unittest.main()
