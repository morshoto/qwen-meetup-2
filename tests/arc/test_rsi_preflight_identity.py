import hashlib
import json
import unittest
from unittest.mock import patch

from src.arc.rsi_preflight_identity import verify_preflight_local_identities
from src.arc.rsi_preflight_spec import prepare_preflight_spec
from tests.arc.test_rsi_preflight_spec import PreflightSpecTests


class PreflightIdentityTests(unittest.TestCase):
    def setUp(self):
        self.fixture = PreflightSpecTests('test_binds_canary_configuration_and_fixed_limits')
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root
        self.model = self.fixture.args['model_dir']
        self.model.mkdir()
        (self.model / 'tokenizer.json').write_bytes(b'tokenizer')
        (self.model / 'tokenizer_config.json').write_bytes(b'configuration')
        self.manifests = {'model_manifest': {'fixture': 'model'},
            'runtime_manifest': {'interpreter_realpath': '/python', 'interpreter_sha256': 'a' * 64},
            'environment_manifest': {'interpreter_realpath': '/python', 'interpreter_sha256': 'a' * 64},
            'source_inventory': {'fixture': 'sources'}}
        self.raw = {key: json.dumps(value, sort_keys=True).encode() for key, value in self.manifests.items()}
        hashes = {key: hashlib.sha256(value).hexdigest() for key, value in self.raw.items()}
        hashes.update(tokenizer=hashlib.sha256(b'tokenizer').hexdigest(),
                      tokenizer_config=hashlib.sha256(b'configuration').hexdigest())
        result = prepare_preflight_spec(**{**self.fixture.args, 'identity_hashes': hashes})
        self.digest = result['sha256']

    def verify(self, **changes):
        args = dict(root=self.root, expected_spec_sha256=self.digest, manifest_bytes=self.raw,
                    source_root=self.root, python_executable='/python')
        args.update(changes)
        return verify_preflight_local_identities(**args)

    def test_local_checks_bind_all_manifests_and_remain_unverified(self):
        with patch('src.arc.rsi_preflight_identity.verify_model_manifest') as model, \
             patch('src.arc.rsi_preflight_identity.verify_runtime_identity') as runtime, \
             patch('src.arc.rsi_preflight_identity.verify_environment') as environment, \
             patch('src.arc.rsi_preflight_identity.verify_source_inventory') as sources:
            result = self.verify()
        self.assertEqual(result['status'], 'preflight_local_identities_consistent')
        self.assertFalse(result['execution_ready'])
        self.assertFalse(result['independently_verified'])
        model.assert_called_once()
        runtime.assert_called_once()
        environment.assert_called_once()
        sources.assert_called_once()

    def test_manifest_and_tokenizer_drift_are_rejected(self):
        changed = dict(self.raw, runtime_manifest=b'{}')
        with self.assertRaises(ValueError):
            self.verify(manifest_bytes=changed)
        (self.model / 'tokenizer.json').write_bytes(b'changed')
        with patch('src.arc.rsi_preflight_identity.verify_model_manifest'), \
             patch('src.arc.rsi_preflight_identity.verify_runtime_identity'), \
             patch('src.arc.rsi_preflight_identity.verify_environment'), \
             patch('src.arc.rsi_preflight_identity.verify_source_inventory'):
            with self.assertRaises(ValueError):
                self.verify()


if __name__ == '__main__':
    unittest.main()
