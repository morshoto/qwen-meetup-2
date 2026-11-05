import json
import shutil
import tempfile
import unittest
from pathlib import Path

from src.arc.mlx_token_driver import adapter_sha256
from src.arc.rsi_model_identity import snapshot_model_manifest, verify_adapter_identity, verify_model_manifest

REV = 'a4d9b2df59d2c150bef02fcbe0d91046b7ca33a4'


class ModelIdentityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.repo = self.root / 'repo'
        self.model = self.repo / 'snapshots' / REV
        self.model.mkdir(parents=True)
        (self.model / 'config.json').write_text(json.dumps({'model_type': 'qwen3', 'quantization': {'bits': 4, 'group_size': 64}}))
        (self.model / 'tokenizer_config.json').write_text(json.dumps({'chat_template': 'fixture template'}))
        (self.model / 'tokenizer.json').write_text('{}')
        (self.model / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': {'a': 'one.safetensors', 'b': 'two.safetensors'}}))
        (self.model / 'one.safetensors').write_bytes(b'fixture one')
        (self.model / 'two.safetensors').write_bytes(b'fixture two')

    def test_manifest_binds_every_file_and_template(self):
        manifest = snapshot_model_manifest(self.model)
        self.assertEqual(manifest['quantization'], {'bits': 4, 'group_size': 64})
        self.assertEqual(set(manifest['files_sha256']), {p.name for p in self.model.iterdir()})
        result = verify_model_manifest(self.model, manifest)
        self.assertEqual(result['status'], 'local_model_files_consistent')
        self.assertFalse(result['execution_ready'])
        (self.model / 'two.safetensors').write_bytes(b'changed')
        with self.assertRaises(ValueError):
            verify_model_manifest(self.model, manifest)

    def test_missing_extra_or_inconsistent_weights_fail(self):
        (self.model / 'two.safetensors').unlink()
        with self.assertRaises(ValueError):
            snapshot_model_manifest(self.model)
        (self.model / 'two.safetensors').write_bytes(b'fixture two')
        (self.model / 'third.safetensors').write_bytes(b'extra')
        with self.assertRaises(ValueError):
            snapshot_model_manifest(self.model)

    def test_blob_symlinks_must_stay_in_repository(self):
        file = self.model / 'one.safetensors'
        file.unlink()
        blob = self.repo / 'blobs' / 'one'
        blob.parent.mkdir()
        blob.write_bytes(b'fixture one')
        file.symlink_to(blob)
        snapshot_model_manifest(self.model)
        file.unlink()
        outside = self.root / 'outside'
        outside.write_bytes(b'fixture')
        file.symlink_to(outside)
        with self.assertRaises(ValueError):
            snapshot_model_manifest(self.model)

    def test_shared_hub_blob_is_allowed_for_named_model_repository(self):
        hub = self.root / 'hub'
        repository = hub / 'models--mlx-community--Qwen3-14B-4bit'
        model = repository / 'snapshots' / REV
        shutil.copytree(self.model, model)
        blob = hub / 'blobs' / 'ab' / 'fixture'
        blob.parent.mkdir(parents=True)
        blob.write_bytes(b'fixture one')
        weight = model / 'one.safetensors'
        weight.unlink()
        weight.symlink_to(blob)
        manifest = snapshot_model_manifest(model)
        self.assertIn('one.safetensors', manifest['files_sha256'])

    def test_adapter_digest_matches_existing_driver_and_detects_drift(self):
        adapter = self.root / 'adapter'
        adapter.mkdir()
        (adapter / 'adapter_config.json').write_text('{}')
        (adapter / 'adapters.safetensors').write_bytes(b'fake weights')
        expected = adapter_sha256(adapter)
        self.assertFalse(verify_adapter_identity(adapter, expected)['execution_ready'])
        (adapter / 'adapters.safetensors').write_bytes(b'changed')
        with self.assertRaises(ValueError):
            verify_adapter_identity(adapter, expected)
        (adapter / 'adapters.safetensors').unlink()
        (adapter / 'adapters.safetensors').symlink_to(self.root / 'outside')
        with self.assertRaises(ValueError):
            verify_adapter_identity(adapter, expected)
