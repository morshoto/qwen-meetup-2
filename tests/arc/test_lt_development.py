import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from src.arc.lt_development import build_development_manifest, verify_development_manifest


class DevelopmentManifestTests(unittest.TestCase):
    def test_selection_is_deterministic_known_exposed_and_outside_reserved(self):
        ids = [f'task-{i:03}' for i in range(20)]
        kwargs = {'known_exposed': set(ids[:10]), 'reserved': {ids[0], ids[1]},
                  'dataset_sha256': 'a' * 64, 'exposure_sha256': 'b' * 64, 'seed': 20261111}
        proposal = build_development_manifest(ids, **kwargs)
        self.assertEqual(proposal, build_development_manifest(list(reversed(ids)), **kwargs))
        self.assertEqual(len(proposal['task_ids']), 3)
        self.assertTrue(set(proposal['task_ids']) <= kwargs['known_exposed'] - kwargs['reserved'])
        self.assertFalse(proposal['execution_ready'])
        verify_development_manifest(proposal, ids, **kwargs)

    def test_rejects_insufficient_exposures_duplicate_ids_and_changed_manifest(self):
        ids = ['a', 'b', 'c', 'd']
        kwargs = {'known_exposed': set(ids), 'reserved': {'a'}, 'dataset_sha256': 'a' * 64,
                  'exposure_sha256': 'b' * 64, 'seed': 1}
        proposal = build_development_manifest(ids, **kwargs)
        proposal['task_ids'][0] = 'a'
        with self.assertRaises(ValueError): verify_development_manifest(proposal, ids, **kwargs)
        with self.assertRaises(ValueError): build_development_manifest(ids + ['a'], **kwargs)
        with self.assertRaises(ValueError): build_development_manifest(ids, **{**kwargs, 'reserved': {'a', 'b'}})

    def test_cli_binds_challenges_and_exposure_registry_without_answers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = {f't{i:03}': {} for i in range(10)}
            challenges = root / 'challenges.json'
            challenges.write_text(json.dumps(data))
            registry = {'status': 'bounded_exposure_proposed_unverified',
                        'task_source_sha256': hashlib.sha256(challenges.read_bytes()).hexdigest(),
                        'known_exposed_task_ids': list(data), 'partitions': {'development': ['t000', 't001']}}
            exposure = root / 'exposure.json'
            exposure.write_text(json.dumps(registry))
            command = [sys.executable, '-m', 'src.arc.lt_development', '--challenges', str(challenges),
                       '--exposure', str(exposure), '--seed', '20261111']
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            manifest = json.loads(result.stdout)
            self.assertTrue(set(manifest['task_ids']) <= set(data) - {'t000', 't001'})
            self.assertEqual(manifest['exposure_sha256'], hashlib.sha256(exposure.read_bytes()).hexdigest())
            challenges.write_text(challenges.read_text() + ' ')
            changed = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(changed.returncode, 0)
            self.assertEqual(changed.stdout, '')



if __name__ == '__main__': unittest.main()
