import unittest

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


if __name__ == '__main__': unittest.main()
