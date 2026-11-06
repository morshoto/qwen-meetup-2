import copy
import hashlib
import json
import unittest

from src.arc.lt_development import build_development_manifest
from src.arc.lt_development_export import build_development_export, load_development_export


def raw(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':')).encode()


class DevelopmentExportTests(unittest.TestCase):
    def setUp(self):
        self.tasks = {f't{i:03}': {'train': [{'input': [[1]], 'output': [[1]]}],
                                     'test': [{'input': [[2]]}]} for i in range(10)}
        self.challenges = raw(self.tasks)
        self.registry = raw({'status': 'bounded_exposure_proposed_unverified',
                             'task_source_sha256': hashlib.sha256(self.challenges).hexdigest(),
                             'known_exposed_task_ids': list(self.tasks),
                             'partitions': {'development': ['t000', 't001']}})
        self.manifest = build_development_manifest(list(self.tasks), known_exposed=set(self.tasks),
            reserved={'t000', 't001'}, dataset_sha256=hashlib.sha256(self.challenges).hexdigest(),
            exposure_sha256=hashlib.sha256(self.registry).hexdigest(), seed=20261111)

    def test_export_contains_only_selected_tasks_and_no_test_answers(self):
        export, receipt = build_development_export(self.challenges, self.registry, self.manifest)
        tasks = load_development_export(export, receipt)
        self.assertEqual(set(tasks), set(self.manifest['task_ids']))
        self.assertEqual(set(export['tasks']), set(self.manifest['task_ids']))
        self.assertTrue(all(set(pair) == {'input'} for item in export['tasks'].values() for pair in item['test']))
        self.assertFalse(receipt['execution_ready'])
        self.assertEqual(receipt['purpose'], 'development_only')

    def test_rejects_tampered_sources_manifest_and_export(self):
        with self.assertRaises(ValueError): build_development_export(self.challenges + b' ', self.registry, self.manifest)
        with self.assertRaises(ValueError): build_development_export(self.challenges, self.registry + b' ', self.manifest)
        manifest = copy.deepcopy(self.manifest)
        manifest['task_ids'][0] = 't000'
        with self.assertRaises(ValueError): build_development_export(self.challenges, self.registry, manifest)
        export, receipt = build_development_export(self.challenges, self.registry, self.manifest)
        export['tasks'][self.manifest['task_ids'][0]]['test'][0]['output'] = [[9]]
        with self.assertRaises(ValueError): load_development_export(export, receipt)


if __name__ == '__main__': unittest.main()
