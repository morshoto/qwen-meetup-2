import copy
import hashlib
import json
import unittest

from src.arc.lt_public import build_public_export, load_public_export


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


class PublicExportTests(unittest.TestCase):
    def setUp(self):
        self.challenge = {'a': {'train': [{'input': [[1]], 'output': [[2]]}],
                                'test': [{'input': [[3]]}]},
                          'excluded': {'train': [{'input': [[0]], 'output': [[0]]}],
                                       'test': [{'input': [[0]]}]}}
        self.receipt = {'comparison_sha256': 'a' * 64,
                        'selected_task_sha256': {'a': digest(self.challenge['a'])}}

    def test_export_contains_only_selected_public_task_and_loads_it(self):
        export = build_public_export(self.challenge, self.receipt)
        self.assertEqual(set(export['tasks']), {'a'})
        tasks = load_public_export(export, self.receipt)
        self.assertEqual(tasks['a'].prompt_data(), self.challenge['a'])
        self.assertEqual(export['comparison_sha256'], self.receipt['comparison_sha256'])

    def test_rejects_test_output_or_changed_task(self):
        changed = copy.deepcopy(self.challenge)
        changed['a']['test'][0]['output'] = [[9]]
        with self.assertRaises(ValueError): build_public_export(changed, self.receipt)
        export = build_public_export(self.challenge, self.receipt)
        export['tasks']['a']['test'][0]['input'] = [[8]]
        with self.assertRaises(ValueError): load_public_export(export, self.receipt)

    def test_rejects_missing_or_extra_task(self):
        export = build_public_export(self.challenge, self.receipt)
        export['tasks']['extra'] = self.challenge['excluded']
        with self.assertRaises(ValueError): load_public_export(export, self.receipt)
        with self.assertRaises(ValueError): build_public_export({}, self.receipt)


if __name__ == '__main__': unittest.main()
