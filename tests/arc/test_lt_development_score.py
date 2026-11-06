import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from src.arc.lt_development import build_development_manifest
from src.arc.lt_development_export import build_development_export


def raw(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':')).encode()


class DevelopmentScoringTests(unittest.TestCase):
    def test_offline_scoring_is_development_only_and_checks_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tasks = {f't{i:03}': {'train': [{'input': [[1]], 'output': [[1]]}],
                                   'test': [{'input': [[2]]}]} for i in range(10)}
            source = raw(tasks)
            registry = raw({'status': 'bounded_exposure_proposed_unverified',
                            'task_source_sha256': hashlib.sha256(source).hexdigest(),
                            'known_exposed_task_ids': list(tasks), 'partitions': {'development': ['t000']}})
            manifest = build_development_manifest(list(tasks), known_exposed=set(tasks), reserved={'t000'},
                dataset_sha256=hashlib.sha256(source).hexdigest(),
                exposure_sha256=hashlib.sha256(registry).hexdigest(), seed=20261111)
            _, receipt = build_development_export(source, registry, manifest)
            selected = manifest['task_ids'][0]
            prediction = {'development_manifest_sha256': receipt['development_manifest_sha256'],
                          'model': '14b', 'task_id': selected, 'status': 'complete',
                          'candidates': [[[[2]]], [[[0]]]], 'tokens': 4, 'runtime_seconds': 1,
                          'peak_memory_bytes': None, 'peak_swap_bytes': None}
            files = {'challenges': source, 'exposure': registry, 'manifest': raw(manifest),
                     'receipt': raw(receipt), 'solutions': raw({task_id: [[[2]]] for task_id in tasks}),
                     'results': raw(prediction) + b'\n'}
            for name, content in files.items(): (root / name).write_bytes(content)
            command = [sys.executable, '-m', 'src.arc.lt_development_score']
            for name in files: command += ['--' + name, str(root / name)]
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads(result.stdout)
            self.assertEqual(report['protocol'], 'lt-phase-a-development')
            self.assertFalse(report['claim_eligible'])
            self.assertEqual(report['models']['14b']['pass_at_1'], 1)
            self.assertEqual(report['models']['14b']['selected'], 3)
            self.assertEqual(report['models']['14b']['status'], 'incomplete')
            self.assertNotIn('solutions', str(report['rows']))
            receipt['selected_task_sha256'][selected] = '0' * 64
            (root / 'receipt').write_bytes(raw(receipt))
            failed = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(failed.returncode, 0)
            self.assertEqual(failed.stdout, '')
            (root / 'receipt').write_bytes(files['receipt'])
            prediction['task_id'] = 't000'
            (root / 'results').write_bytes(raw(prediction) + b'\n')
            excluded = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(excluded.returncode, 0)
            self.assertEqual(excluded.stdout, '')


if __name__ == '__main__': unittest.main()
