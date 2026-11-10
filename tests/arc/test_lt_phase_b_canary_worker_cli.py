import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.cli.lt_phase_b_canary_worker import main


class CanaryWorkerCLITests(unittest.TestCase):
    def test_canary_worker_writes_only_after_registered_execution(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ('canary', 'collection', 'inventory', 'allocation', 'receipt'):
                (root / name).write_text(json.dumps({'fixture': name}))
            args = ['worker', '--canary', str(root / 'canary'), '--collection', str(root / 'collection'),
                    '--inventory', str(root / 'inventory'), '--allocation', str(root / 'allocation'),
                    '--receipt', str(root / 'receipt'), '--config', str(root / 'config'),
                    '--evidence-dir', str(root / 'evidence'), '--output', str(root / 'output'),
                    '--expected-canary-sha256', 'a' * 64, '--expected-collection-sha256', 'b' * 64]
            result = {'status': 'tensor_change_consistent_unverified', 'training_verified': False}
            with patch('sys.argv', args), patch('src.cli.lt_phase_b_canary_worker.run_registered_canary', return_value=result):
                main()
                self.assertEqual(json.loads((root / 'output').read_text()), result)
                with self.assertRaises(FileExistsError): main()

    def test_failure_does_not_create_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ('canary', 'collection', 'inventory', 'allocation', 'receipt'):
                (root / name).write_text(json.dumps({'fixture': name}))
            args = ['worker', '--canary', str(root / 'canary'), '--collection', str(root / 'collection'),
                    '--inventory', str(root / 'inventory'), '--allocation', str(root / 'allocation'),
                    '--receipt', str(root / 'receipt'), '--config', str(root / 'config'),
                    '--evidence-dir', str(root / 'evidence'), '--output', str(root / 'output'),
                    '--expected-canary-sha256', 'a' * 64, '--expected-collection-sha256', 'b' * 64]
            with patch('sys.argv', args), patch('src.cli.lt_phase_b_canary_worker.run_registered_canary', side_effect=ValueError):
                with self.assertRaises(ValueError): main()
            self.assertFalse((root / 'output').exists())


if __name__ == '__main__': unittest.main()
