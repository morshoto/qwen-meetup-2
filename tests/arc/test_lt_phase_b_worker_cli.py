import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.cli.lt_phase_b_worker import main


class WorkerCLITests(unittest.TestCase):
    def test_worker_writes_once_after_registered_task_returns(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ('spec', 'inventory', 'export', 'receipt', 'allocation'):
                (root / name).write_text(json.dumps({'fixture': name}))
            args = ['worker', '--spec', str(root / 'spec'), '--inventory', str(root / 'inventory'),
                    '--export', str(root / 'export'), '--receipt', str(root / 'receipt'), '--allocation', str(root / 'allocation'),
                    '--output', str(root / 'out'), '--task', '00000000',
                    '--expected-run-sha256', 'a' * 64, '--model-path', '/fixture']
            result = {'task_id': '00000000', 'model': '14b', 'status': 'complete'}
            with patch('sys.argv', args), patch('src.cli.lt_phase_b_worker.run_registered_task', return_value=result) as run:
                main()
                self.assertEqual(json.loads((root / 'out').read_text()), result)
                self.assertEqual(run.call_args.kwargs['expected_run_sha256'], 'a' * 64)
                with self.assertRaises(FileExistsError): main()

    def test_worker_does_not_write_on_validation_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ('spec', 'inventory', 'export', 'receipt', 'allocation'):
                (root / name).write_text(json.dumps({'fixture': name}))
            args = ['worker', '--spec', str(root / 'spec'), '--inventory', str(root / 'inventory'),
                    '--export', str(root / 'export'), '--receipt', str(root / 'receipt'), '--allocation', str(root / 'allocation'),
                    '--output', str(root / 'out'), '--task', '00000000',
                    '--expected-run-sha256', 'a' * 64, '--model-path', '/fixture']
            with patch('sys.argv', args), patch('src.cli.lt_phase_b_worker.run_registered_task', side_effect=ValueError):
                with self.assertRaises(ValueError): main()
            self.assertFalse((root / 'out').exists())


if __name__ == '__main__': unittest.main()
