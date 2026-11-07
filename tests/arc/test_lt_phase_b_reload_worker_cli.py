import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.cli.lt_phase_b_reload_worker import main


class ReloadWorkerCLITests(unittest.TestCase):
    def test_worker_writes_once_after_reload(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            names = ('canary', 'collection', 'inventory', 'allocation', 'receipt', 'summary', 'before', 'after')
            for name in names:
                (root / name).write_text(json.dumps({'fixture': name}))
            args = ['worker']
            for name in names:
                args += ['--' + name, str(root / name)]
            args += ['--config', str(root / 'config'), '--output', str(root / 'output'),
                     '--expected-canary-sha256', 'a' * 64, '--expected-collection-sha256', 'b' * 64]
            result = {'status': 'reload_observed_unverified', 'reload_verified': False}
            with patch('sys.argv', args), patch('src.cli.lt_phase_b_reload_worker.run_canary_reload', return_value=result):
                main()
                self.assertEqual(json.loads((root / 'output').read_text()), result)
                with self.assertRaises(FileExistsError): main()

    def test_failed_reload_does_not_create_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            names = ('canary', 'collection', 'inventory', 'allocation', 'receipt', 'summary', 'before', 'after')
            for name in names:
                (root / name).write_text(json.dumps({'fixture': name}))
            args = ['worker']
            for name in names:
                args += ['--' + name, str(root / name)]
            args += ['--config', str(root / 'config'), '--output', str(root / 'output'),
                     '--expected-canary-sha256', 'a' * 64, '--expected-collection-sha256', 'b' * 64]
            with patch('sys.argv', args), patch('src.cli.lt_phase_b_reload_worker.run_canary_reload', side_effect=ValueError):
                with self.assertRaises(ValueError): main()
            self.assertFalse((root / 'output').exists())


if __name__ == '__main__': unittest.main()
