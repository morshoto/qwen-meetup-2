from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.cli import e6f_diagnose


class OfflineDiagnosticTests(unittest.TestCase):
    def test_sealed_six_candidates_are_classified_without_model_calls(self):
        class FixtureSandbox:
            def run(self, code, grid):
                raise RuntimeError('sandbox execution failed: fixture traceback\nIndexError: private')

        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / 'diagnosis'
            with patch.object(e6f_diagnose, 'OUTPUT', output), \
                 patch.object(e6f_diagnose, 'DockerSandbox', return_value=FixtureSandbox()):
                e6f_diagnose.main()
            summary = json.loads((output / 'summary.json').read_text())
            self.assertEqual(summary['classification_counts'], {'runtime_exception': 6})
            self.assertEqual(summary['model_calls'], 0)
            self.assertTrue(all(x['reproduced_exception_type'] == 'IndexError' for x in summary['failures']))
            self.assertNotIn('private', (output / 'summary.json').read_text())
            self.assertTrue((output / 'seal.json').is_file())
            self.assertEqual((output / 'private/attempt-0.log').stat().st_mode & 0o777, 0o600)


if __name__ == '__main__':
    unittest.main()
