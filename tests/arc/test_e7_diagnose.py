import csv
import json
import tempfile
import unittest
from pathlib import Path

from src.cli.e7_diagnose import UNKNOWN, diagnose


class DiagnosisTests(unittest.TestCase):
    def test_sealed_historical_accounting(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'diagnostics'
            diagnose(Path('artifacts/e6g/live-smoke'), out)
            data = json.loads((out / 'diagnostics.json').read_text())
            self.assertEqual(data['candidate_totals'], {'G0_baseline-0': 10,
                             'G0_baseline-1': 3, 'G1_novelty_feedback-0': 20})
            with (out / 'attempts.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertTrue(all(row['input_digest'] == UNKNOWN for row in rows))
            self.assertEqual(data['strict_training_eligible_legacy_oracles'], 0)
            with self.assertRaises(FileExistsError):
                diagnose(Path('artifacts/e6g/live-smoke'), out)


if __name__ == '__main__':
    unittest.main()
