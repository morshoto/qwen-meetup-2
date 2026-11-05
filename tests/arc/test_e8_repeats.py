import tempfile
import unittest
from pathlib import Path
from src.cli.e8_repeats import recount
from src.pipelines.e6g import _verify_g


class E8RepeatAuditTests(unittest.TestCase):
    def test_live_repeats_when_evidence_available(self):
        source = Path('artifacts/e8/live-smoke-retry-20261101')
        if not (source / '14b/seal.json').exists():
            self.skipTest('local completed 14B evidence unavailable')
        with tempfile.TemporaryDirectory(dir='artifacts/e8') as tmp:
            output = Path(tmp) / 'audit'
            rows = recount(source, output)
            self.assertEqual((rows['8b']['emissions'], rows['8b']['repeated_source_emissions']), (6, 2))
            self.assertEqual((rows['14b']['emissions'], rows['14b']['repeated_source_emissions']), (2, 1))
            _verify_g(output)


if __name__ == '__main__': unittest.main()
