import json
import tempfile
import unittest
from pathlib import Path

from src.arc.e8_config import ORDER, load_e8_config
from src.cli.e8_full_audit import audit
from src.pipelines.e6g import _verify_g
from src.pipelines.e8 import run_stage


class FullAuditTests(unittest.TestCase):
    def test_fake_full_has_separate_slots_and_paired_validation(self):
        with tempfile.TemporaryDirectory(dir='artifacts/e8') as tmp:
            root = Path(tmp)
            smoke, full = root / 'smoke', root / 'full'
            for stage in ORDER:
                run_stage(load_e8_config('configs/e8.fake.yml'), stage, smoke)
                run_stage(load_e8_config('configs/e8.full.fake.yml'), stage, full)
            result = audit(full, smoke, root / 'audit')
            self.assertTrue(result['smoke_excluded'])
            self.assertEqual(result['slots']['32b']['slot_ids'], list(range(10, 20)))
            self.assertEqual(json.loads((full / '32b/result.json').read_text())['strict_new_oracles'], 0)
            _verify_g(root / 'audit')


if __name__ == '__main__': unittest.main()
