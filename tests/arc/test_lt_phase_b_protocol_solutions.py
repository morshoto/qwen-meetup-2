import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from src.arc.lt_phase_b_protocol_gate import require_protocol_solutions


class FrozenSolutionsTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(); self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name); (self.root / 'configs').mkdir()
        self.solutions = b'{"training-task":[[[1]]]}'
        self.protocol = {'protocol': 'lt-phase-b-exploratory', 'status': 'frozen', 'data': {'solution_sha256': hashlib.sha256(self.solutions).hexdigest()}}
        self.path = self.root / 'configs/lt.phase-b.protocol.json'
        self.write_protocol()

    def write_protocol(self):
        self.path.write_text(json.dumps(self.protocol))
        self.inventory = {'files': {'configs/lt.phase-b.protocol.json': hashlib.sha256(self.path.read_bytes()).hexdigest()}}

    def test_matches_source_bound_frozen_protocol(self):
        self.assertIsNone(require_protocol_solutions(self.root, self.inventory, self.solutions))

    def test_other_solutions_draft_or_changed_source_are_rejected(self):
        with self.assertRaises(ValueError): require_protocol_solutions(self.root, self.inventory, b'{}')
        self.protocol['status'] = 'draft_not_frozen'; self.write_protocol()
        with self.assertRaises(ValueError): require_protocol_solutions(self.root, self.inventory, self.solutions)
        self.protocol['status'] = 'frozen'; self.write_protocol()
        self.path.write_text('{}')
        with self.assertRaises(ValueError): require_protocol_solutions(self.root, self.inventory, self.solutions)


if __name__ == '__main__': unittest.main()
