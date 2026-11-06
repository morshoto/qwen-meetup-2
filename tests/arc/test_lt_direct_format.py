import json
import unittest
from src.arc.lt_direct_format import normalize_direct

class DirectFormatTests(unittest.TestCase):
    def test_valid_formats(self):
        for value in ([[[1,2]]], [{'output': [[1,2]]}], {'outputs': [[[1,2]]]}):
            with self.subTest(value=value): self.assertEqual(normalize_direct(json.dumps(value), 1), [[[1,2]]])
    def test_rejects_ambiguous_and_invalid(self):
        for raw in ('[{"output":[[1]],"extra":1}]', '{"outputs":[[[1]]],"output":[[2]]}',
                    '[{"output":[[1]],"output":[[2]]}]', '[[[true]]]', '[[[10]]]',
                    '[{"output":[[1]]}', '```json\n[[[1]]]\n```', '[[[1]],[[2]]]'):
            with self.subTest(raw=raw), self.assertRaises(ValueError): normalize_direct(raw, 1)
