from __future__ import annotations

import json
import unittest
from pathlib import Path

from src.arc.response_diagnostics import diagnose_raw_response


ROOT = Path(__file__).resolve().parents[2]


class ResponseDiagnosticsTests(unittest.TestCase):
    def test_saved_synthetic_raw_responses_are_classified_offline(self):
        fixture = ROOT / 'tests/fixtures/p3b_response_diagnostics.jsonl'
        rows = [json.loads(line) for line in fixture.read_text(encoding='utf-8').splitlines()]
        self.assertGreaterEqual(len(rows), 8)
        for row in rows:
            with self.subTest(name=row['name']):
                result = diagnose_raw_response(
                    row['raw_body'], http_status=row['http_status'],
                    output_format=row['output_format'],
                    transport_error=row.get('transport_error'),
                    downstream_error=row.get('downstream_error'),
                )
                self.assertEqual(result['category'], row['expected_category'])
                if 'expected_code' in row:
                    self.assertEqual(result['intermediate']['code_extraction']['code'], row['expected_code'])
                if 'expected_json' in row:
                    self.assertEqual(result['intermediate']['json_extraction']['value'], row['expected_json'])

    def test_preserves_parser_intermediate_fields_and_redacts_thinking_from_final(self):
        row = json.loads((ROOT / 'tests/fixtures/p3b_response_diagnostics.jsonl').read_text().splitlines()[0])
        result = diagnose_raw_response(row['raw_body'], output_format='code')
        intermediate = result['intermediate']
        self.assertEqual(intermediate['thinking_open_count'], 1)
        self.assertEqual(intermediate['thinking_close_count'], 1)
        self.assertEqual(intermediate['end_tokens'], ['<|im_end|>'])
        self.assertNotIn('compare shapes', intermediate['final_answer'])
        self.assertEqual(intermediate['code_extraction']['status'], 'parsed')
        self.assertEqual(result['usage']['completion_tokens'], 20)
        self.assertEqual(result['finish_reason'], 'stop')

    def test_downstream_categories_are_supported_without_model_calls(self):
        cases = [
            ('sandbox_error', 'sandbox_error'),
            ('verifier_error', 'verifier_error'),
        ]
        raw = '{"choices":[{"message":{"content":"def transform(grid):\\n    return grid"},"finish_reason":"stop"}]}'
        for downstream, expected in cases:
            with self.subTest(category=expected):
                result = diagnose_raw_response(raw, output_format='code', downstream_error=downstream)
                self.assertEqual(result['category'], expected)


if __name__ == '__main__':
    unittest.main()
