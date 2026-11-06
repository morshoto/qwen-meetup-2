import unittest

from src.arc.rsi_comparison_usage import _summarize_usage


class ComparisonUsageTests(unittest.TestCase):
    def test_missing_and_partial_usage_are_not_imputed(self):
        events = [{'usage': None, 'token_reservation': 100},
                  {'usage': {'prompt_tokens': 3}, 'token_reservation': 100},
                  {'usage': {'prompt_tokens': 4, 'completion_tokens': 2, 'total_tokens': 6},
                   'token_reservation': 100}]
        result = _summarize_usage(events)
        self.assertEqual(result['complete_requests'], 1)
        self.assertEqual(result['prompt_tokens_reported'], 7)
        self.assertEqual(result['prompt_tokens_missing_requests'], 1)
        self.assertEqual(result['completion_tokens_missing_requests'], 2)
        self.assertIsNone(result['total_tokens_if_complete'])

    def test_invalid_or_over_budget_usage_fails(self):
        for usage in ({'prompt_tokens': True}, {'prompt_tokens': -1},
                      {'prompt_tokens': 4, 'completion_tokens': 2, 'total_tokens': 5},
                      {'total_tokens': 101}, {'completion_tokens': 513}):
            with self.subTest(usage=usage), self.assertRaises(ValueError):
                _summarize_usage([{'usage': usage, 'token_reservation': 100}])


if __name__ == '__main__':
    unittest.main()
