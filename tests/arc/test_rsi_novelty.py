"""Conservative novelty screening over an explicitly bound reference pool."""
import hashlib
import unittest

from src.arc.ast_fingerprint import stable_ast_sha256
from src.arc.rsi_novelty import screen_novelty


CODE = 'def transform(grid):\n    return [row[::-1] for row in grid]'
OTHER = 'def transform(grid):\n    return grid[::-1]'
PROBE = 'a' * 64


def ref(code=OTHER, behavior='b' * 64, probes=PROBE):
    return {'source_sha256': hashlib.sha256(code.strip().encode()).hexdigest(),
            'ast_sha256': stable_ast_sha256(code), 'behavior_sha256': behavior,
            'probe_set_sha256': probes}


class NoveltyTests(unittest.TestCase):
    def test_unmatched_candidate_is_only_provisionally_novel(self):
        result = screen_novelty(CODE, 'c' * 64, PROBE, [ref()])
        self.assertEqual(result['status'], 'novel_unverified')
        self.assertFalse(result['eligible_for_training'])

    def test_duplicate_source_ast_and_behavior_are_rejected(self):
        self.assertEqual(screen_novelty(CODE, 'c' * 64, PROBE, [ref(CODE)])['status'], 'duplicate_source')
        same_ast = CODE + '\n# comment'
        self.assertEqual(screen_novelty(same_ast, 'c' * 64, PROBE, [ref(CODE)])['status'], 'duplicate_ast')
        self.assertEqual(screen_novelty(CODE, 'b' * 64, PROBE, [ref()])['status'], 'sampled_behavior_match')

    def test_mismatched_probe_sets_fail_closed(self):
        with self.assertRaisesRegex(ValueError, 'probe'):
            screen_novelty(CODE, 'b' * 64, PROBE, [ref(probes='f' * 64)])

    def test_rejects_missing_or_invalid_reference_fingerprints(self):
        with self.assertRaises(ValueError):
            screen_novelty(CODE, 'b' * 64, PROBE, [{'ast_sha256': 'no'}])
        with self.assertRaises(ValueError):
            screen_novelty(CODE, 'bad', PROBE, [])


if __name__ == '__main__':
    unittest.main()
