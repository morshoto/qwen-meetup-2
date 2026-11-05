"""Contract for constructing a proposed research split from audited IDs."""
import unittest

from src.arc.rsi_split import make_split


class RsiSplitTests(unittest.TestCase):
    def test_split_is_sorted_disjoint_and_excludes_exposed_development(self):
        result = make_split(['e', 'c', 'a', 'd', 'b'], {'e', 'b'}, development_size=2)
        self.assertEqual(result['development'], ['c', 'd'])
        self.assertEqual(result['generation'], ['a', 'b', 'e'])
        self.assertEqual(result, make_split(['a', 'b', 'c', 'd', 'e'], {'b', 'e'}, development_size=2))

    def test_rejects_insufficient_unexposed_tasks(self):
        with self.assertRaisesRegex(ValueError, 'eligible'):
            make_split(['a', 'b'], {'b'}, development_size=2)

    def test_rejects_duplicate_or_invalid_task_ids(self):
        for ids in (['a', 'a'], ['a', ''], ['a', 3]):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                make_split(ids, set(), development_size=1)

    def test_rejects_invalid_development_size(self):
        for size in (0, -1, True, 1.5):
            with self.subTest(size=size), self.assertRaises(ValueError):
                make_split(['a', 'b'], set(), development_size=size)

    def test_requires_a_nonempty_generation_partition(self):
        with self.assertRaisesRegex(ValueError, 'generation'):
            make_split(['a', 'b'], set(), development_size=2)

    def test_rejects_invalid_exposure_ids(self):
        with self.assertRaises(ValueError):
            make_split(['a', 'b'], {'a', ''}, development_size=1)


if __name__ == '__main__':
    unittest.main()
