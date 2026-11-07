import unittest

from src.arc.lt_phase_b_comparison_selection import propose_comparison


class ComparisonSelectionTests(unittest.TestCase):
    def setUp(self):
        self.universe = {f'{x:08x}' for x in range(1000)}
        ids = sorted(self.universe)
        self.groups = {'training': set(ids[:100]), 'phase_a': set(ids[100:300]),
                       'development': set(ids[300:303]), 'protected': set(ids[303:553]),
                       'scheduled': set(ids[553:573]), 'exposed': set(ids[573:590]),
                       'official': {f'official-{x}' for x in range(120)}}

    def test_fixed_selection_excludes_all_reserved_groups(self):
        result = propose_comparison(self.universe, **self.groups)
        self.assertEqual(len(result['task_ids']), 50)
        self.assertEqual(result['task_ids'], sorted(result['task_ids']))
        self.assertEqual(len(set(result['task_ids'])), 50)
        for group in self.groups.values(): self.assertFalse(set(result['task_ids']) & group)
        self.assertEqual(result, propose_comparison(set(reversed(sorted(self.universe))), **self.groups))
        self.assertEqual(result['status'], 'proposal_not_frozen')

    def test_fails_on_missing_or_invalid_inventory(self):
        for name in ('training', 'phase_a', 'development', 'protected', 'scheduled', 'official'):
            bad = dict(self.groups, **{name: set()})
            with self.subTest(name=name), self.assertRaises(ValueError):
                propose_comparison(self.universe, **bad)
        bad = dict(self.groups, training=self.groups['training'] | {'outside'})
        with self.assertRaises(ValueError): propose_comparison(self.universe, **bad)
        bad = dict(self.groups, exposed=set(self.universe))
        with self.assertRaises(ValueError): propose_comparison(self.universe, **bad)


if __name__ == '__main__': unittest.main()
