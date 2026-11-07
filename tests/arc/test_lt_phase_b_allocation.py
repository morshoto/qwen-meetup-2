import unittest

from src.arc.lt_phase_b_allocation import audit_allocation


class AllocationTests(unittest.TestCase):
    def setUp(self):
        self.args = dict(training=['a', 'b'], universe={'a', 'b', 'c', 'd'},
                         phase_a={'c'}, development={'d'}, protected={'e'},
                         exposed={'f'}, official={'g'}, scheduled={'h'}, expected_count=2)

    def test_valid_allocation(self):
        self.assertEqual(audit_allocation(**self.args)['training_count'], 2)

    def test_rejects_each_exclusion_and_unknown_id(self):
        for group in ('phase_a', 'development', 'protected', 'exposed', 'official', 'scheduled'):
            with self.subTest(group=group):
                args = dict(self.args, **{group: {'a'}})
                with self.assertRaises(ValueError):
                    audit_allocation(**args)
        with self.assertRaises(ValueError):
            audit_allocation(**dict(self.args, training=['a', 'z']))

    def test_duplicate_or_wrong_count_fails_closed(self):
        for training in (['a', 'a'], ['a']):
            with self.assertRaises(ValueError):
                audit_allocation(**dict(self.args, training=training))


if __name__ == '__main__':
    unittest.main()
