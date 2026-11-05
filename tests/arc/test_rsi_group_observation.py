"""Process listings are observations, not proof of containment."""
import unittest
from src.arc.rsi_group_observation import parse_group_listing


class GroupObservationTests(unittest.TestCase):
    def test_distinguishes_zombies_and_live_members(self):
        result=parse_group_listing('100 100 Z\n101 100 S\n200 200 R\n',100)
        self.assertEqual(result['members'],[{'pid':100,'state':'Z'},{'pid':101,'state':'S'}])
        self.assertEqual(result['live_member_pids'],[101])
        self.assertFalse(result['group_cleanup_verified'])
        self.assertFalse(result['escaped_processes_excluded'])
    def test_rejects_missing_leader_duplicate_or_malformed(self):
        for listing in ('101 100 S\n','100 100 Z\n100 100 S\n','100 100 ?!\n','100 x Z\n'):
            with self.subTest(listing=listing),self.assertRaises(ValueError):parse_group_listing(listing,100)
