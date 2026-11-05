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

    def test_live_fixture_records_unreaped_leader_only_as_observation(self):
        import subprocess,sys,time
        from src.arc.rsi_group_observation import observe_exited_group
        from src.arc.rsi_process_monitor import _exited_unreaped
        p=subprocess.Popen([sys.executable,'-c','pass'],start_new_session=True)
        try:
            deadline=time.monotonic()+3
            while _exited_unreaped(p) is None and time.monotonic()<deadline: time.sleep(0.01)
            result=observe_exited_group(p)
            self.assertEqual(result['status'],'process_group_listing_unverified')
            self.assertIn(p.pid,[member['pid'] for member in result['members']])
            self.assertFalse(result['group_cleanup_verified'])
            self.assertFalse(result['escaped_processes_excluded'])
        finally:
            if p.poll() is None: p.kill()
            p.wait()
