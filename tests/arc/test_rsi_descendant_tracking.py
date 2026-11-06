import unittest
import subprocess
import sys
import time
from unittest.mock import patch

from src.arc.rsi_descendant_tracking import (DescendantTracker, parse_ps_listing, begin_owned_observation,
                                             sample_descendant_observation)


def row(pid, parent, started, state='S'):
    return {'pid': pid, 'ppid': parent, 'start': started, 'state': state}


class DescendantTrackingTests(unittest.TestCase):
    def test_observes_real_fixture_descendant_after_parent_exit(self):
        script = ('import subprocess,sys; '
                  'p=subprocess.Popen([sys.executable,"-c","import time; time.sleep(2)"], '
                  'stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL); '
                  'print(p.pid,flush=True); sys.stdin.readline()')
        process = subprocess.Popen([sys.executable, '-u', '-c', script], start_new_session=True,
                                   stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                   text=True)
        descendant = None
        try:
            descendant = int(process.stdout.readline().strip())
            tracker, initial = begin_owned_observation(process)
            self.assertIn(descendant, initial['live_observed_pids'])
            process.stdin.write('\n')
            process.stdin.flush()
            process.wait(timeout=3)
            observation = sample_descendant_observation(tracker)
            self.assertIn(descendant, observation['live_observed_pids'])
            self.assertIn(process.pid, observation['missing_observed_pids'])
            self.assertFalse(observation['cleanup_verified'])
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=3)
            process.stdin.close()
            process.stdout.close()
            # The fixture descendant has a fixed two-second lifetime and is never signalled.
            if descendant is not None:
                time.sleep(2.1)

    def test_parse_ps_start_time_and_reject_truncated_listing(self):
        rows = parse_ps_listing('100 1 Fri Nov 6 16:00:25 2026 Ss\n101 100 Fri Nov 6 16:00:26 2026 S\n')
        self.assertEqual(rows[1], row(101, 100, 'Fri Nov 6 16:00:26 2026', 'S'))
        for text in ('', '100 1 Fri Nov\n', '100 x Fri Nov 6 16:00:25 2026 S\n'):
            with self.subTest(text=text), self.assertRaises(ValueError): parse_ps_listing(text)

    def test_owned_process_start_is_observed_without_proving_cleanup(self):
        process = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(10)'],
                                   start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            tracker, initial = begin_owned_observation(process)
            self.assertIn(process.pid, initial['live_observed_pids'])
            self.assertEqual(initial['root_pgid'], process.pid)
            self.assertEqual(initial['root_session'], process.pid)
            self.assertFalse(initial['cleanup_verified'])
            self.assertIsInstance(tracker, DescendantTracker)
        finally:
            process.terminate()
            process.wait(timeout=5)

    def test_unavailable_ps_fails_without_signalling_child(self):
        process = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(10)'], start_new_session=True)
        try:
            with patch('src.arc.rsi_descendant_tracking.subprocess.check_output', side_effect=OSError('unavailable')):
                with self.assertRaises(ValueError): begin_owned_observation(process)
            self.assertIsNone(process.poll())
        finally:
            process.terminate()
            process.wait(timeout=5)

    def test_retains_observed_descendant_after_reparenting(self):
        tracker = DescendantTracker(root_pid=100, root_start='root')
        tracker.observe([row(100, 1, 'root'), row(101, 100, 'child')])
        result = tracker.observe([row(101, 1, 'child')])
        self.assertEqual(result['live_observed_pids'], [101])
        self.assertEqual(result['missing_observed_pids'], [100])
        self.assertFalse(result['cleanup_verified'])

    def test_pid_reuse_is_ambiguous_not_tracked_as_owned(self):
        tracker = DescendantTracker(root_pid=100, root_start='root')
        tracker.observe([row(100, 1, 'root'), row(101, 100, 'child')])
        result = tracker.observe([row(101, 1, 'other')])
        self.assertEqual(result['reused_pids'], [101])
        self.assertEqual(result['live_observed_pids'], [])
        self.assertFalse(result['cleanup_verified'])

    def test_incomplete_or_malformed_snapshot_is_rejected(self):
        tracker = DescendantTracker(root_pid=100, root_start='root')
        for rows in ([], [row(100, 1, 'wrong')], [row(100, 1, 'root'), row(100, 2, 'root')],
                     [row(100, 100, 'root')]):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                tracker.observe(rows)

    def test_no_snapshot_can_establish_complete_cleanup(self):
        tracker = DescendantTracker(root_pid=100, root_start='root')
        tracker.observe([row(100, 1, 'root')])
        result = tracker.observe([row(100, 1, 'root', 'Z')])
        self.assertEqual(result['live_observed_pids'], [])
        self.assertEqual(result['zombie_observed_pids'], [100])
        self.assertFalse(result['cleanup_verified'])
