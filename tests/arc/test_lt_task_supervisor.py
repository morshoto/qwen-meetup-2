import sys
import unittest
from src.arc.lt_task_supervisor import run_worker

class SupervisorTests(unittest.TestCase):
    def test_success_and_exit_status(self):
        r=run_worker([sys.executable,'-c','print("ok")'], timeout=5)
        self.assertEqual((r['status'],r['returncode'],r['stdout'].strip()),('complete',0,'ok'))
        self.assertTrue(r['leader_reaped'])
    def test_timeout_reaps_worker(self):
        r=run_worker([sys.executable,'-c','import time; time.sleep(30)'], timeout=.1)
        self.assertEqual(r['status'],'timeout')
        self.assertTrue(r['leader_reaped'])
    def test_nonzero_is_failure(self):
        r=run_worker([sys.executable,'-c','raise SystemExit(7)'], timeout=5)
        self.assertEqual((r['status'],r['returncode']),('error',7))
