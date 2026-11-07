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
    def test_start_callback_and_cleanup_on_callback_failure(self):
        seen=[]
        r=run_worker([sys.executable,'-c','print(1)'], timeout=5, on_start=seen.append)
        self.assertEqual(len(seen),1)
        self.assertTrue(r['group_absent'])
        def fail(pid):
            seen.append(pid)
            raise RuntimeError('cannot persist process')
        with self.assertRaises(RuntimeError): run_worker([sys.executable,'-c','import time; time.sleep(30)'],timeout=5,on_start=fail)
        import os
        with self.assertRaises(ProcessLookupError): os.killpg(seen[-1],0)
