import subprocess
import sys
import unittest


class DeadlineWatchdogTests(unittest.TestCase):
    @unittest.skipIf(sys.platform == 'win32', 'POSIX libc test')
    def test_kills_parent_without_relying_on_parent_threads(self):
        script='''
import ctypes,os,subprocess,sys,time
p=subprocess.Popen([sys.executable,'-m','src.arc.deadline_watchdog',str(os.getpid()),str(time.monotonic()+0.4)],stdin=subprocess.PIPE)
print('ready',flush=True)
ctypes.PyDLL(None).sleep(5)
'''
        proc=subprocess.Popen([sys.executable,'-u','-c',script],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
        try:
            self.assertEqual(proc.stdout.readline().strip(),'ready')
            self.assertNotEqual(proc.wait(timeout=3),0)
        finally:
            if proc.poll() is None: proc.kill();proc.wait()
            if proc.stdout: proc.stdout.close()
            if proc.stderr: proc.stderr.close()

    def test_shortening_deadline_is_enforced(self):
        script="""
import os,subprocess,sys,time
p=subprocess.Popen([sys.executable,'-m','src.arc.deadline_watchdog',str(os.getpid()),str(time.monotonic()+5)],stdin=subprocess.PIPE)
p.stdin.write((str(time.monotonic()+0.3)+'\\n').encode());p.stdin.flush()
print('ready',flush=True)
time.sleep(5)
"""
        proc=subprocess.Popen([sys.executable,'-u','-c',script],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
        try:
            self.assertEqual(proc.stdout.readline().strip(),'ready')
            self.assertNotEqual(proc.wait(timeout=3),0)
        finally:
            if proc.poll() is None: proc.kill();proc.wait()
            if proc.stdout: proc.stdout.close()
            if proc.stderr: proc.stderr.close()

    def test_cancellation_disarms_watchdog(self):
        script='''
import os,subprocess,sys,time
p=subprocess.Popen([sys.executable,'-m','src.arc.deadline_watchdog',str(os.getpid()),str(time.monotonic()+0.4)],stdin=subprocess.PIPE)
p.stdin.write(b'cancel\\n');p.stdin.flush();p.stdin.close()
assert p.wait(timeout=2)==0
time.sleep(0.5)
'''
        result=subprocess.run([sys.executable,'-c',script],capture_output=True,text=True,timeout=3)
        self.assertEqual(result.returncode,0,result.stderr)
