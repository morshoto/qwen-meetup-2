"""Monitor a reload-like fixture without running a model or asserting provenance."""
import json
import os
import subprocess
import sys
import unittest

from tests.arc.test_rsi_process_monitor import state
from src.arc.rsi_reload_stream import watch_reload_stream, review_reload_observation


def child(script):
    return subprocess.Popen([sys.executable, '-u', '-c', script], start_new_session=True,
                            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)


class ReloadStreamTests(unittest.TestCase):
    def test_identity_failure_stops_reload_fixture(self):
        process = child('import time; time.sleep(10)')
        try:
            result = watch_reload_stream(process, baseline=state(), sampler=lambda _: state(),
                identity_sampler=lambda _: None, timeout_seconds=3, interval_seconds=0.05,
                sample_timeout_seconds=0.2, terminate_grace_seconds=0.1)
            self.assertEqual(result['stop_reason'], 'identity_monitor_unavailable')
            self.assertFalse(result['reload_verified'])
        finally:
            if process.poll() is None:
                os.killpg(process.pid, 9)
                process.wait()
            process.stdout.close()

    def watch(self, script, *, sampler=None, limits=None):
        process=child(script)
        self.addCleanup(process.stdout.close)
        try:
            baseline=state()
            result=watch_reload_stream(process, baseline=baseline, sampler=sampler or (lambda _: state()),
                timeout_seconds=3, interval_seconds=0.05, sample_timeout_seconds=0.2,
                terminate_grace_seconds=0.1, resource_limits=limits)
            return baseline,result
        finally:
            if process.poll() is None:
                os.killpg(process.pid,9); process.wait()

    def test_complete_valid_output_remains_unverified(self):
        output={'status':'model_reload_observed_unverified','model_key':'14b','matched_tensor_count':28,
                'adapter_sha256':'a'*64,'reload_verified':False,'training_allowed':False,'promotion_allowed':False,
                'evidence_type':'caller_supplied_runtime_and_local_files'}
        baseline,result=self.watch('import time; print('+repr(json.dumps(output))+',flush=True); time.sleep(0.1)')
        # Exercise the review using fixture evidence; this is not containment proof.
        review_fixture=dict(result,group_signals_confirmed=True)
        checked=review_reload_observation(baseline=baseline,observation=review_fixture,model_key='14b',adapter_sha256='a'*64)
        self.assertEqual(checked['status'],'reload_observation_consistent_unverified')
        self.assertFalse(checked['reload_verified']);self.assertFalse(checked['training_allowed'])
        self.assertFalse(result['escaped_processes_excluded'])
        for flag in ('reload_verified', 'training_allowed', 'promotion_allowed'):
            altered = dict(output, **{flag: 0})
            changed = dict(review_fixture, output=json.dumps(altered))
            self.assertEqual(review_reload_observation(baseline=baseline,observation=changed,model_key='14b',
                             adapter_sha256='a'*64)['status'],'reload_observation_rejected')
        self.assertEqual(review_reload_observation(baseline=baseline,observation=review_fixture,model_key='32b',
                         adapter_sha256='a'*64)['status'],'reload_observation_rejected')

    def test_overflow_and_resource_stop_are_not_accepted(self):
        baseline,result=self.watch("import time; print('x'*70000,flush=True); time.sleep(2)")
        self.assertFalse(result['stream_complete'])
        self.assertEqual(review_reload_observation(baseline=baseline,observation=result,model_key='14b',
                         adapter_sha256='a'*64)['status'],'reload_observation_rejected')
        limits={'timeout_seconds':3,'minimum_disk_gib':20,'minimum_free_memory_percent':60,
                'maximum_swap_growth_mib':512,'maximum_process_rss_gib':32,'sample_interval_seconds':1}
        baseline,result=self.watch('import time; time.sleep(2)',sampler=lambda _:state(free=55),limits=limits)
        self.assertEqual(result['stop_reason'],'memory_pressure_limit')
        self.assertFalse(result['reload_verified'])
