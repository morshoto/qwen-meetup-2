import tempfile
import unittest
from pathlib import Path
from src.arc.data import Task, Verifier
from src.arc.lt_batch_execution import run_batch
from src.arc.lt_batch_journal import BatchJournal
from src.arc.lt_scoring import score_comparison

class BatchExecutionTests(unittest.TestCase):
    def test_resume_and_scoring_do_not_repeat_completed_task(self):
        task=Task('a', (([[1]], [[1]]),), ([[2]],))
        class Backend:
            def __init__(self): self.calls=[]
            def generate(self, prompt, *, kind, candidate):
                self.calls.append(candidate); self.raw='[[[2]]]'; self.stats={'prompt_tokens':2,'completion_tokens':3,'peak_memory_gb':1,'finish_reason':'stop'}
                return self.raw
        with tempfile.TemporaryDirectory() as d:
            journal=BatchJournal(Path(d), ['a'], 'frozen'); backend=Backend()
            self.assertEqual(run_batch(journal, {'a':task}, backend, '14b', 10), 1)
            self.assertEqual(run_batch(journal, {'a':task}, backend, '14b', 10), 0)
            self.assertEqual(backend.calls, [0,1])
            import json
            row=json.loads((Path(d)/'a.result.json').read_text())
            self.assertEqual(score_comparison(['a'],[row],Verifier({'a':[[[2]]]}))['models']['14b']['pass_at_2'],1)
    def test_interrupt_leaves_marker_and_does_not_retry(self):
        task=Task('a', (([[1]], [[1]]),), ([[2]],))
        class Backend:
            def generate(self, *args, **kwargs): raise KeyboardInterrupt()
        with tempfile.TemporaryDirectory() as d:
            journal=BatchJournal(Path(d), ['a'], 'frozen')
            with self.assertRaises(KeyboardInterrupt): run_batch(journal, {'a':task}, Backend(), '14b', 1)
            with self.assertRaises(ValueError): journal.pending(1)
    def test_timeout_is_recorded_without_fabricating_prediction(self):
        task=Task('a', (([[1]], [[1]]),), ([[2]],))
        class Backend:
            def generate(self, *args, **kwargs): raise TimeoutError('deadline')
        with tempfile.TemporaryDirectory() as d:
            journal=BatchJournal(Path(d), ['a'], 'frozen')
            self.assertEqual(run_batch(journal, {'a':task}, Backend(), '32b', 1), 1)
            import json
            row=json.loads((Path(d)/'a.result.json').read_text())
            self.assertEqual(row['candidates'], ['timeout','timeout'])
            self.assertIsNone(row['tokens'])
