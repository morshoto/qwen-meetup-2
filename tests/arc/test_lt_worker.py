import unittest
from src.arc.data import Task
from src.arc.lt_worker import execute_task

class WorkerTests(unittest.TestCase):
    def test_two_calls_preserve_raw_tokens_and_grids(self):
        class Backend:
            def generate(self, prompt, *, kind, candidate):
                self.raw='[[[2]]]'; self.stats={'prompt_tokens':3,'completion_tokens':2,'peak_memory_gb':1.5,'finish_reason':'stop'}
                return self.raw
        row=execute_task(Task('a', (([[1]],[[1]]),), ([[2]],)), Backend(), '14b')
        self.assertEqual(row['candidates'],[[[[2]]],[[[2]]]])
        self.assertEqual(row['tokens'],10)
        self.assertEqual([x['candidate'] for x in row['calls']],[0,1])
    def test_timeout_is_explicit_and_unknown_costs_are_null(self):
        class Backend:
            def generate(self,*a,**k): raise TimeoutError('deadline')
        row=execute_task(Task('a', (([[1]],[[1]]),), ([[2]],)), Backend(), '14b')
        self.assertEqual(row['candidates'],['timeout','timeout'])
        self.assertIsNone(row['tokens'])
