import tempfile
import unittest
from pathlib import Path
from src.arc.lt_batch_journal import BatchJournal

class BatchJournalTests(unittest.TestCase):
    def test_resume_does_not_repeat_completed_tasks(self):
        with tempfile.TemporaryDirectory() as d:
            j=BatchJournal(Path(d), ['a','b'], 'hash')
            self.assertEqual(j.pending(2), ['a','b'])
            j.start('a'); j.finish('a', {'task_id':'a','status':'complete'})
            self.assertEqual(BatchJournal(Path(d), ['a','b'], 'hash').pending(2), ['b'])
            with self.assertRaises(ValueError): j.start('a')
    def test_interruption_and_partial_files_fail_closed(self):
        with tempfile.TemporaryDirectory() as d:
            j=BatchJournal(Path(d), ['a'], 'hash'); j.start('a')
            with self.assertRaises(ValueError): j.pending(1)
            (Path(d)/'a.result.json').write_text('{')
            with self.assertRaises(ValueError): j.pending(1)
    def test_wrong_identity_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            BatchJournal(Path(d), ['a'], 'hash')
            with self.assertRaises(ValueError): BatchJournal(Path(d), ['a'], 'other')
