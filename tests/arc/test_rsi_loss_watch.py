import unittest

from src.arc.rsi_loss_watch import TrainingLossWatch


class LossWatchTests(unittest.TestCase):
    def test_incremental_finite_and_nonfinite_reports(self):
        watch = TrainingLossWatch()
        watch.feed(b'loading model\nIter 1: Train lo')
        self.assertEqual(watch.snapshot()['status'], 'ok')
        watch.feed(b'ss 0.123, Learning Rate 1e-4\n')
        self.assertEqual(watch.snapshot()['last_step'], 1)
        self.assertEqual(watch.snapshot()['last_loss'], 0.123)
        watch.feed(b'Iter 2: Train loss nan, Learning Rate 1e-4\n')
        self.assertEqual(watch.snapshot()['status'], 'nonfinite_loss')
        watch.feed(b'Iter 3: Train loss 0.2, Learning Rate 1e-4\n')
        self.assertEqual(watch.snapshot()['status'], 'nonfinite_loss')

    def test_malformed_duplicate_and_oversized_reports_fail_closed(self):
        for content in (b'Iter 1: Train loss ???, x\n', b'Iter 1: Train loss 1.0, x\nIter 1: Train loss 1.0, x\n',
                        b'x' * 16385):
            watch = TrainingLossWatch()
            watch.feed(content)
            self.assertEqual(watch.snapshot()['status'], 'unavailable')
        watch = TrainingLossWatch()
        watch.feed(b'Iter 1: Train loss 1')
        watch.finish()
        self.assertEqual(watch.snapshot()['status'], 'unavailable')


if __name__ == '__main__':
    unittest.main()
