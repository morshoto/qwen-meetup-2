"""Resource sampling covers descendants and fails on incomplete process lists."""
import unittest
from pathlib import Path
from unittest.mock import patch

from src.arc.rsi_process_sampling import process_tree_rss_bytes, sample_process


class ProcessSamplingTests(unittest.TestCase):
    def test_sums_only_root_and_descendants(self):
        listing = '100 1 100\n101 100 50\n102 101 25\n200 1 999\n'
        self.assertEqual(process_tree_rss_bytes(listing, 100), 175 * 1024)

    def test_missing_duplicate_or_malformed_process_rows_fail_closed(self):
        for listing in ('1 0 12\n', '100 1 12\n100 1 13\n', '100 1 nope\n',
                        '100 102 1\n101 100 1\n102 101 1\n'):
            with self.subTest(listing=listing), self.assertRaises(ValueError):
                process_tree_rss_bytes(listing, 100)

    def test_sample_binds_host_metrics_and_process_tree(self):
        host = {'swap_used_mib': 0.0, 'memory_free_percent': 80, 'disk_free_bytes': 30 * 1024**3}
        with patch('src.arc.rsi_process_sampling.snapshot', return_value=host), \
             patch('src.arc.rsi_process_sampling.subprocess.check_output', return_value='100 1 10\n101 100 20\n'):
            result = sample_process(Path('.'), 100)
        self.assertEqual(result['process_tree_rss_bytes'], 30 * 1024)
        self.assertEqual(result['memory_free_percent'], 80)
        self.assertIsInstance(result['monotonic_seconds'], float)


if __name__ == '__main__':
    unittest.main()
