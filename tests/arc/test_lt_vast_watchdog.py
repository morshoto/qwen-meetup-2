import json
import subprocess
import unittest
from unittest.mock import patch
from types import SimpleNamespace


class VastWatchdogTests(unittest.TestCase):
    def test_targets_one_instance_and_verifies_absence(self):
        from src.cli.lt_vast_watchdog import destroy_and_verify
        results = [SimpleNamespace(returncode=0, stdout='{}'),
                   SimpleNamespace(returncode=0, stdout=json.dumps([{'id': 44}]))]
        with patch('src.cli.lt_vast_watchdog.subprocess.run', side_effect=results) as run:
            self.assertTrue(destroy_and_verify(123, attempts=1))
        self.assertEqual(run.call_args_list[0].args[0], ['vastai', 'destroy', 'instance', '123', '--yes', '--raw'])
        self.assertEqual(run.call_args_list[1].args[0], ['vastai', 'show', 'instances', '--raw'])

    def test_retries_when_target_still_exists(self):
        from src.cli.lt_vast_watchdog import destroy_and_verify
        results = [SimpleNamespace(returncode=0, stdout='{}'),
                   SimpleNamespace(returncode=0, stdout='[{"id":123}]'),
                   SimpleNamespace(returncode=0, stdout='{}'),
                   SimpleNamespace(returncode=0, stdout='[]')]
        with patch('src.cli.lt_vast_watchdog.subprocess.run', side_effect=results), \
             patch('src.cli.lt_vast_watchdog.time.sleep'):
            self.assertTrue(destroy_and_verify(123, attempts=2))

    def test_failure_and_invalid_list_are_not_success(self):
        from src.cli.lt_vast_watchdog import destroy_and_verify
        for results in ([SimpleNamespace(returncode=1, stdout='')],
                        [SimpleNamespace(returncode=0, stdout='{}'), SimpleNamespace(returncode=0, stdout='{}')],
                        [subprocess.TimeoutExpired('vastai', 5)]):
            with self.subTest(results=results), patch('src.cli.lt_vast_watchdog.subprocess.run', side_effect=results):
                self.assertFalse(destroy_and_verify(123, attempts=1))

    def test_rejects_invalid_instance_id(self):
        from src.cli.lt_vast_watchdog import destroy_and_verify
        for value in (0, -1, '123', True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                destroy_and_verify(value)
