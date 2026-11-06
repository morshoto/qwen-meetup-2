import unittest

from src.arc.lt_metal_observation import parse_metal_processes, compare_observations


class MetalObservationTests(unittest.TestCase):
    def test_parses_only_exact_service_and_rejects_ambiguous_rows(self):
        service = '/System/Library/Frameworks/Metal.framework/Versions/A/XPCServices/MTLCompilerService.xpc/Contents/MacOS/MTLCompilerService'
        listing = f' 10 1 10 {service}\n 11 1 11 /bin/other\n'
        self.assertEqual(parse_metal_processes(listing), {10: {'ppid': 1, 'pgid': 10}})
        for malformed in (f' 10 1 10 {service}\n 10 2 10 {service}\n', f' broken 1 2 {service}\n'):
            with self.subTest(malformed=malformed), self.assertRaises(ValueError):
                parse_metal_processes(malformed)

    def test_comparison_reports_observations_without_attributing_cleanup(self):
        before = {1: {'ppid': 1, 'pgid': 1}}
        during = {**before, 7: {'ppid': 1, 'pgid': 7}, 8: {'ppid': 1, 'pgid': 8}}
        after = {**before, 8: {'ppid': 1, 'pgid': 8}}
        report = compare_observations(before, during, after)
        self.assertEqual(report['newly_observed'], [7, 8])
        self.assertEqual(report['newly_observed_still_listed'], [8])
        self.assertFalse(report['ownership_verified'])
        self.assertFalse(report['cleanup_verified'])
        self.assertFalse(report['execution_ready'])


if __name__ == '__main__': unittest.main()
