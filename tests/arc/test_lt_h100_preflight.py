import importlib.metadata
import subprocess
import unittest
from unittest.mock import patch


class H100PreflightTests(unittest.TestCase):
    def test_reports_missing_compiler_before_gpu_checks(self):
        from src.cli.lt_h100_preflight import check
        with patch('src.cli.lt_h100_preflight.shutil.which', return_value=None):
            with self.assertRaisesRegex(RuntimeError, 'C compiler'):
                check('0.33.0')

    def test_rejects_compiler_that_cannot_build(self):
        from src.cli.lt_h100_preflight import check_compiler
        with patch('src.cli.lt_h100_preflight.shutil.which', return_value='/usr/bin/cc'), \
             patch('src.cli.lt_h100_preflight.subprocess.run', side_effect=subprocess.CalledProcessError(1, 'cc')):
            with self.assertRaisesRegex(RuntimeError, 'compile'):
                check_compiler()

    def test_compiles_and_runs_program(self):
        from src.cli.lt_h100_preflight import check_compiler
        with patch('src.cli.lt_h100_preflight.shutil.which', return_value='/usr/bin/cc'), \
             patch('src.cli.lt_h100_preflight.subprocess.run') as run:
            self.assertEqual(check_compiler(), '/usr/bin/cc')
            self.assertEqual(run.call_count, 2)
            self.assertTrue(all(c.kwargs['check'] for c in run.call_args_list))

    def test_rejects_missing_cuda_and_unsupported_bf16(self):
        from src.cli.lt_h100_preflight import check_cuda
        from types import SimpleNamespace
        for available, bf16, message in ((False, False, 'CUDA'), (True, False, 'BF16')):
            fake = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: available, is_bf16_supported=lambda: bf16))
            with self.subTest(message=message), self.assertRaisesRegex(RuntimeError, message):
                check_cuda(fake)

    def test_checks_registered_vllm_version(self):
        from src.cli.lt_h100_preflight import check_version
        with patch('src.cli.lt_h100_preflight.importlib.metadata.version', return_value='0.32.0'):
            with self.assertRaisesRegex(RuntimeError, 'vLLM version'):
                check_version('0.33.0')
        with patch('src.cli.lt_h100_preflight.importlib.metadata.version', side_effect=importlib.metadata.PackageNotFoundError('vllm')):
            with self.assertRaisesRegex(RuntimeError, 'vLLM'):
                check_version('0.33.0')
