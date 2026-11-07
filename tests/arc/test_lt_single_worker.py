import unittest
from unittest.mock import patch


class SingleWorkerTests(unittest.TestCase):
    def test_vllm_worker_rejects_unregistered_version(self):
        from src.cli.lt_single_worker import make_backend
        with patch('src.cli.lt_single_worker.importlib.metadata.version', return_value='0.17.0'):
            with self.assertRaisesRegex(ValueError, 'runtime version'):
                make_backend('vllm', '0.16.0', '/model')

    def test_vllm_worker_selects_registered_backend(self):
        from src.cli.lt_single_worker import make_backend
        with patch('src.cli.lt_single_worker.importlib.metadata.version', return_value='0.16.0'), \
             patch('src.arc.lt_vllm_backend.VLLMBackend', return_value='backend') as backend:
            self.assertEqual(make_backend('vllm', '0.16.0', '/model'), 'backend')
            backend.assert_called_once_with('/model')
