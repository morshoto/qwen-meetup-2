import io
import unittest
from unittest.mock import MagicMock, patch

from src.arc.sandbox import DockerSandbox, IDENTITY_CODE


class SandboxInterruptTests(unittest.TestCase):
    def test_interrupt_attempts_to_stop_container_and_reap_client(self):
        process = MagicMock()
        process.stdin = io.BytesIO()
        process.stdout = io.BytesIO()
        process.stderr = io.BytesIO()
        process.wait.side_effect = [KeyboardInterrupt(), 0]
        with patch('src.arc.sandbox.shutil.which', return_value='/usr/bin/docker'), \
             patch('src.arc.sandbox.subprocess.Popen', return_value=process) as popen, \
             patch('src.arc.sandbox.subprocess.run') as run:
            with self.assertRaises(KeyboardInterrupt):
                DockerSandbox().run(IDENTITY_CODE, [[1]])
        name = popen.call_args.args[0][4]
        run.assert_called_once()
        self.assertEqual(run.call_args.args[0], ['/usr/bin/docker', 'kill', name])
        process.kill.assert_called_once()
        self.assertEqual(process.wait.call_count, 2)


if __name__ == '__main__': unittest.main()
