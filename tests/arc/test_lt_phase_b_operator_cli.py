import io
import json
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from src.cli import lt_phase_b_training, lt_phase_b_real_reload


class OperatorCLITests(unittest.TestCase):
    def test_training_and_reload_forward_registered_paths(self):
        options = ['--run-dir', '/run', '--challenges', '/challenge', '--solutions', '/solution',
                   '--official-archive', '/archive', '--python', '/python', '--model-path', '/model',
                   '--expected-training-sha256', 'a'*64, '--expected-canary-sha256', 'b'*64,
                   '--expected-collection-sha256', 'c'*64]
        for module, name in ((lt_phase_b_training, 'run_training'), (lt_phase_b_real_reload, 'run_real_reload_dispatch')):
            with self.subTest(module=module.__name__), patch.object(sys, 'argv', ['cli', *options]), \
                 patch.object(module, name, return_value={'status': 'fixture'}) as call, redirect_stdout(io.StringIO()) as out:
                module.main()
                self.assertEqual(json.loads(out.getvalue()), {'status': 'fixture'})
                args = call.call_args.kwargs
                self.assertEqual(args['run_dir'], Path('/run'))
                self.assertEqual(args['solution_path'], Path('/solution'))
                self.assertEqual(args['expected_training_sha256'], 'a'*64)


if __name__ == '__main__': unittest.main()
