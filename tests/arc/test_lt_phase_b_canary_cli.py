import io
import json
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from src.cli.lt_phase_b_canary import main


class CanaryCLITests(unittest.TestCase):
    def test_cli_routes_explicit_inputs_and_prints_status(self):
        args = ['canary', '--run-dir', '/run', '--expected-collection-sha256', 'a' * 64,
                '--expected-canary-sha256', 'b' * 64, '--challenges', '/challenges',
                '--official-archive', '/archive', '--python', '/python', '--model-path', '/model']
        result = {'status': 'training_observed_reload_pending', 'training_verified': False, 'promotion_allowed': False}
        with patch('sys.argv', args), patch('src.cli.lt_phase_b_canary.run_canary', return_value=result) as run, \
                redirect_stdout(io.StringIO()) as out:
            main()
        self.assertEqual(json.loads(out.getvalue()), result)
        self.assertEqual(run.call_args.kwargs['expected_canary_sha256'], 'b' * 64)
        self.assertEqual(run.call_args.kwargs['expected_collection_sha256'], 'a' * 64)


if __name__ == '__main__': unittest.main()
