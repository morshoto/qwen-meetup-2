import io
import json
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from src.cli.lt_phase_b_collection import main


class CollectionCLITests(unittest.TestCase):
    def test_cli_routes_only_explicit_inputs_and_prints_status(self):
        argv = ['collection', '--run-dir', '/run', '--expected-run-sha256', 'a' * 64,
                '--challenges', '/challenges', '--official-archive', '/archive',
                '--python', '/python', '--model-path', '/model', '--limit', '2']
        with patch('sys.argv', argv), patch('src.cli.lt_phase_b_collection.run_collection',
                return_value={'completed': 2, 'stop_reason': 'batch_limit'}) as run, redirect_stdout(io.StringIO()) as out:
            main()
        self.assertEqual(json.loads(out.getvalue()), {'completed': 2, 'stop_reason': 'batch_limit'})
        self.assertEqual(run.call_args.kwargs['expected_run_sha256'], 'a' * 64)
        self.assertEqual(run.call_args.kwargs['limit'], 2)


if __name__ == '__main__': unittest.main()
