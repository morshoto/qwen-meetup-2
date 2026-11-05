import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from src.cli.guarded_mlx_server import main


class GuardedServerCLITests(unittest.TestCase):
    def test_unreviewed_version_rejected_before_loading_model(self):
        with tempfile.TemporaryDirectory() as d:
            revision='a'*40; path=Path(d)/revision; path.mkdir()
            args=['guarded', '--model-dir',str(path),'--model-id','test','--revision',revision,
                  '--expected-mlx-lm-version','99.0','--expected-generate-step-sha256','f'*64,
                  '--ledger',str(Path(d)/'ledger.sqlite'),'--total-tokens','100',
                  '--max-prompt-tokens','10','--max-new-tokens','2']
            with patch.object(sys,'argv',args), patch('src.cli.guarded_mlx_server.importlib.metadata.version',return_value='99.0'), patch('src.cli.guarded_mlx_server.MLXTokenDriver') as driver:
                with self.assertRaises(SystemExit): main()
                driver.assert_not_called()

    def test_revision_must_match_snapshot_directory(self):
        with tempfile.TemporaryDirectory() as d:
            args=['guarded','--model-dir',d,'--model-id','test','--revision','a'*40,
                  '--expected-mlx-lm-version','0.30.7','--expected-generate-step-sha256','f'*64,
                  '--ledger',str(Path(d)/'ledger.sqlite'),'--total-tokens','100',
                  '--max-prompt-tokens','10','--max-new-tokens','2']
            with patch.object(sys,'argv',args), patch('src.cli.guarded_mlx_server.MLXTokenDriver') as driver:
                with self.assertRaises(SystemExit): main()
                driver.assert_not_called()
