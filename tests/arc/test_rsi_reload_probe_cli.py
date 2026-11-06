"""CLI preparation tests use a fake loader and never load model weights."""
import hashlib
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import Mock, patch

from src.cli.rsi_reload_probe import main


class ReloadProbeCliTests(unittest.TestCase):
    def test_cli_binds_manifest_bytes_before_requesting_runtime(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest=Path(tmp)/'model.json'; raw=b'{"model_id":"fixture"}'
            manifest.write_bytes(raw)
            args=['--model-key','32b','--model-dir','/model','--adapter-dir','/adapter',
                  '--adapter-sha256','a'*64,'--model-manifest',str(manifest),
                  '--model-manifest-sha256',hashlib.sha256(raw).hexdigest()]
            observed={'status':'model_reload_observed_unverified','reload_verified':False,
                      'training_allowed':False,'promotion_allowed':False}
            with patch('src.cli.rsi_reload_probe._runtime',return_value=(Mock(),Mock(),Mock(),Mock())) as runtime, \
                 patch('src.cli.rsi_reload_probe.observe_model_reload',return_value=observed) as probe, \
                 redirect_stdout(StringIO()) as out:
                main(args)
                self.assertEqual(json.loads(out.getvalue()),observed)
                self.assertEqual(probe.call_args.kwargs['model_manifest'],{'model_id':'fixture'})
                runtime.assert_called_once()
            args[-1]='0'*64
            with patch('src.cli.rsi_reload_probe._runtime') as runtime, self.assertRaises(ValueError): main(args)
            runtime.assert_not_called()
