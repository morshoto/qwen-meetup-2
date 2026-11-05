import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from src.arc.e8_config import MODELS, ORDER, load_e8_config
from src.arc.e8_resource import ResourceMonitor, check_resources
from src.pipelines.e8 import _model_manifest, run_stage, training_gate
from src.pipelines.e6g import _verify_g


class E8Tests(unittest.TestCase):
    def setUp(self):
        self.s = load_e8_config('configs/e8.fake.yml')
        Path('artifacts/e8').mkdir(parents=True, exist_ok=True)

    def test_profile_limits_and_equal_slots(self):
        self.assertEqual(self.s.slots, 2)
        self.assertEqual(self.s.candidate_sandbox, 30)
        self.assertEqual(tuple(MODELS), ('8b', '14b', '32b'))
        self.assertTrue(all(len(x[1]) == 40 for x in MODELS.values()))

    def test_full_slots_and_seeds_do_not_overlap_smoke(self):
        from src.pipelines.e8 import _protocol
        smoke = self.s
        full = load_e8_config('configs/e8.full.fake.yml')
        source = {'initial_pool': {}}
        a = _protocol(smoke, None, 'fake', source)
        b = _protocol(full, None, 'fake', source)
        self.assertEqual([x['slot'] for x in b['slots']], list(range(10, 20)))
        self.assertTrue({x['slot'] for x in a['slots']}.isdisjoint(x['slot'] for x in b['slots']))
        self.assertTrue({v for x in a['slots'] for v in x['attempt_seeds']}.isdisjoint(
            v for x in b['slots'] for v in x['attempt_seeds']))
        self.assertEqual(b['rules']['aggregate_budget']['calls'], 96)
        for path in ('src/arc/e7_config.py', 'src/arc/e7_protocol.py', 'src/arc/e5_plugins.py'):
            self.assertIn(path, b['dependencies'])

    def test_full_fake_completes_all_thirty_slots(self):
        settings = load_e8_config('configs/e8.full.fake.yml')
        with tempfile.TemporaryDirectory(dir='artifacts/e8') as tmp:
            root = Path(tmp) / 'full'
            for stage in ORDER:
                run_stage(settings, stage, root)
                result = json.loads((root / stage / 'result.json').read_text())
                self.assertEqual(result['completed_slots'], 10)
                self.assertEqual(result['strict_new_oracles'], 0)
                self.assertEqual(len(list((root / stage / 'slots').iterdir())), 10)
                _verify_g(root / stage)

    def test_five_novel_candidates_do_not_stop_full(self):
        settings = load_e8_config('configs/e8.full.fake.yml')
        def slot(**kw):
            i = kw['index']
            return {'arm': 'G0_baseline', 'slot': i, 'completion': 'completed',
                    'status': 'passed', 'oracle_valid': True, 'verified_novel': True}
        with tempfile.TemporaryDirectory(dir='artifacts/e8') as tmp, \
             patch('src.pipelines.e8._slot', side_effect=slot) as candidate:
            root = Path(tmp) / 'full'
            run_stage(settings, '8b', root)
            result = json.loads((root / '8b/result.json').read_text())
            self.assertEqual(candidate.call_count, 10)
            self.assertEqual(result['fixture_novel_oracles'], 10)
            self.assertEqual(result['strict_new_oracles'], 0)

    def test_fake_e2e_resume_and_seals(self):
        with tempfile.TemporaryDirectory(dir='artifacts/e8') as tmp:
            root = Path(tmp) / 'campaign'
            with patch('src.pipelines.e5.train_adapter', side_effect=AssertionError('training called')):
                for stage in ORDER:
                    run_stage(self.s, stage, root)
            for stage in ORDER:
                _verify_g(root / stage)
                d = json.loads((root / stage / 'result.json').read_text())
                self.assertEqual(d['status'], 'completed')
                self.assertEqual(d['completed_slots'], 2)
                self.assertEqual(d['strict_new_oracles'], 0)
                self.assertEqual(d['model_calls'], 4)  # two canaries plus two fixture generations
                self.assertEqual(d['sandbox_starts'], 40)
                self.assertFalse(d['training_invoked'])
                self.assertEqual(d['revision'], MODELS[stage][1])
            with patch('src.arc.e6g_fake.FakeBackend.generate', side_effect=AssertionError('extra inference')):
                run_stage(self.s, '8b', root, resume=True)
            with self.assertRaisesRegex(ValueError, 'identity changed'):
                run_stage(replace(self.s, seed=43), '8b', root, resume=True)
            (root / '8b' / 'slots.json').write_text('tamper')
            with self.assertRaisesRegex(ValueError, 'seal mismatch'):
                run_stage(self.s, '8b', root, resume=True)

    def test_resume_after_interrupted_slot_never_replays_it(self):
        with tempfile.TemporaryDirectory(dir='artifacts/e8') as tmp:
            root = Path(tmp) / 'campaign'
            with patch('src.pipelines.e8._slot', side_effect=KeyboardInterrupt):
                with self.assertRaises(KeyboardInterrupt):
                    run_stage(self.s, '8b', root)
            run_stage(self.s, '8b', root, resume=True)
            result = json.loads((root / '8b' / 'result.json').read_text())
            slots = json.loads((root / '8b' / 'slots.json').read_text())
            self.assertEqual(slots[0]['status'], 'interrupted_uncertain')
            self.assertEqual(slots[1]['completion'], 'completed')
            self.assertEqual(result['status'], 'incomplete_due_to_resource_cap')
            self.assertEqual(result['model_calls'], 3)
            _verify_g(root / '8b')

    def test_cross_model_pool_isolation(self):
        from src.arc.e6f_fake import PROGRAMS
        def unique_fixture(_self, prompt, *, kind, candidate):
            return PROGRAMS[2]
        with tempfile.TemporaryDirectory(dir='artifacts/e8') as tmp:
            root = Path(tmp) / 'campaign'
            with patch('src.arc.e6g_fake.FakeBackend.generate', unique_fixture):
                for stage in ORDER:
                    run_stage(self.s, stage, root)
            results = [json.loads((root / stage / 'result.json').read_text()) for stage in ORDER]
            self.assertEqual([x['fixture_novel_oracles'] for x in results], [1, 1, 1])
            for stage in ORDER:
                slots = json.loads((root / stage / 'slots.json').read_text())
                self.assertTrue(all(not p['feedback_applied'] for slot in slots for p in slot['prompts']))
                self.assertEqual([slot['prompts'][0]['seed_requested'] for slot in slots], [42, 45])

    def test_order_and_permissions(self):
        live = load_e8_config('configs/e8.smoke.yml')
        with tempfile.TemporaryDirectory(dir='artifacts/e8') as tmp:
            root = Path(tmp) / 'campaign'
            with self.assertRaisesRegex(ValueError, 'start with 8B'):
                run_stage(self.s, '14b', root)
            with self.assertRaisesRegex(ValueError, 'independent permission'):
                run_stage(live, '32b', root, live=True)
            with self.assertRaisesRegex(ValueError, 'requires --live'):
                run_stage(live, '8b', root)
            full = load_e8_config('configs/e8.full.yml')
            with self.assertRaisesRegex(ValueError, 'full requires explicit'):
                run_stage(full, '8b', root, live=True)

    def test_resource_limits(self):
        before = {'disk_free_bytes': 100, 'swap_used_mib': 0, 'memory_free_percent': 60}
        self.assertEqual(check_resources(before, dict(before, memory_free_percent=5),
            min_free_percent=10, max_swap_increase_mib=100, min_disk_bytes=20), 'memory_pressure_limit')
        self.assertEqual(check_resources(before, dict(before, swap_used_mib=101),
            min_free_percent=10, max_swap_increase_mib=100, min_disk_bytes=20), 'swap_increase_limit')
        with tempfile.TemporaryDirectory(dir='artifacts/e8') as tmp:
            root = Path(tmp) / 'campaign'
            with patch('src.pipelines.e8.snapshot', return_value={**before, 'memory_free_percent': 5}):
                run_stage(self.s, '8b', root)
            d = json.loads((root / '8b' / 'result.json').read_text())
            self.assertEqual(d['status'], 'resource_infeasible')
            self.assertEqual(d['model_calls'], 0)
            _verify_g(root / '8b')

    def test_monitor_kills_only_own_server_at_swap_limit(self):
        root = Path('/tmp/e8-monitor-test').resolve()
        before = {'disk_free_bytes': 1000, 'swap_used_mib': 0, 'memory_free_percent': 60}
        current = dict(before, swap_used_mib=101)
        monitor = ResourceMonitor(root, before, min_free_percent=10,
                                  max_swap_increase_mib=100, min_disk_bytes=20)
        listing = (f'123 100 python -m src.cli.guarded_mlx_server {root}/ledger\n'
                   '456 100 python -m src.cli.guarded_mlx_server /tmp/other/ledger\n')
        with patch('src.arc.e8_resource.snapshot', return_value=current), \
             patch('src.arc.e8_resource.subprocess.check_output', return_value=listing), \
             patch('src.arc.e8_resource.os.kill') as kill:
            monitor._monitor()
        self.assertEqual(monitor.stop_reason, 'swap_increase_limit')
        self.assertEqual(kill.call_count, 1)
        self.assertEqual(kill.call_args.args[0], 123)

    def test_download_failure_is_sealed_and_not_retried(self):
        live = load_e8_config('configs/e8.smoke.yml')
        with tempfile.TemporaryDirectory(dir='artifacts/e8') as tmp:
            root = Path(tmp) / 'campaign'
            with patch('src.pipelines.e8._check_no_other_model_server'), \
                 patch('src.pipelines.e8._model_dir', return_value=Path(tmp) / 'absent'), \
                 patch('src.pipelines.e8._download', side_effect=RuntimeError('download fixture failure')) as download:
                run_stage(live, '8b', root, live=True)
                result = json.loads((root / '8b' / 'result.json').read_text())
                self.assertEqual(result['completed_slots'], 0)
                self.assertIn('download fixture failure', result['stop_reason'])
                self.assertEqual(result['model_calls'], 0)
                _verify_g(root / '8b')
                run_stage(live, '8b', root, live=True, resume=True)
                self.assertEqual(download.call_count, 1)

    def test_model_revision_and_quantization_are_pinned(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / MODELS['14b'][1]
            root.mkdir()
            (root / 'config.json').write_text(json.dumps({'model_type': 'qwen3', 'quantization': {'bits': 4, 'group_size': 64}}))
            (root / 'tokenizer_config.json').write_text(json.dumps({'chat_template': 'fixture'}))
            (root / 'weights.safetensors').write_bytes(b'fixture')
            manifest = _model_manifest(root, MODELS['14b'][1])
            self.assertEqual(manifest['quantization']['bits'], 4)
            (root / 'config.json').write_text(json.dumps({'model_type': 'qwen3', 'quantization': {'bits': 8, 'group_size': 64}}))
            with self.assertRaisesRegex(ValueError, '4-bit'):
                _model_manifest(root, MODELS['14b'][1])
            with self.assertRaisesRegex(ValueError, 'revision snapshot'):
                _model_manifest(root, MODELS['8b'][1])

    def test_per_model_five_oracle_gate(self):
        self.assertEqual(training_gate(4, fake=False), 'below_five')
        self.assertEqual(training_gate(5, fake=False), 'eligible_for_future_planning')
        self.assertEqual(training_gate(5, fake=True), 'not_applicable_fake_fixture')
        with self.assertRaises(ValueError): training_gate(-1, fake=False)

    def test_operator_search_not_imported(self):
        import subprocess, sys
        subprocess.run([sys.executable, '-c', 'import sys; import src.pipelines.e8; '
                        'assert "src.arc.operator_search" not in sys.modules'], check=True)


if __name__ == '__main__':
    unittest.main()
