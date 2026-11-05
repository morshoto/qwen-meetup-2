import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from src.arc.e7_config import load_e7_config
from src.pipelines.e7 import run_e7
from src.pipelines.e6g import _verify_g
from src.cli.e7_audit import audit


class E7PipelineTests(unittest.TestCase):
    def test_fake_e2e_and_completed_resume_do_not_repeat_inference(self):
        settings, base = load_e7_config('configs/e7.cpu.fake.yml')
        with tempfile.TemporaryDirectory(dir='artifacts/e7') as tmp:
            root = Path(tmp) / 'run'
            run_e7(settings, base, run_dir=root)
            comparison = json.loads((root / 'comparison.json').read_text())
            self.assertEqual(comparison['completed_slots'], {'G0_baseline': 3, 'G1_novelty_feedback': 3})
            self.assertEqual(comparison['novel_oracles'], {'G0_baseline': 0, 'G1_novelty_feedback': 2})
            self.assertEqual(comparison['feasibility_gate'], 'not_applicable_fake_fixture')
            self.assertFalse(comparison['training_invoked'])
            _verify_g(root)
            self.assertTrue(audit(root)['seal_valid'])
            with patch('src.arc.e6g_fake.FakeBackend.generate', side_effect=AssertionError('extra inference')):
                run_e7(settings, base, run_dir=root, resume=True)
            with self.assertRaisesRegex(ValueError, 'protocol'):
                run_e7(replace(settings, seed=43), base, run_dir=root, resume=True)
            (root / 'slots.csv').write_text('tampered')
            with self.assertRaisesRegex(ValueError, 'seal mismatch'):
                run_e7(settings, base, run_dir=root, resume=True)


    def test_uncertain_interruption_is_not_regenerated(self):
        settings, base = load_e7_config('configs/e7.cpu.fake.yml')
        with tempfile.TemporaryDirectory(dir='artifacts/e7') as tmp:
            root = Path(tmp) / 'run'
            with patch('src.pipelines.e7._slot', side_effect=RuntimeError('simulated crash')):
                with self.assertRaisesRegex(RuntimeError, 'simulated crash'):
                    run_e7(settings, base, run_dir=root)
            run_e7(settings, base, run_dir=root, resume=True)
            rows = json.loads((root / 'slots.json').read_text())
            first = next(r for r in rows if r['arm'] == 'G0_baseline' and r['slot'] == 0)
            self.assertEqual(first['status'], 'interrupted_uncertain')
            self.assertEqual(first['completion'], 'incomplete')
            self.assertEqual(json.loads((root / 'comparison.json').read_text())['decision'],
                             'inconclusive_budget_or_missing_data')

    def test_live_requires_explicit_flags(self):
        settings, base = load_e7_config('configs/e7.live.smoke.yml')
        with self.assertRaisesRegex(ValueError, 'explicit live and smoke'):
            run_e7(settings, base)


    def test_full_needs_explicit_matching_budget(self):
        settings, base = load_e7_config('configs/e7.live.full.yml')
        with self.assertRaisesRegex(ValueError, 'explicit matching total budget'):
            run_e7(settings, base, live=True, allow_full=True)

    def test_fake_does_not_invoke_training(self):
        settings, base = load_e7_config('configs/e7.cpu.fake.yml')
        with tempfile.TemporaryDirectory(dir='artifacts/e7') as tmp:
            with patch('src.pipelines.e5.train_adapter', side_effect=AssertionError('training invoked')):
                run_e7(settings, base, run_dir=Path(tmp) / 'run')

    def test_execution_import_graph_excludes_operator_search(self):
        import subprocess
        import sys
        command = 'import sys; import src.pipelines.e7; assert "src.arc.operator_search" not in sys.modules'
        subprocess.run([sys.executable, '-c', command], check=True)


    def test_candidate_budget_stops_without_counting_as_failure(self):
        from src.arc.e6f_fake import BAD, PROGRAMS
        settings, base = load_e7_config('configs/e7.cpu.fake.yml')
        settings = replace(settings, slots=1, candidate_sandbox=20, arm_sandbox=20, total_sandbox=40)
        def generated(_self, prompt, *, kind, candidate):
            return BAD if candidate == 0 else PROGRAMS[0]
        with tempfile.TemporaryDirectory(dir='artifacts/e7') as tmp:
            root = Path(tmp) / 'run'
            with patch('src.arc.e6g_fake.FakeBackend.generate', generated):
                run_e7(settings, base, run_dir=root)
            rows = json.loads((root / 'slots.json').read_text())
            self.assertTrue(all(r['status'] == 'candidate_budget_exhausted' for r in rows))
            self.assertTrue(all(r['completion'] == 'incomplete' for r in rows))
            self.assertTrue(all(r['budget']['sandbox_runs'] == 20 for r in rows))
            self.assertEqual(audit(root)['decision'], 'inconclusive_budget_or_missing_data')


if __name__ == '__main__':
    unittest.main()
