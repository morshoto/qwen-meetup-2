import hashlib
import json
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from src.arc.rsi_final_journal import claim_diagnostic, complete_diagnostic, observe_diagnostic


class DiagnosticJournalTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.proposal = {'status': 'final_diagnostic_proposed_unverified', 'execution_ready': False,
            'selected_adapter': 'a' * 64, 'comparison_sha256': ['b' * 64],
            'generation_task_content_manifest_sha256': 'c' * 64,
            'diagnostic_task_hashes': {'eval': 'd' * 64}, 'diagnostic_content_hashes': {'eval': 'e' * 64},
            'harness_sha256': 'f' * 64, 'settings_sha256': 'a' * 64,
            'budget': {'model_calls': 1, 'reserved_tokens': 1000, 'sandbox_runs': 10, 'wall_seconds': 60},
            'competitive_target_percent': 42.4, 'target_split': 'arc_agi_2_semi_private',
            'diagnostic_split': 'arc_agi_2_public_evaluation',
            'evidence_type': 'caller_supplied_screen_and_unwitnessed_proposal'}
        path = self.root / 'final.diagnostic.proposal.json'
        path.write_text(json.dumps(self.proposal))
        path.chmod(0o600)
        self.digest = hashlib.sha256(path.read_bytes()).hexdigest()

    def test_interrupted_claim_is_consumed_and_completion_needs_token(self):
        self.assertEqual(observe_diagnostic(self.root, expected_proposal_sha256=self.digest)['status'], 'unstarted')
        claim = claim_diagnostic(self.root, expected_proposal_sha256=self.digest)
        self.assertFalse(claim['execution_ready'])
        self.assertEqual(observe_diagnostic(self.root, expected_proposal_sha256=self.digest)['status'],
                         'started_outcome_uncertain')
        with self.assertRaises(FileExistsError):
            claim_diagnostic(self.root, expected_proposal_sha256=self.digest)
        with self.assertRaises(ValueError):
            complete_diagnostic(self.root, expected_proposal_sha256=self.digest,
                                claim_token='0' * 64, evidence_sha256='1' * 64)
        complete_diagnostic(self.root, expected_proposal_sha256=self.digest,
                            claim_token=claim['claim_token'], evidence_sha256='1' * 64)
        result = observe_diagnostic(self.root, expected_proposal_sha256=self.digest)
        self.assertEqual(result['status'], 'completion_recorded_unverified')
        self.assertFalse(result['independently_verified'])
        with self.assertRaises(FileExistsError):
            complete_diagnostic(self.root, expected_proposal_sha256=self.digest,
                                claim_token=claim['claim_token'], evidence_sha256='1' * 64)

    def test_invalid_or_changed_proposal_cannot_be_claimed_or_completed(self):
        with self.assertRaises(ValueError):
            claim_diagnostic(self.root, expected_proposal_sha256='0' * 64)
        claim = claim_diagnostic(self.root, expected_proposal_sha256=self.digest)
        (self.root / 'final.diagnostic.proposal.json').write_text('{}')
        with self.assertRaises(ValueError):
            complete_diagnostic(self.root, expected_proposal_sha256=self.digest,
                                claim_token=claim['claim_token'], evidence_sha256='1' * 64)

    def test_concurrent_claims_have_one_winner(self):
        def attempt():
            try:
                claim_diagnostic(self.root, expected_proposal_sha256=self.digest)
                return 'claimed'
            except FileExistsError:
                return 'already_claimed'
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(lambda _: attempt(), range(2)))
        self.assertCountEqual(results, ['claimed', 'already_claimed'])


if __name__ == '__main__':
    unittest.main()
