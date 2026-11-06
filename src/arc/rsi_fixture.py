"""Fixture-only slot integration; no model, external sandbox, or trainer is called."""
import hashlib

from src.arc.e6_fake import FAKE_PROGRAMS, FakeE6Sandbox
from src.arc.rsi_candidate import inspect_candidate
from src.arc.rsi_novelty import screen_novelty
from src.arc.rsi_sft_draft import make_sft_draft
from src.arc.rsi_task_generation import task_prompt, task_repair_prompt


def run_fixture_slot(task, replies: list[str], *, public_probes: list, private_probes: list,
                     before_attempt=None) -> dict:
    if (not isinstance(replies, list) or not 1 <= len(replies) <= 3 or
            any(not isinstance(code, str) or code not in FAKE_PROGRAMS for code in replies)):
        raise ValueError('fixtures require one to three known fake programs')
    sandbox = FakeE6Sandbox()
    prompts, seen = [], set()
    prompt = task_prompt(task)
    for index, code in enumerate(replies):
        if before_attempt is not None:
            before_attempt(index)
        prompts.append(prompt)
        code_hash = hashlib.sha256(code.strip().encode()).hexdigest()
        if code_hash in seen:
            return {'status': 'repeated_code', 'prompts': prompts}
        seen.add(code_hash)
        candidate = inspect_candidate(task, code, public_probes=public_probes,
                                      private_probes=private_probes, sandbox=sandbox)
        status = candidate['status']
        if status == 'verified_candidate':
            novelty = screen_novelty(code, candidate['private_behavior_sha256'],
                                     candidate['private_probe_set_sha256'], [])
            provenance = {'source_type': 'generator_distillation',
                          'model_id': 'mlx-community/Qwen3-14B-4bit',
                          'model_revision': 'a4d9b2df59d2c150bef02fcbe0d91046b7ca33a4',
                          'request_id': f'fixture-{index}', 'raw_response_sha256': code_hash,
                          'execution_mode': 'fake_fixture'}
            draft = make_sft_draft(task, code, candidate, novelty, provenance,
                                   generation=0, parent_adapter='none')
            return {'status': 'fixture_draft', 'prompts': prompts, 'draft': draft}
        if status != 'visible_demonstration_mismatch' or index == len(replies) - 1:
            return {'status': status, 'prompts': prompts}
        prompt = task_repair_prompt(task, code, status)
    raise AssertionError('fixture response loop unexpectedly exhausted')
