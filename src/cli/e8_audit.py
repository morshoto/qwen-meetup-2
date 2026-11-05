"""Independently check E8 seals, request provenance and comparison boundaries."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import sqlite3
from pathlib import Path
from src.arc.e5_selfplay import _generator_prompt
from src.arc.prompting import FINAL_ANSWER_SYSTEM_PROMPT
from src.pipelines.e8 import CANARIES, _base, _model_dir, _model_manifest, _source_pool
from src.arc.e8_config import E8Settings
from src.arc.e6f_repair import repair_prompt
from src.arc.e6g_novelty import classify_novelty
from src.arc.http_backend import final_answer
from src.pipelines.e5 import _verify_mutable_file
from src.pipelines.e6g import _live_provenance, _seal_g, _verify_g
from src.pipelines.e6f import _sha


def audit(campaign: Path, output: Path) -> dict:
    lock = campaign / 'protocol.lock.yml'
    _verify_mutable_file(lock); _verify_mutable_file(campaign / 'initial_pool.json')
    if (campaign / 'protocol.sha256').read_text().strip() != _sha(lock):
        raise ValueError('campaign lock hash mismatch')
    protocol = json.loads(lock.read_text())
    initial = json.loads((campaign / 'initial_pool.json').read_text())
    settings = E8Settings(**protocol['settings'])
    pool, _, _, _, source = _source_pool(settings, _base(settings), protocol['sandbox_image_id'])
    if initial != pool or source != protocol['verifier_identity']:
        raise ValueError('seed pool or verifier identity mismatch')
    for file, digest in protocol['dependencies'].items():
        if _sha(Path(file)) != digest:
            raise ValueError('current E8 dependency differs from protocol: ' + file)
    stages = {}
    for stage in protocol['model_order']:
        root = campaign / stage
        if not root.exists() or not (root / 'seal.json').exists():
            continue
        _verify_g(root)
        result = json.loads((root / 'result.json').read_text())
        expected_model = protocol['model_identities'][stage]
        if (result['model_id'], result['revision']) != tuple(expected_model[:2]):
            raise ValueError('model identity differs from protocol')
        if result['campaign_protocol_sha256'] != _sha(lock):
            raise ValueError('stage protocol mismatch')
        manifest = result.get('model_manifest', {})
        if manifest and not manifest.get('fixture'):
            actual = _model_manifest(_model_dir(result['model_id'], result['revision']), result['revision'])
            if actual != {k: v for k, v in manifest.items() if k != 'model_id'}:
                raise ValueError('model files differ from registered hashes')
            registered = protocol.get('model_file_manifests', {}).get(stage)
            if registered is not None and actual != registered:
                raise ValueError('model files differ from preregistered manifest')
        pool = json.loads(json.dumps(initial))
        slot_rows = []
        counts = {'generation_requests_verified': 0, 'sandbox_dispatches': 0,
                  'model_id': result['model_id'], 'revision': result['revision'],
                  'canary_request_count': 0}
        for slot in protocol['slots']:
            i = slot['slot']
            slotroot = root / 'slots' / str(i)
            if not slotroot.exists(): continue
            _verify_g(slotroot)
            row = json.loads((slotroot / 'result.json').read_text())
            if row.get('slot') != i or row.get('arm') != 'G0_baseline':
                raise ValueError('slot identity mismatch')
            slot_rows.append(row)
            budget = row.get('budget', {})
            if (budget.get('model_calls', 0) > 3 or budget.get('reserved_tokens', 0) > 15000 or
                    budget.get('sandbox_runs', 0) > protocol['settings']['candidate_sandbox']):
                raise ValueError('slot budget exceeded')
            events_path = slotroot / 'events.jsonl'
            if not events_path.exists(): continue
            events = [json.loads(x) for x in events_path.read_text().splitlines()]
            responses = [json.loads(x) for x in (slotroot / 'logs/e6/llm_responses.jsonl').read_text().splitlines()]
            requests = [json.loads(x) for x in (slotroot / 'logs/e6/llm_requests.jsonl').read_text().splitlines()]
            codes, failures = {}, {}
            for e in events:
                if e['event'] == 'generation_requested':
                    attempt = e['attempt']
                    prompt = _generator_prompt() if attempt == 0 else repair_prompt(codes[attempt-1], *failures[attempt-1])
                    if (e['feedback_applied'] or hashlib.sha256(prompt.encode()).hexdigest() != e['prompt_sha256'] or
                            e['seed_requested'] != protocol['settings']['seed'] + i * 3 + attempt):
                        raise ValueError('prompt, feedback or seed mismatch')
                    index = i * 3 + attempt
                    matches = [x for x in responses if x['candidate'] == index]
                    if len(matches) != 1: raise ValueError('response missing or duplicated')
                    code = final_answer(matches[0]['generated_text']).strip()
                    evidence = _live_provenance(slotroot, prompt, index, code)
                    request = next(x for x in requests if x['request_id'] == evidence['request_id'])
                    if request['model'] != result['model_id'] or request['revision'] != result['revision'] or request['thinking_mode_effective'] != 'off':
                        raise ValueError('request model or thinking mismatch')
                    codes[attempt] = code
                    counts['generation_requests_verified'] += 1
                elif e['event'] == 'attempt_finished':
                    failures[e['attempt']] = (e['category'], e.get('exception_type'))
                elif e['event'] == 'sandbox_dispatch':
                    if e['image_digest'] != protocol['sandbox_image_id']:
                        raise ValueError('sandbox image mismatch')
                    counts['sandbox_dispatches'] += 1
            if len(codes) != len(requests): raise ValueError('unaccounted model request')
            if row.get('oracle_valid'):
                if (any(row.get('phase_starts', {}).get(p, 0) < 10 for p in ('public_validation', 'private_acceptance')) or
                        row.get('sandbox_successful_executions', 0) < 20):
                    raise ValueError('verified candidate lacks twenty successful executions')
                if row['novelty'] != classify_novelty(codes[max(codes)], row['behavior_sha256'], pool):
                    raise ValueError('novelty classification differs')
                if bool(row.get('verified_novel')) != (row['novelty'] == 'novel'):
                    raise ValueError('strict novelty flag differs')
            elif row.get('verified_novel'):
                raise ValueError('unverified candidate marked novel')
            if row.get('accepted_item'): pool.append(row['accepted_item'])
        expected_summary = {
            'planned_slots': protocol['settings']['slots'],
            'completed_slots': sum(row.get('completion') == 'completed' for row in slot_rows),
            'valid_oracles': sum(bool(row.get('oracle_valid')) for row in slot_rows),
            'strict_new_oracles': sum(bool(row.get('verified_novel')) for row in slot_rows),
            'ast_duplicates': sum(bool(row.get('ast_duplicate')) for row in slot_rows),
            'source_duplicates': sum(bool(row.get('raw_duplicate')) for row in slot_rows),
            'behavior_matches': sum(bool(row.get('behavior_match')) for row in slot_rows),
            'behavior_evaluated': sum('behavior_sha256' in row for row in slot_rows),
        }
        if protocol['settings']['profile'] in ('fake', 'fake_full') or result['status'] == 'protocol_invalid':
            expected_summary['strict_new_oracles'] = 0
        if any(result.get(key) != value for key, value in expected_summary.items()):
            raise ValueError('stage summary does not match slot records')
        counts['summary_recomputed'] = True
        canary_log = root / 'logs/e6/llm_requests.jsonl'
        if canary_log.exists():
            canaries = [json.loads(x) for x in canary_log.read_text().splitlines()]
            counts['canary_request_count'] = len(canaries)
            response_path = root / 'logs/e6/llm_responses.jsonl'
            responses = {r['request_id']: r for r in (json.loads(x) for x in response_path.read_text().splitlines())}
            for index, request in enumerate(canaries):
                if request['thinking_mode_effective'] != 'off' or request['model'] != result['model_id']:
                    raise ValueError('canary model or thinking mismatch')
                prompt = CANARIES[index]
                messages = [{'role': 'system', 'content': FINAL_ANSWER_SYSTEM_PROMPT}, {'role': 'user', 'content': prompt}]
                digest = hashlib.sha256(json.dumps(messages, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
                body = responses[request['request_id']]['raw_http_body_text']
                if request['prompt_sha256'] != digest or request['response_sha256'] != hashlib.sha256(body.encode()).hexdigest():
                    raise ValueError('canary prompt or body hash mismatch')
                if json.loads(body)['choices'][0]['message']['content'] != responses[request['request_id']]['generated_text']:
                    raise ValueError('canary HTTP body content differs')
        for ledger in root.rglob('*ledger.sqlite'):
            with sqlite3.connect(f'file:{ledger.resolve()}?mode=ro&immutable=1', uri=True) as db:
                limit, used = db.execute('select token_limit,reserved from budget').fetchone()
                reserved = db.execute('select coalesce(sum(reserved_tokens),0) from requests').fetchone()[0]
            if used != reserved or used > limit: raise ValueError('guarded ledger mismatch')
        if (counts['sandbox_dispatches'] > result['sandbox_starts'] or
            counts['generation_requests_verified'] + counts['canary_request_count'] > result['model_calls'] or
            result['model_calls'] > protocol['settings']['calls'] or
            result['sandbox_starts'] > protocol['settings']['sandbox'] or
            result['reserved_tokens'] > protocol['settings']['reserved_tokens']):
            raise ValueError('budget or trace mismatch')
        if result['status'] == 'completed' and (counts['sandbox_dispatches'] != result['sandbox_starts'] or
            counts['generation_requests_verified'] + counts['canary_request_count'] != result['model_calls']):
            raise ValueError('completed stage has missing traces')
        if result['training_invoked'] or result['adapter_updated'] or result['promoted']:
            raise ValueError('unexpected training or promotion')
        stages[stage] = counts
    answer = {'protocol_sha256': _sha(lock), 'stages': stages, 'seal_valid': True,
              'note': 'audits recorded request evidence; does not prove semantic equivalence'}
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    (output / 'summary.json').write_text(json.dumps(answer, indent=2) + '\n')
    (output / 'report.md').write_text('# E8 independent audit\n\nSeals, protocol identities, G0 prompts, seeds, Qwen response bodies, Sandbox traces, budgets and guarded ledgers were checked. See summary.json for per-stage counts.\n')
    for path in output.iterdir(): os.chmod(path, 0o600)
    _seal_g(output)
    return answer


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('campaign', type=Path)
    p.add_argument('--output', type=Path, required=True); a = p.parse_args()
    print(json.dumps(audit(a.campaign, a.output), indent=2))


if __name__ == '__main__': main()
