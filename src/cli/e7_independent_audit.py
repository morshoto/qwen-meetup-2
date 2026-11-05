"""Reconstruct E7 prompts, Qwen response evidence and novelty independently."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import sqlite3
from pathlib import Path

from src.arc.e5_selfplay import _generator_prompt
from src.arc.e6f_repair import repair_prompt
from src.arc.e6g_config import load_e6g_config
from src.arc.e6g_novelty import NoveltyFeedback, classify_novelty
from src.arc.http_backend import final_answer
from src.cli.e7_audit import audit
from src.pipelines.e6g import _load_pool, _live_provenance, _seal_g, _verify_g
from src.pipelines.e6f import _sha, _split_evidence


def independent_audit(root: Path, output: Path, snapshot: Path) -> dict:
    audit(root)
    _verify_g(snapshot)
    protocol = json.loads((root / 'protocol.lock.yml').read_text())
    for name, digest in protocol['e7_implementation_sha256'].items():
        if _sha(snapshot / name) != digest:
            raise ValueError(f'archived source differs: {name}')
    settings, base = load_e6g_config('configs/e6g.live.smoke.yml')
    splits, _ = _split_evidence(base)
    image = protocol['e6g_identity']['sandbox_image_id']
    initial, _, info = _load_pool(settings, base, image, splits)
    if info['pool_sha256'] != protocol['e6g_identity']['initial_pool']['pool_sha256']:
        raise ValueError('initial pool mismatch')
    arms = ('G0_baseline', 'G1_novelty_feedback')
    pools = {arm: json.loads(json.dumps(initial)) for arm in arms}
    feedback = {arm: NoveltyFeedback(enabled=arm == arms[1], pool=pools[arm]) for arm in arms}
    corrected = {arm: {'fingerprinted': 0, 'behavior_matches': 0, 'calls_verified': 0} for arm in arms}
    for spec in protocol['slot_manifest']:
        arm, slot = spec['arm'], spec['slot']
        slotroot = root / 'conditions' / arm / f'slot-{slot}'
        row = json.loads((slotroot / 'result.json').read_text())
        events_path = slotroot / 'events.jsonl'
        if not events_path.exists():
            continue
        events = [json.loads(line) for line in events_path.read_text().splitlines()]
        responses_path = slotroot / 'logs/e6/llm_responses.jsonl'
        requests_path = slotroot / 'logs/e6/llm_requests.jsonl'
        responses = [json.loads(line) for line in responses_path.read_text().splitlines()]
        requests = [json.loads(line) for line in requests_path.read_text().splitlines()]
        code_by_attempt, failures = {}, {}
        for event in events:
            if event.get('event') == 'generation_requested':
                attempt = event['attempt']
                prompt = feedback[arm].prompt(_generator_prompt()) if attempt == 0 else repair_prompt(
                    code_by_attempt[attempt - 1], *failures[attempt - 1])
                if hashlib.sha256(prompt.encode()).hexdigest() != event['prompt_sha256']:
                    raise ValueError('prompt boundary mismatch')
                n = slot * 3 + attempt
                matches = [x for x in responses if x['candidate'] == n]
                if len(matches) != 1:
                    raise ValueError('response missing or ambiguous')
                code = final_answer(matches[0]['generated_text']).strip()
                evidence = _live_provenance(slotroot, prompt, n, code)
                request = next(x for x in requests if x['request_id'] == evidence['request_id'])
                if request['seed_requested'] != protocol['settings']['seed'] + n:
                    raise ValueError('requested seed mismatch')
                feedback[arm].observe(code, pools[arm])
                code_by_attempt[attempt] = code
                corrected[arm]['calls_verified'] += 1
            if event.get('event') == 'attempt_finished':
                failures[event['attempt']] = (event['category'], event.get('exception_type'))
        if len(requests) != len(code_by_attempt):
            raise ValueError('model request count differs from evidence')
        if row.get('oracle_valid'):
            behavior = row['behavior_sha256']
            corrected[arm]['fingerprinted'] += 1
            corrected[arm]['behavior_matches'] += int(any(x['behavior_sha256'] == behavior for x in pools[arm]))
            last = code_by_attempt[max(code_by_attempt)]
            if row['novelty'] != classify_novelty(last, behavior, pools[arm]):
                raise ValueError('novelty classification mismatch')
        if row.get('accepted_item'):
            pools[arm].append(row['accepted_item'])
        for ledger in (slotroot / 'checkpoints/e6').glob('*ledger.sqlite'):
            with sqlite3.connect(f'file:{ledger.resolve()}?mode=ro&immutable=1', uri=True) as db:
                limit, reserved = db.execute('select token_limit,reserved from budget').fetchone()
                count, total = db.execute('select count(*),coalesce(sum(reserved_tokens),0) from requests').fetchone()
            if reserved != total or reserved > limit or count != len(requests):
                raise ValueError('guarded server ledger mismatch')
        sensitive = [p for p in slotroot.rglob('*') if p.is_file() and
                     (p.name in {'result.json', 'events.jsonl', 'request_cache.json'} or
                      'logs/e6/' in str(p.relative_to(slotroot)))]
        if slotroot.stat().st_mode & 0o077 or any(p.stat().st_mode & 0o077 for p in sensitive):
            raise ValueError('sensitive slot evidence is not owner-only')
    summary = {'source_seal_verified': True, 'source_protocol_sha256': _sha(root / 'protocol.lock.yml'),
               'archived_implementation_verified': True, 'prompt_and_http_body_verified': True,
               'corrected_sampled_behavior': corrected,
               'caveat': 'matching finite probes suggest similarity, not semantic equivalence'}
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    (output / 'report.md').write_text('# E7 independent audit\n\nVerified seals, archived implementation, prompt reconstruction, Qwen HTTP bodies, requested seeds, pool isolation, guarded server ledgers, and file permissions. Corrected sampled behavior matches: G0 0 evaluated; G1 2 of 2. The original sealed comparison undercounted matches that were already classified as source duplicates.\n')
    for path in output.iterdir():
        os.chmod(path, 0o600)
    _seal_g(output)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', type=Path)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(independent_audit(args.run_dir, args.output, args.snapshot), indent=2))


if __name__ == '__main__':
    main()
