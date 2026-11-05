"""Replay exploratory captures without claiming independent execution origin."""
import hashlib
import json
from pathlib import Path

from src.arc.data import Task
from src.arc.e5_selfplay import solve_prompt
from src.arc.rsi_collected_slot_audit import _decode, _index
from src.arc.rsi_comparison_audit import _read, _read_jsonl
from src.arc.rsi_comparison_freeze import _hash
from src.arc.rsi_comparison_usage import _summarize_usage
from src.arc.rsi_evaluation import InvalidModelOutput, evaluate_task, score_task
from src.arc.rsi_exploratory_budget import ExploratoryBudget
from src.arc.rsi_exploratory_report import summarize_exploratory
from src.arc.rsi_slot_audit import task_sha256
from src.arc.rsi_slot_lifecycle import _directory, _exists
from src.arc.rsi_transport_audit import _json_object, audit_transport


def audit_exploratory_captures(root, *, expected_sha256, tasks, sandbox, verifier):
    if not _hash(expected_sha256): raise ValueError('invalid exploratory result digest')
    root=Path(root)
    with _directory(root) as directory:
        if _exists(directory,'exploratory.incomplete.json'): raise ValueError('incomplete exploratory comparison')
        _,started=_read(directory,'exploratory.started.json')
        _,completed=_read(directory,'exploratory.completed.json')
        raw,data=_read(directory,'exploratory-results.json')
        if (hashlib.sha256(raw).hexdigest()!=expected_sha256 or completed.get('result_sha256')!=expected_sha256 or
                completed.get('registration_sha256')!=started.get('registration_sha256') or
                completed.get('claim_eligible') is not False):
            raise ValueError('exploratory completion differs')
        summary=summarize_exploratory(data)
        if not summary['complete']: raise ValueError('exploratory results incomplete')
        ids=data['task_ids']
        if (not isinstance(tasks,dict) or set(tasks)!=set(ids) or
                any(not isinstance(tasks[k],Task) or tasks[k].id!=k or task_sha256(tasks[k])!=started['task_hashes'][k] for k in ids)):
            raise ValueError('exploratory task identity differs')
        captures={}
        for arm,mapped in (('base','parent'),('trained','candidate')):
            requests=_read_jsonl(directory,f'{mapped}.capture.jsonl')
            with _directory(root/f'{mapped}.responses') as arm_dir:
                events=_read_jsonl(arm_dir,'requests.jsonl')
                responses=_read_jsonl(arm_dir,'llm_responses.jsonl')
            captures[arm]=tuple(map(_index,(requests,events,responses)))
        seen={'base':set(),'trained':set()}; reserved=0
        for index,key in enumerate(ids):
            for arm in ('base','trained'):
                _,record=_read(directory,f'exploratory-{index:02d}-{arm}.json')
                if (set(record)!={'task_id','result','score','request_ids','reported_usage','row'} or
                        record['task_id']!=key or record['row']!=data['arms'][arm]['results'][index]):
                    raise ValueError('exploratory task record differs')
                result=record['result']; sources=result.get('sample_sources'); statuses=result.get('sample_statuses')
                request_ids=record['request_ids']; usages=record['reported_usage']
                if any(not isinstance(x,list) or len(x)!=2 for x in (sources,statuses,request_ids,usages)):
                    raise ValueError('exploratory sample evidence incomplete')
                def generate(_prompt, *, sample):
                    return InvalidModelOutput() if statuses[sample]=='invalid_model_output' else sources[sample]
                replay=evaluate_task(tasks[key],generate=generate,sandbox=sandbox)
                score=score_task(tasks[key],replay,verifier)
                if replay!=result or score!=record['score'] or score['solved']!=record['row']['solved']:
                    raise ValueError('exploratory task replay differs')
                for sample,rid in enumerate(request_ids):
                    requests,events,responses=captures[arm]
                    if rid in seen[arm] or rid not in requests or rid not in events or rid not in responses:
                        raise ValueError('exploratory request missing or duplicated')
                    seen[arm].add(rid); request,event,response=requests[rid],events[rid],responses[rid]
                    request_raw=_decode(request.get('body_base64')); response_raw=_decode(response.get('raw_http_body_base64'))
                    body=_json_object(request_raw); response_body=_json_object(response_raw)
                    reservation=512+len(json.dumps(body['messages'],ensure_ascii=False).encode())+1024
                    if (request.get('body_sha256')!=hashlib.sha256(request_raw).hexdigest() or
                            response.get('http_status')!=200 or event.get('kind')!='e5_self_solve' or
                            event.get('token_reservation')!=reservation or event.get('usage')!=response_body.get('usage')):
                        raise ValueError('exploratory capture metadata differs')
                    _summarize_usage([event]); reserved+=reservation
                    u=event.get('usage'); normalized=({k:u[k] for k in ('prompt_tokens','completion_tokens')}
                        if isinstance(u,dict) and all(k in u for k in ('prompt_tokens','completion_tokens')) else None)
                    if normalized!=usages[sample]: raise ValueError('exploratory usage differs')
                    audit_transport(request_raw,response_raw,event,expected_prompt=solve_prompt(tasks[key]),
                        expected_code=sources[sample],expected_seed=record['row']['seeds'][sample],
                        expected_invalid=statuses[sample]=='invalid_model_output')
                coverage=all(x is not None for x in usages)
                for keyname in ('prompt_tokens','completion_tokens'):
                    total=sum(x[keyname] for x in usages) if coverage else None
                    if record['row'][keyname]!=total: raise ValueError('exploratory row usage differs')
        for arm in captures:
            if len(seen[arm])!=100 or any(set(group)!=seen[arm] for group in captures[arm]):
                raise ValueError('exploratory capture coverage differs')
        after=completed.get('budget_after')
        if (not isinstance(after,dict) or after.get('model_calls')!=200 or after.get('reserved_tokens')!=reserved or
                type(after.get('sandbox_runs')) is not int or not 0<=after['sandbox_runs']<=2000):
            raise ValueError('exploratory budget evidence differs')
        live=ExploratoryBudget.open(root,registration_sha256=started['registration_sha256'],generation=started['generation']).snapshot()
        if any(live[k]!=after[k] for k in ('model_calls','reserved_tokens','sandbox_runs')):
            raise ValueError('exploratory budget changed after completion')
    return {'status':'exploratory_capture_consistent_unverified','checked_requests':200,
            'independently_verified':False,'claim_eligible':False,'elapsed_time_independently_verified':False,
            'evidence_type':'caller_supplied_captures_sandbox_and_verifier'}
