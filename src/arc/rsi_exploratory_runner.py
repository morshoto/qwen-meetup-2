"""Run a bounded exploratory comparison with unauthenticated supplied components.

This is not an execution registration or authorization to dispatch model calls.
"""
import hashlib
import json
import math
import re
import time
from pathlib import Path

from src.arc.data import Task
from src.arc.prompting import system_prompt_for_kind
from src.arc.rsi_budget_adapters import BudgetedSandbox
from src.arc.rsi_comparison_freeze import _SETTINGS
from src.arc.rsi_evaluation import InvalidModelOutput, evaluate_task, score_task
from src.arc.rsi_exploratory_budget import ExploratoryBudget
from src.arc.rsi_exploratory_report import summarize_exploratory
from src.arc.rsi_slot_audit import manifest_sha256, task_sha256
from src.arc.rsi_slot_lifecycle import _create, _directory


def run_exploratory(*, root, registration_sha256, generation, registration, task_ids, confirmatory_task_ids,
                    tasks, task_hashes, seeds, task_types, candidate_adapter, generate, sandbox, verifier, budget):
    if (type(budget) is not ExploratoryBudget or budget.identity != {'registration_sha256':registration_sha256,'generation':generation} or
            Path(budget.root).resolve() != Path(root).resolve() or not isinstance(registration,dict) or
            registration.get('status') != 'exploratory_registration_draft_unverified' or
            registration.get('execution_ready') is not False or not isinstance(task_ids,list) or len(task_ids)!=50 or
            not isinstance(confirmatory_task_ids,list) or len(confirmatory_task_ids)!=200 or
            any(not isinstance(k,str) or not k for k in task_ids+confirmatory_task_ids) or
            len(set(task_ids+confirmatory_task_ids))!=250 or
            registration.get('exploratory_manifest_sha256')!=manifest_sha256(task_ids) or
            registration.get('development_manifest_sha256')!=manifest_sha256(confirmatory_task_ids) or
            not isinstance(candidate_adapter,str) or not re.fullmatch(r'[0-9a-f]{64}',candidate_adapter) or
            not callable(generate) or not isinstance(tasks,dict) or set(tasks)!=set(task_ids) or
            not isinstance(task_hashes,dict) or set(task_hashes)!=set(task_ids) or
            not isinstance(seeds,dict) or set(seeds)!=set(task_ids) or
            not isinstance(task_types,dict) or set(task_types)!=set(task_ids)):
        raise ValueError('exploratory comparison identity differs')
    all_seeds=[]
    for key in task_ids:
        task=tasks[key]; pair=seeds[key]; kind=task_types[key]
        if (not isinstance(task,Task) or task.id!=key or not task.train or not task.test_inputs or
                task_sha256(task)!=task_hashes[key] or verifier.output_count(key)!=len(task.test_inputs) or
                not isinstance(pair,list) or len(pair)!=2 or any(type(x) is not int or x<0 for x in pair) or
                (kind is not None and (not isinstance(kind,str) or not kind.strip()))):
            raise ValueError('exploratory task, type or seeds differ')
        all_seeds.extend(pair)
    if len(set(all_seeds))!=100: raise ValueError('sample seeds must be distinct')
    before=budget.snapshot()
    if any(before[k]!=0 for k in ('model_calls','reserved_tokens','sandbox_runs')):
        raise ValueError('exploratory budget must be fresh')
    with _directory(root) as directory:
        _create(directory,'exploratory.started.json',{'registration_sha256':registration_sha256,'generation':generation,
                                                     'task_hashes':task_hashes,'budget_before':before})
    settings=dict(_SETTINGS)
    data={'kind':'exploratory_50','model':_SETTINGS['model_id'],'revision':_SETTINGS['model_revision'],
          'task_ids':task_ids,'confirmatory_task_ids':confirmatory_task_ids,
          'arms':{name:{'adapter':adapter,'settings':settings,'results':[]}
                  for name,adapter in (('base','none'),('trained',candidate_adapter))}}
    seen=set(); metered=BudgetedSandbox(sandbox,budget)
    try:
        for index,key in enumerate(task_ids):
            task=tasks[key]
            for arm,adapter in (('base','none'),('trained',candidate_adapter)):
                started=time.monotonic(); usages=[]; request_ids=[]
                def call(prompt, *, sample):
                    messages=[{'role':'system','content':system_prompt_for_kind('e5_self_solve')},
                              {'role':'user','content':prompt}]
                    reservation=512+len(json.dumps(messages,ensure_ascii=False).encode('utf-8'))+1024
                    budget.reserve_model(reservation,kind='evaluation')
                    response=generate(prompt,arm=arm,adapter=adapter,task_id=key,sample=sample,
                                      seed=seeds[key][sample],timeout=budget.remaining())
                    if not isinstance(response,dict) or set(response) not in ({'code','request_id','usage'},
                            {'code','request_id','usage','failure'}): raise ValueError('invalid transport response')
                    rid=response['request_id']; usage=response['usage']
                    if not isinstance(rid,str) or not re.fullmatch(r'[0-9a-f]{32}',rid) or rid in seen:
                        raise ValueError('invalid or duplicate request ID')
                    seen.add(rid); request_ids.append(rid)
                    if usage is not None and (not isinstance(usage,dict) or set(usage)!={'prompt_tokens','completion_tokens'} or
                            any(type(v) is not int or v<0 for v in usage.values())):
                        raise ValueError('invalid reported usage')
                    usages.append(usage)
                    if 'failure' in response:
                        if response['failure']!='invalid_model_output' or response['code'] is not None: raise ValueError('invalid failure')
                        return InvalidModelOutput()
                    return response['code']
                evaluated=evaluate_task(task,generate=call,sandbox=metered)
                score=score_task(task,evaluated,verifier)
                elapsed=time.monotonic()-started
                if not math.isfinite(elapsed) or elapsed<0: raise ValueError('invalid elapsed time')
                complete_usage=len(usages)==2 and all(x is not None for x in usages)
                failed=[x for x in evaluated['sample_statuses'] if x!='accepted']
                row={'task_id':key,'task_type':task_types[key],'solved':score['solved'],
                     'failure_reason':None if score['solved'] else (failed[0] if failed else 'incorrect_output'),
                     'elapsed_seconds':elapsed,'prompt_tokens':sum(x['prompt_tokens'] for x in usages) if complete_usage else None,
                     'completion_tokens':sum(x['completion_tokens'] for x in usages) if complete_usage else None,
                     'seeds':list(seeds[key])}
                with _directory(root) as directory:
                    _create(directory,f'exploratory-{index:02d}-{arm}.json',{'task_id':key,'result':evaluated,
                        'score':score,'request_ids':request_ids,'reported_usage':usages,'row':row})
                data['arms'][arm]['results'].append(row)
        summary=summarize_exploratory(data)
        raw=(json.dumps(data,sort_keys=True,separators=(',',':'))+'\n').encode()
        with _directory(root) as directory: _create(directory,'exploratory-results.json',data)
        return {'status':'exploratory_complete_unverified','claim_eligible':False,
                'independently_verified':False,'sha256':hashlib.sha256(raw).hexdigest(),'summary':summary}
    except Exception as exc:
        report={'status':'exploratory_incomplete','claim_eligible':False,'independently_verified':False,
                'failure_type':type(exc).__name__,'completed':{k:len(v['results']) for k,v in data['arms'].items()}}
        with _directory(root) as directory: _create(directory,'exploratory.incomplete.json',report)
        return report
