"""Summarize caller-supplied exploratory outcomes without asserting provenance."""
from collections import Counter
import math
import re

_MODEL = 'mlx-community/Qwen3-14B-4bit'
_REVISION = 'a4d9b2df59d2c150bef02fcbe0d91046b7ca33a4'


def summarize_exploratory(value):
    if not isinstance(value, dict): raise ValueError('invalid comparison')
    try:
        ids, final, arms = value['task_ids'], value['confirmatory_task_ids'], value['arms']
        if (value.get('kind') != 'exploratory_50' or value.get('model') != _MODEL or value.get('revision') != _REVISION or
                not isinstance(ids,list) or len(ids)!=50 or not isinstance(final,list) or len(final)!=200 or
                any(not isinstance(x,str) or not x for x in ids+final) or len(set(ids))!=50 or len(set(final))!=200 or
                set(ids)&set(final) or not isinstance(arms,dict) or set(arms)!={'base','trained'} or
                arms['base']['adapter']!='none' or not isinstance(arms['trained']['adapter'],str) or
                not re.fullmatch(r'[0-9a-f]{64}',arms['trained']['adapter']) or
                not isinstance(arms['base']['settings'],dict) or not arms['base']['settings'] or
                arms['base']['settings']!=arms['trained']['settings']):
            raise ValueError('comparison identity, settings or task split differs')
        summaries, indexed = {}, {}
        for name in ('base','trained'):
            rows=arms[name]['results']
            if not isinstance(rows,list) or len(rows)>50: raise ValueError('invalid results')
            seen={}; kinds={}; failures=Counter(); elapsed=[]; tokens=[]
            for row in rows:
                if not isinstance(row,dict): raise ValueError('invalid result')
                tid=row['task_id']; solved=row['solved']; kind=row['task_type']; reason=row['failure_reason']; seeds=row['seeds']
                if (tid not in ids or tid in seen or type(solved) is not bool or
                        (kind is not None and (not isinstance(kind,str) or not kind.strip())) or
                        (reason is not None and (not isinstance(reason,str) or not reason.strip())) or
                        (solved and reason is not None) or not isinstance(seeds,list) or len(seeds)!=2 or
                        any(type(x) is not int or x<0 for x in seeds) or seeds[0]==seeds[1]):
                    raise ValueError('invalid task result')
                seen[tid]=row
                group=kinds.setdefault(kind if kind is not None else 'unclassified',{'evaluated':0,'correct':0})
                group['evaluated']+=1; group['correct']+=int(solved)
                if not solved: failures[reason or 'unknown']+=1
                t=row['elapsed_seconds']; pt=row['prompt_tokens']; ct=row['completion_tokens']
                if t is not None:
                    if type(t) not in (int,float) or not math.isfinite(t) or t<0: raise ValueError('invalid time')
                    elapsed.append(t)
                for n in (pt,ct):
                    if n is not None and (type(n) is not int or n<0): raise ValueError('invalid tokens')
                if pt is not None and ct is not None: tokens.append(pt+ct)
            indexed[name]=seen
            summaries[name]={'evaluated':len(rows),'correct':sum(r['solved'] for r in rows),'types':kinds,
                'failure_reasons':dict(failures),'time_coverage':len(elapsed),
                'elapsed_seconds_total':sum(elapsed) if len(elapsed)==50 else None,
                'token_coverage':len(tokens),'total_tokens':sum(tokens) if len(tokens)==50 else None}
        for tid in set(indexed['base']) & set(indexed['trained']):
            if any(indexed['base'][tid][k]!=indexed['trained'][tid][k] for k in ('seeds','task_type')):
                raise ValueError('paired task metadata differs')
    except (KeyError,TypeError,AttributeError) as exc:
        raise ValueError('invalid comparison data') from exc
    return {'kind':'exploratory_50_summary','complete':all(len(x)==50 for x in indexed.values()),
            'claim_eligible':False,'independently_verified':False,'arms':summaries,
            'evidence_type':'caller_supplied_unverified_results'}
