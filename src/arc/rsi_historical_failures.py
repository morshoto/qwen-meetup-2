"""Summarize supplied historical slot metadata without exposing probe output."""
from collections import Counter


def summarize_historical_slots(rows):
    if not isinstance(rows,list) or not rows: raise ValueError('slot records required')
    seen=set(); statuses=Counter(); attempts=Counter(); valid=novel=raw=ast=0
    for row in rows:
        if not isinstance(row,dict): raise ValueError('invalid slot')
        try:
            slot=row['slot']; status=row['status']; records=row['attempts']
            flags=[row[k] for k in ('oracle_valid','verified_novel','raw_duplicate','ast_duplicate')]
            if (type(slot) is not int or slot<0 or slot in seen or row['completion']!='completed' or
                    not isinstance(status,str) or not status or not isinstance(records,list) or not records or
                    any(type(v) is not bool for v in flags) or (flags[1] and not flags[0])):
                raise ValueError('invalid or incomplete slot')
            seen.add(slot); statuses[status]+=1
            for attempt in records:
                if not isinstance(attempt,dict) or not isinstance(attempt.get('category'),str) or not attempt['category']:
                    raise ValueError('invalid attempt category')
                attempts[attempt['category']]+=1
            valid+=flags[0]; novel+=flags[1]; raw+=flags[2]; ast+=flags[3]
        except (KeyError,TypeError) as exc:
            raise ValueError('invalid historical slot metadata') from exc
    return {'slots':len(rows),'slot_statuses':dict(sorted(statuses.items())),
            'attempt_categories':dict(sorted(attempts.items())), 'valid_candidates':valid,'strict_new':novel,
            'raw_duplicate_slots':raw,'ast_duplicate_slots':ast,
            'duplicate_flags_may_overlap':True,'training_eligible':False,
            'evidence_type':'supplied_historical_metadata'}
