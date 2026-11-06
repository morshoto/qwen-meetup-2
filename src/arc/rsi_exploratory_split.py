"""Propose a separate 50-task exploratory split without authorizing execution."""
import hashlib
import re

from src.arc.rsi_transport_audit import _json_object


def propose_exploratory_split(proposal_bytes, *, expected_sha256):
    if (not isinstance(proposal_bytes,bytes) or not isinstance(expected_sha256,str) or
            not re.fullmatch(r'[0-9a-f]{64}',expected_sha256) or
            hashlib.sha256(proposal_bytes).hexdigest()!=expected_sha256):
        raise ValueError('invalid proposal digest')
    value=_json_object(proposal_bytes)
    try:
        generation=value['partitions']['generation']; development=value['partitions']['development']
        known=value['known_exposed_task_ids']
        if (value.get('status')!='bounded_exposure_proposed_unverified' or value.get('execution_ready') is not False or
                not isinstance(generation,list) or len(generation)!=800 or
                not isinstance(development,list) or len(development)!=200 or not isinstance(known,list) or
                any(not isinstance(x,str) or not x for x in generation+development+known) or
                len(set(generation+development))!=1000 or len(set(known))!=len(known) or
                not set(known)<=set(generation+development) or set(known)&set(development)):
            raise ValueError('invalid bounded proposal')
        scheduled=set(sorted(generation)[:20])
        eligible=sorted(set(generation)-set(known)-scheduled)
        if len(eligible)<50: raise ValueError('insufficient eligible exploratory tasks')
        selected=eligible[-50:]
        remaining=sorted(set(generation)-set(selected))
    except (KeyError,TypeError,AttributeError) as exc:
        raise ValueError('invalid bounded proposal') from exc
    return {'status':'exploratory_split_proposed_unverified','execution_ready':False,
            'independent_review_completed':False,'proposal_sha256':expected_sha256,
            'selection_rule':'last_50_sorted_unexposed_outside_first_20_generation',
            'exploratory_task_ids':selected,'generation_task_ids':remaining,'confirmatory_task_ids':development,
            'prior_project_exposure':'unknown','pretraining_exposure':'unknown'}
