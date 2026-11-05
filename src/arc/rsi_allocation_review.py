"""Check proposed task allocation without authorizing or freezing it."""
import hashlib
import json
from pathlib import Path

from src.arc.data import Task
from src.arc.rsi_exploratory_split import propose_exploratory_split
from src.arc.rsi_final_proposal import _task_content
from src.arc.rsi_slot_audit import manifest_sha256
from src.arc.rsi_transport_audit import _json_object


def review_task_allocation(dataset_bytes, bounded_bytes, selection, *, expected_bounded_sha256):
    if not isinstance(dataset_bytes,bytes) or not isinstance(bounded_bytes,bytes):
        raise ValueError('allocation inputs must be bytes')
    data=_json_object(dataset_bytes); bounded=_json_object(bounded_bytes)
    try:
        if hashlib.sha256(dataset_bytes).hexdigest()!=bounded['task_source_sha256']:
            raise ValueError('dataset digest differs')
        expected=propose_exploratory_split(bounded_bytes,expected_sha256=expected_bounded_sha256)
        code=Path(__file__).with_name('rsi_exploratory_split.py').read_bytes()
        if selection!={**expected,'selection_code_sha256':hashlib.sha256(code).hexdigest()}:
            raise ValueError('selection differs')
        groups={'generation':selection['generation_task_ids'],'exploratory':selection['exploratory_task_ids'],
                'development':selection['confirmatory_task_ids']}
        if set(data)!=set(sum(groups.values(),[])): raise ValueError('dataset task IDs differ')
        known=set(bounded['known_exposed_task_ids']); content={}
        for group,ids in groups.items():
            content[group]={}
            for task_id in ids:
                item=data[task_id]
                if any('output' in pair for pair in item['test']): raise ValueError('hidden test labels present')
                task=Task(task_id,tuple((pair['input'],pair['output']) for pair in item['train']),
                          tuple(pair['input'] for pair in item['test']))
                content[group][task_id]=_task_content(task)
        fingerprints=[h for rows in content.values() for h in rows.values()]
        if len(set(fingerprints))!=len(fingerprints): raise ValueError('task content overlaps')
        known_counts={group:len(set(ids)&known) for group,ids in groups.items()}
        if known_counts['exploratory'] or known_counts['development'] or known_counts['generation']!=len(known):
            raise ValueError('known exposure allocation differs')
        content_hashes={group:hashlib.sha256(json.dumps(rows,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
                        for group,rows in content.items()}
        return {'status':'allocation_consistent_unverified','execution_ready':False,'independent_review_completed':False,
                'dataset_sha256':hashlib.sha256(dataset_bytes).hexdigest(),'bounded_sha256':expected_bounded_sha256,
                'selection_sha256':hashlib.sha256(json.dumps(selection,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest(),
                'counts':{group:len(ids) for group,ids in groups.items()},'known_exposed_counts':known_counts,
                'id_manifest_sha256':{group:manifest_sha256(ids) for group,ids in groups.items()},
                'content_manifest_sha256':content_hashes,'exact_content_unique':True,
                'unknown_prior_project_exposure':True,'unknown_pretraining_exposure':True}
    except (KeyError,TypeError,AttributeError,OSError) as exc:
        raise ValueError('invalid allocation inputs') from exc
