"""Bind exploratory partitions to a draft, without approving execution."""
import hashlib
import json
from pathlib import Path

from src.arc.data import Task
from src.arc.rsi_exploratory_split import propose_exploratory_split
from src.arc.rsi_final_proposal import _task_content
from src.arc.rsi_registration import _read_inputs, verify_draft_registration
from src.arc.rsi_slot_audit import manifest_sha256
from src.arc.rsi_task_generation import generation_slots
from src.arc.rsi_transport_audit import _json_object


def prepare_exploratory_registration(root, *, registration, bounded_bytes, bounded_sha256, exploratory):
    root = Path(root).resolve()
    verify_draft_registration(root, registration)
    try:
        inputs, hashes = _read_inputs(root, registration['input_paths'])
        bounded = _json_object(bounded_bytes)
        if (registration.get('status') != 'draft_not_execution_ready' or registration.get('execution_ready') is not False or
                hashes != registration['input_sha256'] or bounded['task_source_sha256'] != hashes['challenges'] or
                inputs['design_registration'].get('supersedes') != 'research/rsi/design-registration-v2.json' or
                inputs['design_registration'].get('exposure_policy') != 'bounded_review_unknown_project_exposure' or
                bounded['partitions'] != inputs['split']['partitions']):
            raise ValueError('draft registration or bounded inventory differs')
        selection_file = root / 'src/arc/rsi_exploratory_split.py'
        code_hash = hashlib.sha256(selection_file.read_bytes()).hexdigest()
        selection = propose_exploratory_split(bounded_bytes, expected_sha256=bounded_sha256)
        expected = {**selection, 'selection_code_sha256':code_hash}
        if exploratory != expected:
            raise ValueError('exploratory selection differs')
        old = bounded['partitions']['generation']
        generation = selection['generation_task_ids']
        for index in (0, 1):
            if generation_slots(old,index) != generation_slots(generation,index):
                raise ValueError('scheduled generation tasks differ')
        if registration['slots'] != generation_slots(generation,registration['generation']):
            raise ValueError('registered slots differ')
        challenges = inputs['challenges']
        groups = {'generation':generation, 'exploratory':selection['exploratory_task_ids'],
                  'development':selection['confirmatory_task_ids']}
        if not isinstance(challenges,dict) or set(challenges)!=set(old+bounded['partitions']['development']):
            raise ValueError('task content source differs')
        contents = {}
        for name, ids in groups.items():
            rows = {}
            for key in ids:
                item = challenges[key]
                if any('output' in pair for pair in item['test']):
                    raise ValueError('task content exposes hidden labels')
                task = Task(key,tuple((pair['input'],pair['output']) for pair in item['train']),
                            tuple(pair['input'] for pair in item['test']))
                rows[key] = _task_content(task)
            if len(set(rows.values())) != len(rows):
                raise ValueError('duplicate task content within partition')
            contents[name] = rows
        sets = [set(rows.values()) for rows in contents.values()]
        if any(sets[i] & sets[j] for i in range(len(sets)) for j in range(i+1,len(sets))):
            raise ValueError('task content overlaps partitions')
        content_hash = {name:hashlib.sha256(json.dumps(rows,sort_keys=True,separators=(',',':'),
                                                     ensure_ascii=False).encode()).hexdigest()
                        for name,rows in contents.items()}
        raw = json.dumps(registration,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()
        result = {'status':'exploratory_registration_draft_unverified', 'execution_ready':False,
                  'independent_review_completed':False, 'training_allowed':False,
                  'bounded_proposal_sha256':bounded_sha256, 'base_registration_sha256':hashlib.sha256(raw).hexdigest(),
                  'selection_code_sha256':code_hash,
                  'generation_manifest_sha256':manifest_sha256(generation),
                  'exploratory_manifest_sha256':manifest_sha256(selection['exploratory_task_ids']),
                  'development_manifest_sha256':manifest_sha256(selection['confirmatory_task_ids']),
                  'generation_content_sha256':content_hash['generation'],
                  'exploratory_content_sha256':content_hash['exploratory'],
                  'development_content_sha256':content_hash['development'],
                  'counts':{'generation':len(generation),'exploratory':len(selection['exploratory_task_ids']),
                            'development':len(selection['confirmatory_task_ids'])},
                  'blockers':['independent_history_review','content_and_ancestry_audit',
                              'exploratory_execution_registration','model_and_runtime_attestation',
                              'separate_execution_authorization']}
        if hashlib.sha256(selection_file.read_bytes()).hexdigest() != code_hash:
            raise ValueError('selection code changed')
        verify_draft_registration(root, registration)
        return result
    except (OSError,KeyError,TypeError,AttributeError) as exc:
        raise ValueError('invalid exploratory registration inputs') from exc
