"""Build a non-authorizing draft binding local RSI experiment inputs.

This detects inconsistent local inputs, but does not authenticate external
model/runtime manifests, prove split independence, or permit execution.
"""
import hashlib
import json
import re
from pathlib import Path

from src.arc.data import validate_grid
from src.arc.e6_curriculum import canonical
from src.arc.rsi_lineage_audit import probes_sha256
from src.arc.rsi_novelty import screen_novelty
from src.arc.rsi_source_inventory import build_source_inventory
from src.arc.rsi_task_generation import generation_slots

_KEYS = {'challenges', 'solutions', 'split', 'public_probes', 'private_probes',
         'reference_pool', 'model_manifest', 'runtime_manifest', 'sandbox_identity', 'design_registration'}
_HASH = re.compile(r'^[0-9a-f]{64}$')
_MODEL = 'mlx-community/Qwen3-14B-4bit'
_REVISION = 'a4d9b2df59d2c150bef02fcbe0d91046b7ca33a4'
_V1 = 'research/rsi/design-registration-v1.json'
_V2_DOCUMENT = 'research/rsi/design-v2.md'
_V2 = 'research/rsi/design-registration-v2.json'
_V3_DOCUMENT = 'research/rsi/design-v3.md'
_V3 = 'research/rsi/design-registration-v3.json'
_V4_DOCUMENT = 'research/rsi/design-v4.md'
_V1_FILES = {'configs/rsi-research-protocol.json', 'docs/rsi-research-protocol.md',
             'research/rsi/provisional-split.json'}


def _read_inputs(root: Path, paths: dict) -> tuple[dict, dict]:
    if (not isinstance(paths, dict) or set(paths) != _KEYS or
            any(not isinstance(value, str) or not value for value in paths.values()) or
            len(set(paths.values())) != len(paths)):
        raise ValueError('registration requires distinct paths for all inputs')
    def unique(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError('duplicate JSON key')
            value[key] = item
        return value
    parsed, hashes = {}, {}
    for key, name in sorted(paths.items()):
        if not isinstance(name, str) or not name:
            raise ValueError('invalid input path')
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts or relative.suffix != '.json':
            raise ValueError('input path must be a workspace JSON file')
        cursor = root
        for part in relative.parts:
            cursor = cursor / part
            if cursor.is_symlink():
                raise ValueError('symlinked registration input')
        if not cursor.is_file() or not cursor.resolve().is_relative_to(root):
            raise ValueError('registration input missing or outside workspace')
        try:
            raw = cursor.read_bytes()
            parsed[key] = json.loads(raw, object_pairs_hook=unique,
                                     parse_constant=lambda _: (_ for _ in ()).throw(ValueError('non-finite JSON')))
        except (OSError, UnicodeDecodeError, ValueError) as exc:
            raise ValueError('unreadable or invalid registration input: ' + key) from exc
        hashes[key] = hashlib.sha256(raw).hexdigest()
    return parsed, hashes


def _validate_data(inputs: dict) -> None:
    challenges, solutions, split = inputs['challenges'], inputs['solutions'], inputs['split']
    if not isinstance(challenges, dict) or len(challenges) != 1000 or not isinstance(solutions, dict) or challenges.keys() != solutions.keys():
        raise ValueError('challenge and solution manifests must contain the same 1000 tasks')
    if (not isinstance(split, dict) or split.get('status') != 'provisional_not_preregistered' or
            split.get('independence_established') is not False or not isinstance(split.get('partitions'), dict)):
        raise ValueError('only the explicitly provisional split is supported')
    generation = split['partitions'].get('generation')
    development = split['partitions'].get('development')
    if (not isinstance(generation, list) or not isinstance(development, list) or
            len(generation) != 800 or len(development) != 200 or
            any(not isinstance(item, str) or not item for item in generation + development) or
            len(set(generation + development)) != 1000 or set(generation + development) != set(challenges)):
        raise ValueError('split must partition all 1000 tasks into disjoint 800/200 sets')
    for task, item in challenges.items():
        if (not isinstance(task, str) or not task or not isinstance(item, dict) or
                not isinstance(item.get('train'), list) or not item['train'] or
                not isinstance(item.get('test'), list) or not item['test']):
            raise ValueError('invalid challenge task')
        try:
            for example in item['train']:
                validate_grid(example['input'])
                validate_grid(example['output'])
            for example in item['test']:
                if 'output' in example:
                    raise ValueError('challenge contains hidden test output')
                validate_grid(example['input'])
            outputs = solutions[task]
            if not isinstance(outputs, list) or len(outputs) != len(item['test']):
                raise ValueError('solution count differs from test inputs')
            for output in outputs:
                validate_grid(output)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError('invalid challenge or solution data') from exc


def _verify_design_chain(root: Path, design: dict, split_hash: str) -> dict[str, str]:
    """Verify the recorded local version chain, not an external timestamp."""
    def file_bytes(relative: str) -> bytes:
        path = root
        for part in Path(relative).parts:
            path = path / part
            if path.is_symlink():
                raise ValueError('symlinked design file')
        try:
            return path.read_bytes()
        except OSError as exc:
            raise ValueError('design file is unavailable') from exc
    def check_files(mapping, expected):
        if not isinstance(mapping, dict) or set(mapping) != expected:
            raise ValueError('design file inventory differs')
        for name in sorted(expected):
            if not isinstance(mapping[name], str) or not _HASH.fullmatch(mapping[name]):
                raise ValueError('invalid design fingerprint')
            if hashlib.sha256(file_bytes(name)).hexdigest() != mapping[name]:
                raise ValueError('design file hash differs: ' + name)
    amended_files = {}
    if design.get('supersedes') == _V3:
        if (design.get('kind') != 'design_only' or design.get('execution_ready') is not False or
                type(design.get('competitive_target_percent')) not in (int,float) or
                design.get('competitive_target_percent') != 42.4 or
                design.get('canary_policy') != 'independent_synthetic_only_no_promotion' or
                design.get('evaluation_policy') != 'separate_exploratory_50_then_development_200' or
                design.get('oracle_policy') != 'new_only_pending_historical_decision'):
            raise ValueError('invalid separate canary design')
        check_files(design.get('files'), {_V3, _V4_DOCUMENT})
        amended_files.update(design['files'])
        try:
            design = json.loads(file_bytes(_V3))
        except (ValueError,UnicodeDecodeError) as exc:
            raise ValueError('invalid prior design registration') from exc
        if not isinstance(design,dict) or design.get('supersedes') != _V2:
            raise ValueError('invalid prior design registration')
    if design.get('supersedes') == _V2:
        if (design.get('exposure_policy') != 'bounded_review_unknown_project_exposure' or
                design.get('kind') != 'design_only' or design.get('execution_ready') is not False or
                type(design.get('competitive_target_percent')) not in (int, float) or
                design.get('competitive_target_percent') != 42.4):
            raise ValueError('invalid exposure amendment')
        check_files(design.get('files'), {_V2, _V3_DOCUMENT})
        amended_files.update(design['files'])
        try:
            design = json.loads(file_bytes(_V2))
        except (ValueError, UnicodeDecodeError) as exc:
            raise ValueError('invalid prior design registration') from exc
        if (not isinstance(design, dict) or design.get('kind') != 'design_only' or
                design.get('execution_ready') is not False or design.get('competitive_target_percent') != 42.4):
            raise ValueError('invalid prior design registration')
    reference = design.get('reference')
    if (design.get('supersedes') != _V1 or not isinstance(reference, dict) or
            reference.get('model') != 'Qwen3.8-27B' or
            reference.get('split') != 'ARC-AGI-2 semi-private' or
            reference.get('reasoning_effort') != 'xhigh' or reference.get('matched_compute') is not False):
        raise ValueError('design reference differs')
    check_files(design.get('files'), {_V1, _V2_DOCUMENT})
    try:
        def unique(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError('duplicate design key')
                result[key] = value
            return result
        v1 = json.loads(file_bytes(_V1), object_pairs_hook=unique)
    except (ValueError, UnicodeDecodeError) as exc:
        raise ValueError('invalid design registration') from exc
    if not isinstance(v1, dict) or v1.get('kind') != 'design_only' or v1.get('execution_ready') is not False:
        raise ValueError('invalid prior design registration')
    check_files(v1.get('files'), _V1_FILES)
    if v1['files']['research/rsi/provisional-split.json'] != split_hash:
        raise ValueError('input split differs from design registration')
    return {**amended_files, **design['files'], **v1['files']}


def prepare_draft_registration(root: str | Path, *, input_paths: dict, generation: int) -> dict:
    root = Path(root).resolve()
    inputs, hashes = _read_inputs(root, input_paths)
    _validate_data(inputs)
    model, runtime, sandbox, design = (inputs[k] for k in
                                       ('model_manifest', 'runtime_manifest', 'sandbox_identity', 'design_registration'))
    if (not isinstance(model, dict) or model.get('model_id') != _MODEL or model.get('revision') != _REVISION or
            model.get('quantization') != {'bits': 4, 'group_size': 64} or
            any(type(model['quantization'][key]) is not int for key in ('bits', 'group_size')) or
            not isinstance(model.get('chat_template_sha256'), str) or not _HASH.fullmatch(model['chat_template_sha256']) or
            not isinstance(model.get('files_sha256'), dict) or not model['files_sha256'] or
            any(not isinstance(k, str) or not k or not isinstance(v, str) or not _HASH.fullmatch(v)
                for k, v in model['files_sha256'].items())):
        raise ValueError('model manifest identity or hashes are invalid')
    if not isinstance(runtime, dict) or not runtime:
        raise ValueError('runtime manifest is missing')
    if (not isinstance(sandbox, dict) or not isinstance(sandbox.get('digest'), str) or
            not re.fullmatch(r'sha256:[0-9a-f]{64}', sandbox['digest'])):
        raise ValueError('sandbox image digest is invalid')
    if (not isinstance(design, dict) or design.get('kind') != 'design_only' or
            design.get('execution_ready') is not False or design.get('competitive_target_percent') != 42.4):
        raise ValueError('design registration is invalid')
    design_hashes = _verify_design_chain(root, design, hashes['split'])
    public, private = inputs['public_probes'], inputs['private_probes']
    public_hash, private_hash = probes_sha256(public), probes_sha256(private)
    if (len({canonical(grid) for grid in public}) != len(public) or
            len({canonical(grid) for grid in private}) != len(private) or
            {canonical(grid) for grid in public} & {canonical(grid) for grid in private}):
        raise ValueError('probe sets must be distinct and disjoint')
    slots = generation_slots(inputs['split']['partitions']['generation'], generation)
    source_inputs = {canonical(example['input'])
                     for slot in slots
                     for section in ('train', 'test')
                     for example in inputs['challenges'][slot['task_id']][section]}
    if source_inputs & {canonical(grid) for grid in public + private}:
        raise ValueError('scheduled source inputs overlap verifier probes')
    screen_novelty('def transform(grid): return grid', '0' * 64, private_hash, inputs['reference_pool'])
    inventory = build_source_inventory(root)
    _, after_hashes = _read_inputs(root, input_paths)
    if hashes != after_hashes or design_hashes != _verify_design_chain(root, design, hashes['split']):
        raise ValueError('registration inputs changed during capture')
    exposure_blocker = ('bounded exposure review not independently completed'
                        if design.get('supersedes') in (_V2, _V3) else 'split independence not established')
    return {'status': 'draft_not_execution_ready', 'execution_ready': False,
            'generation': generation, 'input_paths': dict(input_paths), 'input_sha256': hashes,
            'design_file_sha256': design_hashes,
            'local_sources': inventory, 'public_probe_sha256': public_hash,
            'private_probe_sha256': private_hash, 'slots': slots,
            'blockers': [exposure_blocker, 'model runtime and sandbox not independently authenticated',
                         'external dependency closure not established', 'live execution not authorized']}


def verify_draft_registration(root: str | Path, registration: dict) -> None:
    if not isinstance(registration, dict):
        raise ValueError('invalid draft registration')
    current = prepare_draft_registration(root, input_paths=registration.get('input_paths'),
                                         generation=registration.get('generation'))
    if current != registration:
        raise ValueError('draft registration differs from current inputs')
