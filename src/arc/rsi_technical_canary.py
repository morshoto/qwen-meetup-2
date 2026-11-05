"""Propose a separate synthetic canary; never load weights or start training."""
import re
from pathlib import Path

from src.arc.rsi_canary_dataset import audit_canary_loader

_MODELS = {'14b':('mlx-community/Qwen3-14B-4bit','a4d9b2df59d2c150bef02fcbe0d91046b7ca33a4'),
           '32b':('mlx-community/Qwen3-32B-4bit','bcaaf7f538adf166c1080a2befdb4f6019f66639')}
_IDENTITIES = {'model_manifest','runtime_manifest','environment_manifest','source_inventory','tokenizer','tokenizer_config'}


def propose_technical_canary(*, root, model_key, model_dir, adapter_dir, forbidden_tasks, tokenizer, render,
                             load_local_dataset, expected_dataset_sha256, identity_hashes, resource_limits, seed):
    if (not isinstance(model_key,str) or model_key not in _MODELS or not isinstance(identity_hashes,dict) or set(identity_hashes)!=_IDENTITIES or
            any(not isinstance(x,str) or not re.fullmatch(r'[0-9a-f]{64}',x) for x in identity_hashes.values()) or
            type(seed) is not int or not 0<=seed<2**31):
        raise ValueError('invalid canary identity')
    constraints={'timeout_seconds':(1,300),'minimum_disk_gib':(20,1000000),
                 'minimum_free_memory_percent':(20,100),'maximum_swap_growth_mib':(1,512),
                 'maximum_process_rss_gib':(1,32),'sample_interval_seconds':(1,5)}
    if (not isinstance(resource_limits,dict) or set(resource_limits)!=set(constraints) or
            any(type(resource_limits[k]) is not int or not lo<=resource_limits[k]<=hi
                for k,(lo,hi) in constraints.items())):
        raise ValueError('invalid canary resource limits')
    try:
        root=Path(root).expanduser().resolve(); model=Path(model_dir).expanduser().resolve()
        adapter=Path(adapter_dir).expanduser().resolve(); dataset=(root/'preflight-canary.quarantine').resolve()
    except (TypeError,OSError,RuntimeError) as exc:
        raise ValueError('invalid canary paths') from exc
    model_id,revision=_MODELS[model_key]
    if (not model.is_dir() or model.name!=revision or adapter.exists() or adapter==root or
            any(a==b or a in b.parents or b in a.parents for i,a in enumerate((model,adapter,dataset))
                for b in (model,adapter,dataset)[i+1:])):
        raise ValueError('canary model, dataset or output path differs')
    audit_args=dict(root=root,forbidden_tasks=forbidden_tasks,tokenizer=tokenizer,render=render,
                    expected_dataset_sha256=expected_dataset_sha256,load_local_dataset=load_local_dataset)
    checked=audit_canary_loader(**audit_args)
    if checked.get('status')!='canary_loader_consistent_unverified' or checked.get('training_allowed') is not False:
        raise ValueError('canary dataset audit differs')
    config={'model':str(model),'train':True,'fine_tune_type':'lora','optimizer':'adam','data':str(dataset),
            'seed':seed,'num_layers':2,'batch_size':1,'iters':1,'val_batches':0,'learning_rate':0.0001,
            'steps_per_report':1,'steps_per_eval':2,'resume_adapter_file':None,'adapter_path':str(adapter),
            'save_every':1,'max_seq_length':2048,'grad_checkpoint':False,
            'lora_parameters':{'rank':8,'scale':2.0,'dropout':0.0},'mask_prompt':True,'report_to':None}
    if audit_canary_loader(**audit_args)!=checked or adapter.exists():
        raise ValueError('canary inputs changed during proposal')
    return {'status':'technical_canary_proposed_unverified','execution_ready':False,'training_allowed':False,
            'promotion_allowed':False,'candidate_reuse_allowed':False,'independently_verified':False,
            'model_key':model_key,'model_id':model_id,'model_revision':revision,'data_source':'fixed_synthetic_canary',
            'canary_dataset_sha256':expected_dataset_sha256,'identity_hashes':dict(identity_hashes),
            'resource_limits':dict(resource_limits),'config':config,
            'blockers':['separate_canary_registration','model_and_runtime_attestation','process_containment',
                        'independent_reload_verification','separate_canary_authorization'],
            'evidence_type':'caller_supplied_loader_tokenizer_inventory_and_identity_claims'}
