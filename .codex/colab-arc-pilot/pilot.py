import datetime, hashlib, json, os, pathlib, signal, sys, time, traceback
from vllm import SamplingParams

root = pathlib.Path('/content/arc-colab-pilot-20261108')
sys.path.insert(0, str(root))
from src.arc.lt_direct_format import normalize_direct

ids = ['84db8fc4', 'a64e4611', 'e5c44e8f']
prompt_instruction = ('Return only JSON: an array of grids, one per test input. '
    'Syntax-only example for one output: [[[0,1],[2,3]]]. '
    'Do not copy the example. Compute the output for every actual test input. '
    'Each grid is an array of integer rows. No object wrappers, markdown, or explanation.')
export = json.loads((root/'export.json').read_text())
receipt = json.loads((root/'receipt.json').read_text())
if export['development_manifest_sha256'] != receipt['development_manifest_sha256'] or set(export['tasks']) != set(ids):
    raise RuntimeError('Development export identity mismatch')
if any(set(x) != {'input'} for t in export['tasks'].values() for x in t['test']):
    raise RuntimeError('Unexpected test fields; refusing to run')
if 'llm' not in globals() or 'tokenizer' not in globals() or 'model' not in globals():
    raise RuntimeError('Expected existing notebook model is not loaded')
if 'models--Qwen--Qwen3.8-27B-FP8/snapshots/c3a462cc3d38cf351f3b26cbd763469eeb74ef81' not in str(model):
    raise RuntimeError('Loaded model identity mismatch')

start=time.monotonic()
deadline=start+1800
out=root/'results.json'
if out.exists(): raise RuntimeError('Pilot output already exists; refuse overwrite')
record={'purpose':'development_only','status':'running','started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'development_manifest_sha256':export['development_manifest_sha256'],
        'export_sha256':hashlib.sha256((root/'export.json').read_bytes()).hexdigest(),
        'source_commit':'edcd2cf69bc7496ff282032b6ba45d9a872ca1be',
        'model_id':'Qwen/Qwen3.8-27B-FP8','revision':'c3a462cc3d38cf351f3b26cbd763469eeb74ef81',
        'gpu':'NVIDIA A100-SXM4-40GB','vllm_version':'0.33.0','max_model_len':2048,'temperature':0.6,
        'top_p':1.0,'top_k':-1,'seed_base':20261111,'max_tokens_requested':4096,'candidate_count':2,
        'thinking':False,'deadline_seconds':1800,'records':[]}
def save():
    record['elapsed_seconds']=round(time.monotonic()-start,3)
    tmp=out.with_suffix('.tmp')
    with tmp.open('w') as f:
        json.dump(record,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,out)
def alarm(signum, frame): raise TimeoutError('30-minute pilot deadline reached')
old_handler=signal.signal(signal.SIGALRM,alarm)
signal.setitimer(signal.ITIMER_REAL,1800)
save()
try:
    for task_id in ids:
        task=export['tasks'][task_id]
        payload=json.dumps({'instruction':prompt_instruction,'task':task},sort_keys=True,separators=(',',':'))
        formatted=tokenizer.apply_chat_template([{'role':'user','content':payload}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
        n_prompt=len(tokenizer.encode(formatted,add_special_tokens=False))
        max_tokens=min(4096,2048-n_prompt)
        if max_tokens<=0: raise ValueError(f'{task_id}: prompt exceeds context')
        for candidate in range(2):
            if time.monotonic()>=deadline: raise TimeoutError('30-minute pilot deadline reached')
            started=time.monotonic()
            params=SamplingParams(temperature=0.6,top_p=1.0,top_k=-1,seed=20261111+candidate,max_tokens=max_tokens)
            response=llm.generate([formatted],params,use_tqdm=False)[0]
            raw=response.outputs[0].text
            row={'task_id':task_id,'candidate':candidate,'seed':20261111+candidate,'raw':raw,
                 'prompt_tokens':len(response.prompt_token_ids),'completion_tokens':len(response.outputs[0].token_ids),
                 'max_tokens':max_tokens,'finish_reason':response.outputs[0].finish_reason,
                 'generation_seconds':round(time.monotonic()-started,3)}
            try:
                row['outputs']=normalize_direct(raw,len(task['test']));row['valid']=True;row['parse_error']=None
            except (ValueError,TypeError) as exc:
                row['outputs']=None;row['valid']=False;row['parse_error']=str(exc)
            record['records'].append(row);save()
            print(json.dumps({k:row[k] for k in ('task_id','candidate','valid','prompt_tokens','completion_tokens','generation_seconds','finish_reason')}),flush=True)
            if row['finish_reason']=='length':
                record['status']='stopped_truncated';record['error']='Response reached its available context limit';break
        if record['status']!='running': break
    else: record['status']='complete'
except Exception as exc:
    record['status']='error';record['error']=f'{type(exc).__name__}: {exc}';record['traceback']=traceback.format_exc()
finally:
    signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,old_handler)
    record['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();save()
    print('PILOT STATUS:', record['status'], 'records:',len(record['records']), 'elapsed:',record['elapsed_seconds'],flush=True)
