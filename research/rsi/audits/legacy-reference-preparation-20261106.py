import hashlib,json,os,stat,subprocess,sys,time
from pathlib import Path
from src.arc.rsi_lineage_audit import probes_sha256
from src.arc.rsi_reference_pool import build_legacy_duplicate_reference
from src.arc.sandbox import DockerSandbox
key=sys.argv[1];root=Path.cwd().resolve();draft_path=root/'research/rsi'/f'legacy-reference-preparation-{key}.json';draft=json.loads(draft_path.read_text())
metadata_path=root/'research/rsi/audits/historical-candidates'/f'{key}-static-reference.json';metadata_raw=metadata_path.read_bytes();meta=json.loads(metadata_raw)
index=json.loads((root/'research/rsi/audits/historical-candidates'/f'{key}.json').read_text());e9_path=root/index['e9_run']/'existing_oracles.jsonl';e9_raw=e9_path.read_bytes()
assert hashlib.sha256(metadata_raw).hexdigest()==draft['historical_reference_sha256']
assert hashlib.sha256(e9_raw).hexdigest()==meta['e9_existing_oracles_sha256'] and len(e9_raw.splitlines())==1
record=json.loads(e9_raw)['record'];source=record['messages'][-1]['content'];assert hashlib.sha256(source.strip().encode()).hexdigest()==draft['source_sha256']
gen=json.loads((root/'research/rsi'/f'oracle-generation-registration-{key}-draft.json').read_text());pm=gen['provisional_probe_candidates']['private'];probe_path=Path(pm['local_path']);info=probe_path.lstat()
assert stat.S_ISREG(info.st_mode) and stat.S_IMODE(info.st_mode)==0o600 and info.st_uid==os.getuid()
probe_raw=probe_path.read_bytes();probes=json.loads(probe_raw)
assert hashlib.sha256(probe_raw).hexdigest()==draft['private_probe_file_sha256'] and probes_sha256(probes)==draft['private_probe_canonical_sha256']
assert hashlib.sha256((root/'src/arc/rsi_reference_pool.py').read_bytes()).hexdigest()==draft['reference_builder_sha256']
assert hashlib.sha256((root/'src/arc/sandbox.py').read_bytes()).hexdigest()==draft['sandbox_wrapper_sha256']
image_id=subprocess.check_output(['docker','image','inspect','arc-sandbox:local','--format','{{.Id}}'],text=True,timeout=10).strip();assert image_id==draft['sandbox_image_id']
base=root/'artifacts/rsi-legacy-reference-preparation-20261106-v1';base.mkdir(mode=0o700,exist_ok=True);os.chmod(base,0o700);run=base/key;run.mkdir(mode=0o700)
start=time.monotonic();deadline=start+draft['limits']['wall_seconds']; sandbox=DockerSandbox(image=image_id,timeout=2,memory_mb=128,cpus=1,pids=32,remaining=lambda:max(0,deadline-time.monotonic()))
status={'status':'legacy_reference_preparation_incomplete','model_key':key,'eligible_for_training':False,
        'independently_verified':False,'model_calls':0,'training_steps':0,'private_outputs_exposed':False,
        'registration_sha256':hashlib.sha256(draft_path.read_bytes()).hexdigest()}
try:
 result=build_legacy_duplicate_reference(model_key=key,source=source,metadata_bytes=metadata_raw,
  expected_metadata_sha256=draft['historical_reference_sha256'],private_probes=probes,
  expected_probe_sha256=draft['private_probe_canonical_sha256'],sandbox=sandbox,max_sandbox_starts=draft['limits']['sandbox_starts'])
 assert result['sandbox_starts']==10
 assert hashlib.sha256(probe_path.read_bytes()).hexdigest()==draft['private_probe_file_sha256']
 assert hashlib.sha256(e9_path.read_bytes()).hexdigest()==meta['e9_existing_oracles_sha256']
 assert subprocess.check_output(['docker','image','inspect','arc-sandbox:local','--format','{{.Id}}'],text=True,timeout=10).strip()==image_id
 status.update(status='legacy_reference_preparation_consistent_unverified',reference=result)
except Exception as exc:
 status['failure_type']=type(exc).__name__
status['elapsed_seconds']=time.monotonic()-start
status['limitations']=['local sandbox execution and fingerprints require independent audit','not an RSI training candidate',
                       'no automatic retry or budget extension','sampled behavior does not establish semantic equivalence']
raw=(json.dumps(status,sort_keys=True,separators=(',',':'))+'\n').encode();fd=os.open(run/'result.json',os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
with os.fdopen(fd,'wb') as out:out.write(raw);out.flush();os.fsync(out.fileno())
print(json.dumps({'model_key':key,'status':status['status'],'failure_type':status.get('failure_type'),
                  'sandbox_starts':status.get('reference',{}).get('sandbox_starts'),'elapsed_seconds':status['elapsed_seconds'],
                  'private_outputs_exposed':False}),flush=True)
