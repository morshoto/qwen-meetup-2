"""Create and verify a local exploratory benchmark registration."""
import hashlib
import json
from pathlib import Path
from src.arc.lt_allocation import build_comparison_proposal
from src.arc.lt_live_backend import PROMPT

SOURCE_FILES = ('src/arc/agent.py','src/arc/data.py','src/arc/scoring.py','src/arc/lt_scoring.py',
                'src/arc/lt_direct_format.py','src/arc/lt_live_backend.py','src/arc/lt_worker.py',
                'src/arc/lt_public.py','src/arc/lt_batch_journal.py','src/arc/lt_supervised_batch.py',
                'src/arc/lt_task_supervisor.py','src/arc/lt_benchmark_manifest.py','src/arc/http_backend.py',
                'src/arc/sandbox.py','src/arc/lt_allocation.py','src/arc/lt_comparison.py',
                'src/arc/lt_attestation.py','src/arc/lt_development.py','src/arc/rsi_exploratory_split.py',
                'src/cli/lt_batch_worker.py','src/cli/lt_benchmark.py','src/cli/lt_benchmark_score.py')


def sha(data): return hashlib.sha256(data).hexdigest()

def canonical(value): return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()

def build(root: Path) -> dict:
    challenges=(root/'data/raw/arc-agi_training_challenges.json').read_bytes()
    registry=(root/'research/rsi/bounded-exposure-proposal.json').read_bytes()
    split=(root/'research/rsi/exploratory-split-proposal.json').read_bytes()
    development=json.loads((root/'configs/lt.phase-a.development.json').read_bytes())
    training=json.loads((root/'configs/lt.phase-b.training-pool.json').read_bytes())
    proposal,audit=build_comparison_proposal(challenges,registry,split,development,training,seed=20261108)
    saved=json.loads((root/'configs/lt.phase-a.comparison-proposal.json').read_bytes())
    if proposal != saved or len(training)!=100 or len(proposal['task_ids'])!=200:
        raise ValueError('allocation differs')
    source={name:sha((root/name).read_bytes()) for name in SOURCE_FILES}
    models={}
    for model in ('14b','32b'):
        observed=json.loads((root/f'research/rsi/audits/technical-model-{model}.json').read_bytes())['model']
        models[model]={key:observed[key] for key in ('model_id','revision','files_sha256','chat_template_sha256')}
    value={'protocol':'lt-phase-a-exploratory','task_ids':proposal['task_ids'],
           'challenges_sha256':sha(challenges),
           'solutions_sha256':sha((root/'data/raw/arc-agi_training_solutions.json').read_bytes()),
           'training_ids_sha256':sha(canonical(training)),
           'allocation_audit_sha256':sha(canonical(audit)), 'models':models,
           'source_sha256':source,'prompt_sha256':sha(PROMPT.encode()),
           'thinking':'off','temperature':0.6,'candidate_count':2,'seed_base':20261111,
           'seed_support':'unverified','max_tokens':4096,'task_timeout_seconds':1800,
           'mlx_memory_limit_bytes':48*1024**3,'batch_size':10,'mlx_lm_version':'0.30.7',
           'mlx_version':'0.32.4','swap_measurement':'unavailable',
           'official_public_evaluation':'reserved_for_phase_c',
           'unknown_pretraining_exposure':True,'unknown_project_exposure':True}
    return {**value,'sha256':sha(canonical(value))}


def verify(root: Path, manifest: dict):
    if manifest != build(root): raise ValueError('frozen benchmark registration differs')
