"""Score journaled LT results offline, including incomplete task denominators."""
import argparse
import hashlib
import json
from pathlib import Path
from src.arc.data import load_arc
from src.arc.lt_batch_journal import BatchJournal
from src.arc.lt_benchmark_manifest import verify
from src.arc.lt_scoring import score_comparison

ROOT=Path(__file__).resolve().parents[2]


def score_directory(directory):
    manifest=json.loads((directory/'manifest.json').read_bytes());verify(ROOT,manifest)
    tasks,verifier=load_arc(ROOT/'data/raw/arc-agi_training_challenges.json', ROOT/'data/raw/arc-agi_training_solutions.json')
    if not set(manifest['task_ids']) <= set(tasks): raise ValueError('missing task')
    rows=[]; hashes={}
    for model in ('14b','32b'):
        path=directory/model
        if not path.exists(): continue
        journal=BatchJournal(path,manifest['task_ids'],manifest['sha256']+':'+model)
        journal.pending(1)
        for task in manifest['task_ids']:
            result=path/(task+'.result.json')
            if result.exists():
                raw=result.read_bytes();row=json.loads(raw)
                if row.get('model')!=model: raise ValueError('model identity differs')
                rows.append(row);hashes[model+':'+task]=hashlib.sha256(raw).hexdigest()
    report=score_comparison(manifest['task_ids'],rows,verifier)
    report['evidence']={'comparison_sha256':manifest['sha256'],'results_sha256':hashes,
                        'solutions_sha256':manifest['solutions_sha256'],'protocol':manifest['protocol']}
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--directory',type=Path,required=True)
    args=parser.parse_args();print(json.dumps(score_directory(args.directory.resolve()),sort_keys=True,allow_nan=False))


if __name__=='__main__':main()
