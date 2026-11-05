"""Recount source and AST repeats among all model emissions, including invalid code."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
from src.arc.ast_fingerprint import stable_ast_sha256
from src.arc.e6g_novelty import source_hash
from src.arc.http_backend import final_answer
from src.pipelines.e6g import _seal_g, _verify_g
from src.pipelines.e6f import _sha


def recount(campaign: Path, output: Path) -> dict:
    result = {}
    for stage in ('8b', '14b', '32b'):
        root = campaign / stage
        if not (root / 'seal.json').is_file(): continue
        _verify_g(root)
        seen_source, seen_ast = set(), set()
        raw_repeats = ast_repeats = emissions = 0
        rows = []
        for path in sorted((root / 'slots').glob('*/logs/e6/llm_responses.jsonl')):
            for line in path.read_text().splitlines():
                event = json.loads(line)
                code = final_answer(event['generated_text']).strip()
                raw = source_hash(code)
                try: tree = stable_ast_sha256(code)
                except (SyntaxError, ValueError, TypeError, RecursionError): tree = None
                raw_duplicate, ast_duplicate = raw in seen_source, tree is not None and tree in seen_ast
                raw_repeats += raw_duplicate; ast_repeats += ast_duplicate; emissions += 1
                seen_source.add(raw)
                if tree is not None: seen_ast.add(tree)
                rows.append({'candidate': event['candidate'], 'source_sha256': raw, 'ast_sha256': tree,
                             'prior_same_source': raw_duplicate, 'prior_same_ast': ast_duplicate})
        result[stage] = {'emissions': emissions, 'repeated_source_emissions': raw_repeats,
                         'repeated_ast_emissions': ast_repeats, 'rows': rows,
                         'source_stage_seal_sha256': _sha(root / 'seal.json')}
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    (output / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    (output / 'report.md').write_text('# E8 model emission repeats\n\nCounts include invalid candidates and are separate from duplicates against the verified initial pool. Finite behavior is not inferred from code identity alone.\n')
    for path in output.iterdir(): os.chmod(path, 0o600)
    _seal_g(output)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('campaign', type=Path); p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); print(json.dumps(recount(a.campaign, a.output), indent=2))


if __name__ == '__main__': main()
