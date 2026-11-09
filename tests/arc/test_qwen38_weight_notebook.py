import json
from pathlib import Path


def test_results_notebook_runs_without_artifacts(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv('QWEN38_WEIGHT_RUN_DIR', str(tmp_path))
    notebook = json.loads(Path('nb/results/17_qwen38_weight_optimization.ipynb').read_text())
    namespace = {}
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            exec(compile(''.join(cell['source']), '<notebook>', 'exec'), namespace)
    output = capsys.readouterr().out
    assert 'not run' in output.lower()
    assert 'Gen1' not in output or 'not run' in output.lower()
    assert not list(tmp_path.iterdir())


def test_results_notebook_displays_recovery_report(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv('QWEN38_WEIGHT_RUN_DIR', str(tmp_path))
    reports = tmp_path / 'evaluation/adapter-v2'
    reports.mkdir(parents=True)
    (reports / 'report-fixture.json').write_text(json.dumps({'status': 'incomplete', 'paired': {
        'baseline': {'complete_tasks': 50}, 'gen1': {'complete_tasks': 1}, 'paired': {'complete': 1}}}))
    notebook = json.loads(Path('nb/results/17_qwen38_weight_optimization.ipynb').read_text())
    namespace = {}
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            exec(compile(''.join(cell['source']), '<notebook>', 'exec'), namespace)
    output = capsys.readouterr().out
    assert "'complete_tasks': 1" in output
    assert 'Report status: incomplete' in output
