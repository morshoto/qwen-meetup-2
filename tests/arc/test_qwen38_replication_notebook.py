import json
from pathlib import Path

from src.arc.qwen38_rsi_comparison import digest

NOTEBOOK = Path('nb/results/18_qwen38_frozen_replication.ipynb')


def execute(monkeypatch, tmp_path):
    monkeypatch.setenv('QWEN38_REPLICATION_RUN_DIR', str(tmp_path))
    notebook = json.loads(NOTEBOOK.read_text())
    namespace = {}
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            exec(compile(''.join(cell['source']), '<notebook>', 'exec'), namespace)


def test_notebook_does_not_turn_missing_report_into_zero(monkeypatch, tmp_path, capsys):
    execute(monkeypatch, tmp_path)
    text = capsys.readouterr().out.lower()
    assert 'not available' in text and '0/25' not in text
    assert not list(tmp_path.iterdir())


def test_notebook_displays_only_bound_report_summaries(monkeypatch, tmp_path, capsys):
    registration = json.loads(Path('research/rsi/qwen38-replication-v1.json').read_text())
    report = {'status': 'incomplete', 'registration_sha256': registration['sha256'],
              'paired': {'baseline': {'complete_tasks': 2}, 'gen1': {'complete_tasks': 1},
                         'paired': {'complete': 1}}, 'uncertainty': None,
              'raw': 'SECRET-CANDIDATE'}
    report['sha256'] = digest(report)
    (tmp_path / 'report-fixture.json').write_text(json.dumps(report))
    execute(monkeypatch, tmp_path)
    text = capsys.readouterr().out
    assert 'incomplete' in text and "'complete_tasks': 1" in text
    assert 'SECRET-CANDIDATE' not in text
    report['status'] = 'complete'
    (tmp_path / 'report-fixture.json').write_text(json.dumps(report))
    execute(monkeypatch, tmp_path)
    assert 'not available' in capsys.readouterr().out.lower()


def test_notebook_records_verified_completed_results():
    notebook = json.loads(NOTEBOOK.read_text())
    markdown = '\n'.join(''.join(cell['source']) for cell in notebook['cells']
                         if cell['cell_type'] == 'markdown')
    assert 'bc6a45edb4850feac1f658707b215bb3e66cbd4b5e7b12ebd4bd5636b1690c4d' in markdown
    assert '4/25' in markdown and '1 gained' in markdown and '1 lost' in markdown
