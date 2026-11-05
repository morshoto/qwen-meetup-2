# E4: diagnose and improve transformation search

Completed 2026-11-01 using the saved E2/E3 artifacts and a separate ten-task synthetic suite. E4 did not modify the E2/E3 run artifacts or use heldout ARC labels to select a strategy.

## E4-A: saved-trace diagnosis

E2 ran five ARC tasks against four strategies. Every strategy had zero complete train matches and zero heldout passes. Replaying the saved S0 programs against the public training pairs found 0/38 pair matches and two execution errors on the same task. That program indexed columns 8 and 9 on an 8-wide grid, raising `IndexError`. The saved S0 trace had dropped non-list execution feedback; the offline replay records the missing error detail. New E2 traces now retain a bounded error kind and the final error line.

S0 generated two code attempts for each of five tasks. Three of those five repairs returned the exact same code. The other two changed code but still matched no training outputs. S1 and S3 each generated 15 candidates: 12 unique parsed candidates and three duplicate candidates. The unique candidates passed static checks and executed but matched 0/44 training pairs per strategy. S2's five analysis responses were truncated before code generation. The traces do not save S0's analysis text or an explicit S1/S3 rule hypothesis, so they cannot distinguish a conceptual rule error from an incorrect Python implementation for every task.

E3 Collection attempted 19 tasks and accepted no solutions. Sixteen selected candidates were rejected by the independent verifier; all 16 had 0 training matches (10 had 3 train examples, four had 2, and two had 4). Two tasks had no candidate selected, and the last stopped before dispatch at the budget cap. The registered S1 rule selects the first statically safe candidate that executes on every train input; it does not require any exact train match. E2's S1 selection was inconclusive, so this was not evidence for adopting S1 as a production method.

## E4-B: ten synthetic problems

The fixture contains ten hand-specified transformation families: three color maps, horizontal and vertical reflection, 90° and 180° rotation, two translations, and reflection plus recoloring. Each task has three train pairs and two separate holdout inputs. The challenge and answer files are separate; the agents receive only demonstrations and test inputs. The heldout answer file is read only after the predictions have been produced.

| Method | Full train tasks | Train pairs | Holdout inputs | Calls | Reserved tokens | Sandbox starts |
|---|---:|---:|---:|---:|---:|---:|
| Existing code agent, legacy prompts | 2/10 | 6/30 (20%) | 4/20 (20%) | 28 | 119,188 | 98 |
| Existing code agent, structured prompt experiment | 1/10 | 3/30 (10%) | 2/20 (10%) | 29 | 124,443 | 95 |
| Train-only operator search | 10/10 | 30/30 (100%) | 20/20 (100%) | 0 | 0 | 0 |

The two legacy-agent full matches were the second color-map task and the 180° rotation task. Its failing code included ragged outputs on horizontal reflection and incorrect output dimensions on vertical reflection. The structured-prompt experiment addressed a real prompt conflict: the analysis request asked for a concise rule summary while the shared system prompt said not to explain. It also included the failing input alongside actual and expected outputs during repair. On this ten-task run, that change reduced train matches from 6/30 to 3/30, so it remains opt-in and is not the default.

## E4-C: operator-search prototype

`src/arc/operator_search.py` enumerates eight grid symmetries, shifts of up to two cells in each direction, and a consistent global color mapping learned from the train pairs. It keeps only hypotheses that reproduce every training output, then selects the simplest by a fixed cost order. It produces grids directly, avoiding generated Python for this bounded transformation family. The comparison uses the same ten tasks and scores heldout inputs only after candidate selection.

The synthetic suite was constructed from this operator family, so the 100% result is a capability check for simple transformations, not evidence of arbitrary ARC generalization. The next promotion gate should use fresh ARC training demonstrations with the operator arm fixed in advance and no heldout-output feedback.

## Artifacts and reproduction

- Saved trace diagnosis: `artifacts/e4-synthetic-20261101/diagnosis-v2/diagnosis.md`
- Three-arm results: `artifacts/e4-synthetic-20261101/comparison-v2/report.md`
- Legacy baseline: `artifacts/e4-synthetic-20261101/baseline-v3/`
- Structured-prompt condition: `artifacts/e4-synthetic-20261101/baseline-structured/`
- Synthetic data generator: `scripts/generate_e4_synthetic.py`

To regenerate the fixed synthetic fixture and rerun the train-only comparison:

```bash
python scripts/generate_e4_synthetic.py
uv run --locked python -m src.cli.e4_operator_compare \
  --baseline-dir artifacts/e4-synthetic-20261101/baseline-v3 \
  --structured-dir artifacts/e4-synthetic-20261101/baseline-structured \
  --output-dir artifacts/e4-synthetic-20261101/comparison-rerun
```

The model baseline uses the guarded local MLX server and a fresh token ledger. Its server setup and bounded 200,000-token / 30-call limits are recorded in `artifacts/e4-synthetic-20261101/baseline-structured/manifest.json`. Repeated inference requires a new ledger path and a new output directory.

The full test suite passes: 115 tests and 18 subtests.
