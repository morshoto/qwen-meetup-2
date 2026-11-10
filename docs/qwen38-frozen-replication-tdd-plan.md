# TDD Plan: Frozen Gen 0/Gen 1 replication (#18)

**Type:** research feature. **Complexity:** high. **Issue:** https://github.com/morshoto/qwen-meetup-2/issues/18

Freeze the exact #16 experiment, independently audit its reserved validation parents and register a new disjoint allocation before predictions or target-answer access. Then evaluate both frozen arms under matched settings and report complete and incomplete paired outcomes. The existing 50 tasks are exploratory, not new validation. No training, solver tuning, or official 120-task evaluation is in scope.

> “Freeze the Issue #16 / PR #17 Qwen3.8-27B Weight Optimization outcome, then evaluate the exact frozen Gen 0 and Gen 1 on tasks not used to train, tune, select, or inspect their results.”

| Behaviour | First test seam | Boundary |
| --- | --- | --- |
| Pin prior code, adapter, model, inputs, training and report evidence | `tests/arc/test_qwen38_replication_freeze.py` | Changed, missing or ambiguous evidence fails closed; historical bytes remain unchanged. |
| Audit validation parents against all known IDs, content fingerprints, lineage and exposures | `tests/arc/test_qwen38_replication_allocation.py` | ID, fingerprint and lineage collisions reject; unknown exposure is explicit. |
| Freeze descriptor and prohibit answers in inference | `tests/arc/test_qwen38_replication_registration.py` | Both arms, task set, prompt, seeds, model, adapter, budgets and source hashes are immutable. |
| Supervise resumable adapter OFF/ON predictions | `tests/arc/test_qwen38_replication_run.py` | Crashes and ambiguous starts reconcile; identity mismatch or drift fails closed; no oracle available to workers. |
| Replay and score exact paired receipts | `tests/arc/test_qwen38_replication_report.py` | Missing is incomplete, duplicate raw attempts counted, paired discordance and exact uncertainty reported. |
| Show provenance and honest status in notebook | `tests/arc/test_qwen38_replication_notebook.py` | No missing-as-zero, leaked answers or unsupported confirmatory claim. |

## Test-first cycles

1. **Red:** write `test_freeze_rejects_wrong_adapter_or_report_identity` with source artifacts and changed hashes. Run `python -m pytest tests/arc/test_qwen38_replication_freeze.py -q`, expecting missing module/identity rejection failure. **Green:** add only a freeze verifier with strict identities and immutable receipts. **Refactor:** extract common digest checks only after green. Triangulate missing report, altered base, source revision and original artifact immutability.
2. **Red:** add allocation tests for direct parent, identical challenge/answer fingerprint and transitive lineage overlap, and missing exposure evidence. **Green:** add bounded offline audit; preserve unknown historical/pretraining exposure and label exploratory where necessary. **Refactor:** reuse existing allocation fingerprint helpers without weakening protections. Verify the 25 reserved parents before any selection.
3. **Red:** test descriptor rejects mismatched arm settings, changed task allocation or answer-bearing request. **Green:** register paired frozen identity, immutable task IDs and budgets before inference. **Refactor:** share validated identities with run supervisor. No model predictions while this step is incomplete.
4. **Red:** exercise complete, duplicate, interrupted and ambiguous attempts for each arm using fake workers and private journals. **Green:** reuse the existing supervisor and direct-grid worker where possible with OFF/ON identity and source-bound replay. **Refactor:** retain prior formats untouched and re-run scheduler/worker regression tests.
5. **Red:** require accurate paired counts, validity, errors, duplicates, incomplete denominators and exact two-sided discordance test for known small cases. **Green:** produce offline-only report after both arms terminate, using the independently verified permitted answers. **Refactor:** isolate statistical calculation and preserve raw receipts.
6. **Red:** test notebook's missing/complete output and provenance labels. **Green:** display verified report without raw candidates or labels. **Refactor:** execute from a fresh kernel and review claims.

## Verification and review

Run each new focused file through `/Users/shotomorisaki/Engineering/qwen-meetup-2/artifacts/qwen38-local-env/venv/bin/python -m pytest <file> -q`, then the full `tests/arc/test_qwen38*.py -q` suite and fresh-kernel notebook. Inspect identity hashes, source receipts, disjointness and both arms' termination before reporting; run real inference only after the preregistration and exposure audit pass. Preserve any incomplete or blocked evidence and never report a score for work not executed. Commit each red and green step with short `update:` subjects and open a stacked PR on #17.

Main risks: unrecorded exposure and semantically related tasks cannot be ruled out by hashes; record evidence and avoid confirmatory language if the audit cannot establish independence. Historical FP8 data make this cross-run distillation, not demonstrated RSI. An interrupted worker must not be silently repeated or scored as incorrect. Do not read evaluation labels before preregistration or supply them to a worker.
