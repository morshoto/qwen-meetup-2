# TDD plan: H100 single-model experiment (#11)

Issue: https://github.com/morshoto/qwen-meetup-2/issues/11. Feature, high complexity. Start with a failing registration test for explicit vLLM identity.

## Scope and behaviour

| Behaviour | Test seam | Boundary |
| --- | --- | --- |
| Select MLX or pinned vLLM in configuration and bind runtime to registration | single manifest unit tests | Reject unknown backend/version; preserve legacy config |
| Execute two direct-grid candidates using BF16 vLLM, thinking off, registered settings and local snapshot | backend tests with fake vLLM | Reject invalid requests; preserve raw output and token usage |
| Route supervised worker through selected runtime and record it in score evidence | runner and worker integration tests | Reject runtime mismatch; retain journal and cleanup semantics |
| Prepare and score the frozen 200 tasks | existing allocation/scoring tests and fixture integration | Preserve data/source/model identities and resumability |
| Set up Linux H100 and manually acquire pinned model and ARC data | documented manual procedure | Actual H100 inference and 200-task result deferred to Vast verification |

## Red-green-refactor cycles

1. Red: test configured backend/version registration, tamper detection and legacy defaults in `tests/arc/test_lt_single_manifest.py`; run that file and confirm failure. Green: validate and bind backend identity. Refactor only while green.
2. Red: fake-vLLM tests assert local model BF16 initialization, prompt template, thinking off, sampling, normalization and metrics. Green: implement backend. Refactor and rerun focused tests.
3. Red: runner/worker tests assert backend routing, version check, evidence and no regression for MLX. Green: connect process launch and score evidence. Rerun focused suites.
4. Document reproducible Linux setup, pinned snapshots, resume and scoring. Run `python -m unittest discover -s tests/arc -p 'test_lt*.py'`; report missing data/dependencies and GPU validation honestly. No training is in scope.

## Risks and completion

Model architecture and vLLM compatibility, GPU memory, process cleanup, and performance require a real H100 run. Fixtures cannot establish these. The current per-task supervised process loads the model repeatedly; quantify startup and improve only with equivalent recovery and identity guarantees. Do not claim the issue's full 200-task acceptance until the later authorized Vast run is scored. Keep the PR draft until hardware validation.
