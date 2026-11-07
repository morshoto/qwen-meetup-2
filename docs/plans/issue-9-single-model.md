# Issue 9: configurable single-model ARC experiment

Issue: https://github.com/morshoto/qwen-meetup-2/issues/9

The existing Phase A registration binds its source and two fixed model inventories. Preserve that historical registration. Add a separate single-model registration using the same 200 tasks, direct-grid prompt, two candidates, journal and offline scoring, with a configured model identity and local file inventory. This is exploratory evidence, not an official benchmark or an authorization for a new run. H100/vLLM support is tracked separately in #11.

| Behavior | Test |
| --- | --- |
| One configured model and immutable revision are bound to a registration; invalid identity or file inventory fails closed | Configuration and registration unit tests |
| A registered model runs without the 14B/32B ordering dependency, with public tasks only, resumable journaling, and model identity preserved | Runner fixture integration tests |
| Offline scoring covers only the selected model, retains incomplete denominators and binds the registration | Scorer fixture tests |
| Historical two-model paths and registrations remain unchanged | Existing Phase A tests |

TDD cycles: (1) write failing registration/configuration tests, implement validation and hashing, then refactor green; (2) write failing public-export, execution and resume fixture tests, implement the runner, then refactor green; (3) write failing selected-model scoring tests, implement offline reporting, then refactor green. Run focused tests at each red and green step, then the affected Phase A suite. Document configuration, runtime requirements, registration, execution, recovery, scoring, and evidence limits. No live inference is necessary for fixture verification.
