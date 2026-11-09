# Qwen3.8 exploratory weight optimization implementation plan

Issue #16 is a new experiment; the earlier local comparison and Phase B registrations are immutable. First establish a bounded, independently invocable synthetic canary against the pinned 27B snapshot and toolchain. Record command, full log, model hashes before and after, loss, adapter changes and reload/logit evidence, resource observations and cleanup; a failed or incomplete check blocks collection. Synthetic success is not ARC improvement.

Next register a seeded fresh public Training pool and separate validation set, excluding the existing 50, official Evaluation, previous reservations, known exposures and matching content/lineage. Freeze identities, inputs, prompts, budgets and incomplete allocation before generating. Test overlap, immutable registration and leakage boundaries first.

Collect ordinary full-task prompts breadth first, preserving every raw response and attempt; independently verify against permitted solutions offline and deduplicate parent/content lineage. Audit historical records with source provenance and the same exclusions. Test that only actual verified model outputs enter training and that validation answers never become targets.

After the canary, freeze the small-data gate and training configuration. Train only with at least five verified unique eligible parents after token filtering; otherwise report insufficient data without a score. Verify adapter hashes, tensor changes and reload/logit deltas. If trained, evaluate paired adapter-off/on on the existing exploratory 50 and report incomplete denominators, gains/losses and source composition. Do not claim recursion or generalization from a parameter update.

## Collection implementation status

The new path now has separate full-task request/capture, breadth-first scheduling, a private append-only attempt journal, offline fresh verification, and a pinned-canary bundle replay. A synthetic integration test exercises their data flow without ARC examples or model inference. These components are not yet a dispatchable experiment: they still need a source-bound execution registration, supervised model worker and process-group cleanup, durable verifier receipts and reconciliation, resource/time enforcement across the actual subprocess, and source-bound SFT replay. The previously scored comparison and historical proxy collector remain unchanged.
