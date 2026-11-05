# Technical canary save and reload review

Status: proposed procedure. No adapter has been trained or reloaded under this
procedure. A file-only comparison or a fixture does not demonstrate a model
reload. The 14B and 32B canaries require separate registrations, budgets,
reviews and authorization.

## After an authorized one-step canary

1. The independently observed training process must exit successfully after one
   finite-loss step within its registered limits. Record the trainer process,
   source and runtime identities, completed output stream, resource samples,
   termination evidence, adapter path and exact adapter digest. If process
   containment or resource observation cannot be established, report the result
   as incomplete.
2. Record a digest of the two adapter files. Run
   `inspect_adapter_structure(..., model_key=..., expected_sha256=...)` and
   `probe_adapter_products(..., model_key=..., expected_candidate_sha256=...)`.
   These checks validate file structure and can find an update witness; they do
   not prove training, reloading, or an RSI effect. An inconclusive product
   probe is not proof that no update occurred.
3. In a fresh, independently observed, offline process, verify the registered
   base model, runtime, adapter digest and resource conditions before loading.
   Load **only one** base model at a time with the saved adapter using pinned
   `mlx_lm.utils.load` and `tokenizer_config={'trust_remote_code': False}`.
   Record a separate wall-clock limit and memory, swap, disk and process-tree
   samples for the reload. Stop if the registered limits are exceeded or
   monitoring is unavailable. A 32B reload is not presumed to fit.
4. From the actual loaded model, use the pinned runtime's
   `mlx.utils.tree_flatten(model.parameters())` and select the names ending in
   `.lora_a` or `.lora_b`. Evaluate each tensor with `mlx.core.eval`, convert it
   to a NumPy array without changing shape or dtype, and pass the resulting
   name-to-array mapping to `compare_reloaded_factors` with the same model key
   and adapter digest. The comparison requires exactly the expected names,
   shapes, float32 values and digest, and rechecks the saved files. Keep the
   probe output free of ARC data and generated answers.
5. Record process exit, samples and identities and recheck the base and adapter
   digests after the process. An independent reviewer must examine the complete
   record, including whether the tensors actually came from that loaded model
   in the observed process. The comparison helper alone reports
   `reload_verified: false` because supplied arrays can be fabricated.

The synthetic adapter remains quarantined and cannot be promoted or used as
an RSI training candidate. A successful reload would demonstrate a technical
pipeline property only; it would not establish generation quality or benchmark
performance. Any missing evidence or out-of-limit observation prevents a
successful canary claim.
