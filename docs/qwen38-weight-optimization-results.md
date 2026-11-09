# Qwen3.8 weight optimization: exploratory results

The paired evaluation is **in progress**. The training result below establishes
a parameter update; it does not yet establish any change in ARC accuracy.
This is a single mixed-source fine-tuning round, not demonstrated recursive
improvement or a pure 4-bit self-training result.

## Registered data and collection

The separately registered execution descriptor is
`research/rsi/qwen38-weight-execution-v1.json` (SHA-256
`08e4c2cabaca7f331c43a5bb12ddf5e1924ac68e573e3819e1e685044cf1038b`).
It records 40 fresh generation parents and 25 separate validation parents,
after the registered exclusions. The previously examined 50-task comparison
set is protected from training and validation; it is exploratory, not an
untouched validation set. Official evaluation tasks are excluded. Known
lineage was checked, but historical and pretraining exposure cannot be ruled out.
See [the allocation audit](qwen38-weight-allocation-audit.md) and
[protocol](qwen38-weight-optimization-plan.md) for the selection and limits.

Fresh collection recorded 294 attempts: 7 verified-correct, 243 incorrect,
44 invalid, and no errors. The historical FP8 source was replayed from its
saved candidates and checked against the same exclusions and permitted
Training solutions. The resulting SFT manifest has 23 records before the
actual tokenizer length filter and 14 unique parents after it: 2 current
4-bit and 12 historical FP8. Nine records exceeded the 2048-token limit,
including five of the seven fresh records. Fresh-only data therefore did not
meet the five-parent gate. Historical FP8 to local 4-bit training is
cross-run distillation and carries source and selection bias.

The private SFT manifest SHA-256 is
`52d6f734d61dc1aeb90dfc2160246ea62e6d775b58b9756e83e1cb65cb3379a0`;
raw generations and target grids remain in the private run artifacts. Source
replay checked the bundle against the recorded evidence before training.

## Compatibility and real-data training

The synthetic one-step 27B canary passed before collection; its tensor,
reload, base-hash and process evidence is documented in
[the canary receipt](qwen38-issue16-canary.md). It is a technical compatibility
check, not an ARC experiment.

The real adapter used the pinned MLX 4-bit model revision
`10c35caafbb80f7dc6a7a432cdd11af10a6d4818`, rank 8, batch size 1,
learning rate 2e-5, seed 20261111, 2048-token maximum, gradient
checkpointing and completion-only loss. All 20 training losses were finite;
training elapsed 621.76 seconds. The recorded supervisor samples show a
minimum 30% free memory and zero swap bytes during training. The training
process group and separate reload probe process group were gone after exit.

The adapter SHA-256 is
`b363fd2bd15114577d96141b2e03500e7d3d241a4b4a64b6176f25dbb46ac9f6`;
the training-result SHA-256 is
`4719ebaf28fa19144be919a0efdb100af14224f79a688c2b9178a37ae86267d9`.
Compared with saved initial tensors, 992 LoRA tensors changed. The separate
reload probe recorded a maximum absolute logit difference of 0.125. These
measurements establish an actual adapter update, not generalization.

## Paired evaluation

Pending completion and offline scoring. The first evaluation attempt stopped
before model loading because the adapter directory permissions failed the
private-directory check; its failure evidence was retained. A separately
registered recovery uses the same adapter bytes and matched evaluation
settings after correcting directory permissions. No accuracy is inferred from
either the failure or the running recovery.
