# Qwen3.8 weight optimization: exploratory results

The mixed-source adapter produced a parameter update and an exploratory gain
on the previously examined comparison set. This single fine-tuning round is
not demonstrated recursive improvement or a pure 4-bit self-training result.

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

The first evaluation attempt stopped
before model loading because the adapter directory permissions failed the
private-directory check; its failure evidence was retained. A separately
registered recovery uses the same adapter bytes and matched evaluation
settings after correcting directory permissions. The recovery finished all
100 attempts; all 100 worker receipts report successful exit and cleaned-up
process groups. Its termination receipt reports 13,317.38 seconds wall time.

| Measure | Gen 0, adapter off | Gen 1, adapter on |
| --- | ---: | ---: |
| First recorded candidate correct | 2/50 | 3/50 |
| Either recorded candidate correct | 2/50 | 5/50 |
| Valid candidate grids | 67/100 | 95/100 |
| Invalid candidate grids | 33/100 | 5/100 |
| Duplicate raw candidate pairs | 3/50 | 5/50 |
| Sum of recorded candidate runtimes | 6,001.76 s | 10,285.96 s |

All 50 task pairs were complete: four tasks gained, one was lost, one was
correct in both arms, and 44 were incorrect in both, using the either-candidate
criterion. There were no failed, partial, missing or incomplete tasks and no
evaluation errors. The two-candidate measure describes recorded attempts;
duplicate pairs and shared conditions preclude assuming independent samples.

The recovery descriptor SHA-256 is
`70fec8e84c0bf03e4bcc524399cd7551405b0708eb4ee9fffa2d2b8a746f3277`,
and the offline report SHA-256 is
`023ac3d4afa8982d43a24e2b328f4471e127a7c2f9a24109d582d9e89d193b70`.
The comparison suggests a gain on these already examined tasks, but does not
establish a generalization improvement, an official ARC score, or recursive
self-improvement. The historical component makes it cross-run distillation.
