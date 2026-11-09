# Issue #16: synthetic 27B compatibility canary

On 2026-11-09 the pinned local `mlx-community/Qwen3.8-27B-4bit` revision `10c35caafbb80f7dc6a7a432cdd11af10a6d4818` completed one synthetic SFT step with MLX 0.32.4, mlx-vlm 0.6.8 and datasets 5.1.0. The prompt was `Return one.` and the target was `1`; no ARC task or answer was used. This is technical compatibility evidence, not ARC improvement.

The seed was 20261111, rank 8, batch size 1, learning rate 2e-5, completion-only loss and gradient checkpointing. Training loss was 0.00962566. An instrumented trainer saved all initialized trainable tensors before optimization; comparison against the saved adapter found 496 changed LoRA tensors. The adapter reloaded in the pinned model and changed the probe's maximum absolute logit by 0.15625. All three base weight shard hashes matched the prior local comparison registration before training and remained identical after training and the reload probe. Both process groups had exited after their respective runs.

The trainer reported peak MLX memory of 16.920 GB. Sampling at five-second intervals observed a minimum 60% system free memory during training and zero swap in the samples; the reload probe sampled 84% free and zero swap. These are sampled system observations, not exact system peaks. Each subprocess was configured to stop at less than 10% free, more than 16 GiB swap, an unavailable memory probe, or 900 seconds. The training and reload processes completed in approximately 10.0 and 5.0 seconds respectively, excluding before/after hashing.

Evidence remains in the Git-ignored local directory `artifacts/qwen38-issue16-canary-capture-20261109/`. Its SHA-256 values are:

| Artifact | SHA-256 |
| --- | --- |
| registration.json | `9f8fbb1895ea9e10ada57c661c98f6edc5a074b385983248f48c45e83f08d85b` |
| train.log | `c1ae6bdbd756cde386440e2e71c275e20fbef3b17c627ce449e7a3bf67a2149b` |
| execution.json | `c46d6e6f2184102bd6b09665b1bd2922908855c415441654a76bf7337c7b1e33` |
| initial.safetensors | `5918c5ca9a48e73630a1043e47994f61e911938919d7725be0778c5c511ba981` |
| adapter/adapters.safetensors | `a7fb4e68dcab6998982f1487ad582700f13207d186b64afb6e8486309a57c6fe` |
| adapter/adapter_config.json | `8c58de57198438da131ccb7ad55db5a4a2dac90c74e32cc30fc763dfdd503ce0` |
| tensor-probe.json | `0e4e9c5649611866bad82ebfb0d6e5f2823d4f7bb5302aea275e7e81214b02f5` |
| probe.log | `64b758594b98a1473085cc28f61a4ca4735103e3c2bd6c9e33bfa44f72004126` |
| probe-execution.json | `c32f9914cae915846f2513713cfdec4e792f4e1a90a7294e79bac2dc34d4881a` |

An earlier uninstrumented execution is retained separately at `artifacts/qwen38-issue16-canary-20261109/`; its nonzero adapter alone cannot establish a tensor update. No fresh collection, real-data adapter, or paired Gen 1 score has been produced by this canary.
