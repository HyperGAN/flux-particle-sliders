# Flux: completed provisional RAW32 tie

As of 2026-11-09, **G0.012 and G0.001 tie at0.0** for soft diffuse illumination
on the same native512 teapot case, seed7/four native sampling steps. Each arm
completed32 updates at horizon128/batch2, ordinary rank4/alpha1 LoRA. There is
**no particle bank** or prior LR. Actual D is0.018000000000000002 for both arms.
This is a completed scoped **provisional model-instrument comparison**;
`trait_qualified:false` remains explicit. No unique LR/default/global optimum or
released-weight claim follows.

| G | Saved OFF level | Saved ON level | Normalized ON−OFF gain |
|---|---|---|---|
| 0.012 | 4/4 | 4/4 | 0.0 |
| 0.001 | 4/4 | 4/4 | 0.0 |

The original0–4 levels are normalized by4, then ON minus matched OFF is averaged
with equal case weights (here one case, weight1). Every image received the maximum
intensity, so this coarse instrument is saturated on the comparison. There is
no formal trait, interval or direction qualification. Preserve the zero tie and
its limits; do not introduce anchors, rerating, fresh confirmation or a forced
winner merely to report the completed result. The saved OFF first was reused;
only the genuinely missing new ON observation was added.

Both authentic FULL32 candidates separately passed their qualified categorical
catastrophic guards. That is an eligibility check for catastrophic failure, not
an effect score or formal trait validation. Weak/absent illumination gain is not
failure. The [portable summary](finished-output-rate.json) retains per-case
levels, exact rate products, original result/guard hashes and RAW/FULL metadata.
Retained checkpoints, teacher tensors and original first responses are private;
this repository does not bundle a trained slider, public weight download or
standalone campaign reproduction.

## Distinct execution histories

The G0.012 arm used a CUDA critic optimizer for update1 and the unchanged official
FP32/FP64 CPU critic optimizer for updates2–32. The G0.001 arm used that same CPU
optimizer from fresh update1 through32. Native forwards, full rank3 penalty and
original RNG remained on GPU. These are honest saved **execution scopes**, not a
bitwise-CUDA or LR-only causal-isolation claim. Both reused all16 original teacher
cache rows; the exact2→32 continuations ran only30 missing updates and did not
initialize again. Original FULL/cache/init/update counts and prior failures stay.

The CPU-optimizer and range-continuation wrappers are execution-owned Source,
not public CLI flags. Their FULL schemas remain distinct. Do not feed a retained
CPU-range FULL into the ordinary public loader, retag it, or invent a
`--cpu-critic` option. The summary pins their actual evidence; it does not change
the public checkpoint contract or move an old source-bound RAW to a new factory.

## Actual public options and how-to

The current native project is [native-candidate](../native-candidate/README.md),
namespace `ps_flux_native95`, PG0.9.5/core0.14.1 at the captured core41f Source.
The original root PG0.9.4 prerequisite remains separate. The two literal
`SessionConfig.options` fragments accepted by the native product are:

```json
{"lr": 0.012, "d_lr_mult": 1.5, "prior_lr_mult": null, "ema_decay": null}
{"lr": 0.001, "d_lr_mult": 18.0, "prior_lr_mult": null, "ema_decay": null}
```

G=`lr`, D=`lr*d_lr_mult`; the product forwards these exact G/D literals through
`ConditionalPG095(recipe_overrides=...)`. A non-null `prior_lr_mult` is refused:
no bank is inserted. The official `adapter_gan` objective, complete packed
rank>=3 penalty and optimizer law are unchanged. Omitted options preserve upstream
defaults, not optimized rates. Copy saved literals rather than rounded quotients.

For a genuinely new cohort, use the actual
[SessionConfig example](../native-candidate/README.md) with authenticated staged
model/cache paths and hashes, native height/width512, text_tokens256,
`steps=128` (fixed horizon), batch2/seed7 and the declared source/init policy.
`--stop-after 32` means32 committed updates within horizon128. The existing public
commands are:

```sh
flux-native95 train --config YOUR-NATIVE-CONFIG.json --output NEW-RUN --stop-after 32
flux-native95 train --config SAME-NATIVE-CONFIG.json --output NEW-CONTINUATION --resume EXACT-ORDINARY-LATEST-FULL.pt --stop-after 32
flux-native95 render-pair --artifact SOURCE-MATCHED-RAW.pt --model-dir STAGED-SCHNELL --model-lock-sha256 LOCK_SHA --output NEW-PAIR --prompt 'A red ceramic teapot on a wooden table, crisp directional afternoon window light, detailed photograph.' --steps 4 --seed 7 --strength 1 --device cuda:0 --precision bfloat16
```

These demonstrate real [CLI options](../native-candidate/src/ps_flux_native95/cli.py),
not instructions to repeat completed arms or a claim the public CLI reproduces
the retained CPU-range execution. Ordinary continuation uses the same config and
complete latest FULL (both optimizers/RNG/init, no RAW resume) and a fresh output
directory. Only genuinely missing updates may run. Native rendering reuses one
source-matched frozen DiT, original scheduler/packing/VAE/text encoders and matched
OFF/ON seed. Schnell has no guidance embedding; its captured true CFG1 uses
native guidance_scale0. This is finished-image inference, not a hidden-field
substitute. Model/resource admission and hosted ordinary Generate are not supplied
by these commands.

Fresh product `open_training(config)` declares actual intent before allocation
and optimizers. Scoped official QR preserves the HNDL down Kaiming(a=1) host RMS,
zero-UP/KEEP, critic host laws, original Parameters and frozen pretrained base;
there is no bank/R2 draw to invent. `open_training(config, resume=FULL)` validates
and reuses the original receipt, never initializing restored weights. RAW
inference never initializes. Weight EMA is disabled for all added weights;
official scalar optimizer running statistics remain unchanged.

## Ranking and reuse

Use only the equal-case mean finished intended-effect ON minus matched OFF for
LR ranking. Preserve negatives, zeros and exact ties; missing/error judgments
remain unmeasured, not zero or pass. Teacher/student SSE is diagnostic, never a
selector/tiebreak. Preservation, beauty, confidence and weak effect are not
ranking gates. The separate qualified catastrophic-only gate stops failed
candidates before further updates/resume/promotion.

Reuse completed searches, teacher cache, updates, RAW/FULL states, native images
and frozen first observations. No old qualification/proof replay, new confirmation
or extra validation loop is required. A new concept, budget or sampling profile
cannot inherit these scores or guards.
