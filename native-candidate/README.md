# Native packed Flux PG0.11.1 candidate

This is a new, separately installable Source candidate. It preserves native
`[1,image_tokens,64]` velocity responses and the ordinary rank4/alpha1 HNDL
down/up LoRA from the published Flux prerequisite. It supplies a genuine shared
`ResponseBackend`, full native teacher cache, official PG0.11.1 `adapter_gan`
updates, exact FULL restore, RAW-only native inference and the official
Diffusers Flux finished-image pipeline. No response projection, subsampling,
token-axis reinterpretation or synthetic particle bank is used. The official
rank>=3 penalty branch remains unchanged. Frozen positive native velocity on a
neutral trajectory is this cohort's target; it does not migrate the historical
scheduler-step latent objective.

All17 random-tiny native CPU checks passed at measured Source commit
`01ac3bb2166656427252776b591ba47a82ba35d7` on2026-11-09 (0 failures/errors/skips,
1.546 seconds). This includes real tiny native FULL/resume/RAW/PIL contracts,
with no pretrained model or GPU use. See [the scoped actual evidence](docs/CPU17-VALIDATION.md).
Later retained actual pretrained RAW32 training and finished images support a
[completed provisional comparison](../docs/finished-output-learning-rates.md):
G0.012/G0.001 tie at0.0, one equal case, all OFF/ON levels4/4. Both corresponding
FULL32 catastrophic guards pass. The trait remains unqualified; differing
critic execution histories preclude an LR-only causal or bitwise-CUDA claim.
No unique/default LR, released weight or ordinary hosted Generate is established. The core dependency pins the shared runtime at
`ec1161f5d74481be3540cb0dfa63d0e8f8b54e90` (core0.15.2/PG0.11.1). The original preview remains at its archived source revision. Use a fresh
environment for the current runtime and keep old packets with their original dependencies.

Native model scope is the declared `black-forest-labs/FLUX.1-schnell` revision
`741f7c3ce8b383c54771c7003378a50191e9efe9`. A preparation owner must stage the
complete original local pipeline. Create `flux-model.lock.json` with schema
`flux-native-local-model-lock-v1`, that `model`/`revision`, and `files` mapping
every file in transformer, scheduler, VAE, both text encoders and both tokenizers
plus model_index.json to `{ "bytes": integer, "sha256": "..." }`. The loader
checks the complete named inventory, every whole-file identity, original model
config, factor8/latent16 VAE and native source. There is no download fallback.
A lock is a preparation receipt, not pretrained or native execution evidence.

Install this subdirectory in the new admitted environment and use
`flux-native check` for an allocation-free profile description. Framework
imports and model loads happen only in explicitly requested execution commands.
Builder provenance pins the original two-line graph, HNDL0.10.0 and the exact
clean external HyperGAN builder. No removed product bundle is required.

The fresh initializer admits only the exact HNDL0.10 `Linear` with
`equalized=False`, no bias, children, buffers or parametrizations. HNDL's class
subclasses `nn.Linear`; it is not the literal PyTorch leaf accepted by the shared
initializer. A parameter-only plain `nn.Module` holder registers the same
original weight Parameter with explicit a=1 Kaiming Uniform down/KEEP zero-up
declarations. This preserves host forward behavior, Parameter identity and the
shared helper's strict leaf boundary. No global initializer registration or
shared-core relaxation is introduced. Fresh/resume receipts remain bound to
the actual product initialization Source. All9 targeted HNDL regressions passed
as part of the17-check actual CPU run; original failed results remain retained.

Cache preparation accepts a JSON list of distinct `neutral`/`positive` strings:

```sh
flux-native cache --model-dir MODEL --model-lock-sha256 LOCK_SHA \
  --pairs pairs.json --output NEW_CACHE --height 512 --width 512 \
  --text-tokens 256 --steps 4 --seeds 7 19 --device cuda:0 --base-precision bfloat16
```

Every saved full latent/condition/positive and OFF response is content-bound.
Training never repeats teachers. The critic sees every native response token
and channel plus original image-position IDs. Its condition vector contains all
three original context tensors and raw time, with no lossy sketch. Only learned
critic features/logits are aggregated. The full packed rank3 tensor is passed
unchanged to the official penalty. G/D own FP32 trainable leaves; the frozen
base is explicitly FP32 or BF16. BF16 and accelerator placement are a new
declared numerical identity, not the old CPU FP32 prerequisite proof. Native
checkpointing remains disabled until a separately justified scale-safe path is
needed. Actual memory fit is for the execution owner to measure/admit.

Train through the product frontdoor, which binds actual fresh/restore intent
before factory and optimizer construction. The registered backend uses the
shared `SessionConfig`/`ResponseSession` contract. An unbound direct registry
factory call refuses rather than guessing whether a packet will later restore.

```python
from particle_sliders.lifecycle import SessionConfig
from ps_flux_native.lifecycle import open_training
from ps_flux_native.contracts import INITIALIZATION_POLICY

config = SessionConfig("flux_native_packed95", "my-new-flux-cohort",
    model_dir="MODEL", model_lock_sha256="LOCK_SHA",
    cache_dir="CACHE", cache_index_sha256="CACHE_SHA",
    device="cuda:0", base_precision="bfloat16",
    height=512, width=512, text_tokens=256,
    steps=32, batch=2, seed=7, ema_decay=None,
    pg_version="0.11.1", pg_source_profile="release-0.11.1",
    penalty_input_domain="auto", initialization_policy=INITIALIZATION_POLICY)
with open_training(config) as session:
    session.train()
    session.checkpoint("FULL.pt")
    session.export_inference("RAW.pt", selection="raw")
```

These are budgets/examples, not optimized settings or executed evidence.
Optional `lr` is the common G LR and `d_lr_mult` its D multiplier; exact supplied
scalar bits are forwarded unchanged to the official recipe. Omissions preserve
its defaults. This ordinary adapter has no prior/bank, so `prior_lr_mult` must be
omitted or null. The [saved provisional tie](../docs/finished-output-learning-rates.md) adds explicit
measured guidance without changing these defaults. Weight EMA is refused.
Fresh initialization uses the current shared scoped helper exactly once before
optimizers: LoRA down keeps its true Kaiming(a=1) RMS law under named QR seeds,
up remains zero/KEEP, and critic leaves retain their declared host RMS. Frozen
pretrained leaves are outside ownership. FULL restore and RAW inference retain
the original receipt and never call that helper; temporary constructor values
are replaced by exact saved state, never used to reset a loaded checkpoint.

Use `open_training(config, resume="FULL.pt")` for exact continuation of the
original fixed budget. The CLI counterpart is `flux-native train --config
config.json --output NEW_OUTPUT [--resume FULL.pt] [--stop-after N]`; its config
is `SessionConfig.as_dict()`. It saves each actual step as a distinct FULL and
one RAW export. Interrupted unsaved work is not claimed durable by the CLI.
Keep the artifacts and recorded update counts; never rerun completed LR work.

RAW-only finished rendering does not construct a cache, critic, optimizer, PG
game or initializer. The native public pipeline reuses the exact one bound DiT,
its scheduler, native packing/unpacking, original VAE and image postprocessing.
It loads both text encoders and VAE once and explicitly moves the full render
pipeline to the selected device, so admission must budget that placement.

```sh
flux-native render-pair --artifact RAW.pt --model-dir MODEL \
  --model-lock-sha256 LOCK_SHA --output NEW_PREVIEW --prompt "a windswept traveler" \
  --steps 4 --seed 7 --strength 1 --device cuda:0 --precision bfloat16
```

OFF/ON are real same-seed native PNGs, each with a fresh cloned scheduler;
strength and pipeline state restore after success or error. Geometry and text
length stay exact to the HNDL training graph. True CFG is1, Schnell has no
guidance embedding; native guidance_scale0 matches its public pipeline usage.
The RAW bytes, original packet checksum and full native/factory identity are
strictly checked. Partial outputs and truthful failure metadata are retained;
there is no retry or posthoc latent interpolation.

Pass an open RAW Python session to `ps_flux_native.scheduled_render.render`
with `step_gains` and the original render arguments. Supply one gain per native
step, with each `strength * gain` in `[0, 1]`. Read the actual sigma grid and
applied strengths from `image.info["strength_schedule"]`.
Constant gains matched the original finished pixels in the tiny native CPU
fixture.

For the measured **Ink Wash RAW18** panel at 512×512 with four native BF16
Euler steps, use nominal strength `2 / 3` and
`step_gains=[0.5, 0.5, 1.5, 1.5]`. The applied strengths were
`[1/3, 1/3, 1, 1]` on sigmas `[1, 0.75, 0.5, 0.25, 0]`. Across eight
matched cases, the mean finished Ink Wash effect relative to the shared
neutral outputs was +0.001493 for late-heavy strength, +0.000475 for
early-heavy strength, +0.000304 for equal-dose constant strength, and
+0.000696 for constant strength 1. All matched outputs passed independent
integrity checks. These settings apply to this checkpoint and tested sampling
profile. [Per-case scores and image hashes](docs/ink-wash-raw18-scheduled-strength.json).

### Activation-gated RAW rendering

Use the same open RAW session and original native flow renderer with an explicit
`LatentGatePlan`. In the single conditioned forward of each Schnell Euler
interval, the renderer reads the live `[1, image_tokens, 64]` packed latent
before the DiT's adapter and averages its spatial tokens in FP32. The shared
CFP or CLABS gate projects that vector onto the supplied unit direction. A
firing applies the declared nominal strength to the existing RAW LoRA for
that forward. Otherwise the step runs at exact OFF. The native sigma grid,
conditioning, scheduler and VAE still own generation. This uses the paper's
allocation equations for the native LoRA edit at diffusion intervals.

```python
from ps_flux_native.activation_gate import LatentGatePlan
from ps_flux_native.scheduled_render import render

plan = LatentGatePlan(
    core_wheel=verified_core_269_wheel,
    method="cfp",
    parameter=calibration.parameter,
    direction=flux_packed_channel_unit_direction,
    calibration_sha256=calibration_record_sha256,
)
image = render(raw_session, prompt=prompt, steps=steps, seed=seed,
               strength=nominal_strength, gate_plan=plan)
receipt = image.info["strength_schedule"]
```

The [shared calibration workflow](https://github.com/HyperGAN/particle-sliders/blob/b1f780decc516808cfd3019d5e5b674457161f86/docs/strength-schedules.md#activation-dependent-steering)
provides `contrastive_sae_direction` and active `calibrate_gate_budget`. Fit
against separate positive/neutral trajectory captures at this exact Flux
readout. The saved velocity cache pairs two velocities on the same neutral
latent, so dedicated positive latent captures and a fitted Flux direction
are required. Use a disjoint set of matched prompts and initial noise for
active threshold or bias calibration. Record the fitted probe, exact model
and saved RAW identities, precision, geometry, solver grid and calibration
cases in the record whose SHA-256 goes into the plan.

For CLABS, pass `method="clabs"`, the calibrated bias as `parameter`, a local
gate `seed`, and an explicit dictionary with `mean`, `std`, `epsilon`,
`setpoint`, `proportional_gain`, `integral_gain` and `beta`. For the matched
random control use `method="random"`, target firing rate as `parameter`, and
a local `seed`. Each request starts a fresh gate. Read the projections,
probabilities, decisions and effective strengths per interval and the
realized firing rate and sigma-weighted strength from the receipt. Save it
alongside the image. For fixed replay divide the logged strengths by the
nominal strength and pass them as `step_gains` in a separate request.

Build the [pinned core wheel](https://github.com/HyperGAN/particle-sliders/pull/269)
from commit `b1f780decc516808cfd3019d5e5b674457161f86` and supply it as a file.
The product authenticates the steering module's SHA-256 before loading it
into a private namespace. Its installed core and strict RAW/FULL initialization
receipts retain the existing dependency pin. The older Ink Wash RAW18 panel
belongs to its archived runtime. Keep the eight recorded case/seed pairs out
of fitting and calibration and use their frozen finished Ink Wash effect and
separate integrity checks when evaluating that archive in its source-bound
runtime. Native adaptive output gain is unmeasured.

For LR work, rank only the equal-case intended-effect ON−OFF mean in finished
outputs. Preserve ties/zeros/negatives and reuse existing completed measurements.
Teacher SSE and GAN diagnostics are not selectors. Integrate the shared
qualified catastrophic-failure guard at an actual finished-output checkpoint
boundary before more candidate updates/resume/promotion; weak effect is not
failure, missing/error is unmeasured. This Source does not invent a universal
renderer or claim automatic qualified all-product deployment.

## Native T5 tokenizer dependencies

The native package explicitly requires SentencePiece 0.2.3 and protobuf 7.36.2,
which the staged Schnell T5 tokenizer needs for its configured slow-to-fast
conversion. A fresh isolated environment passed the actual local tokenizer
loader with these two additions and all61 previously qualified versions
retained. See [the scoped validation](docs/TOKENIZER-VALIDATION.md). This check
does not establish pretrained neural inference, training, GPU fit or LR quality.
