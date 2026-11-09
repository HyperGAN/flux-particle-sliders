# Native packed Flux PG0.9.5 candidate

This is a new, separately installable Source candidate. It preserves native
`[1,image_tokens,64]` velocity responses and the ordinary rank4/alpha1 HNDL
down/up LoRA from the published Flux prerequisite. It supplies a genuine shared
`ResponseBackend`, full native teacher cache, official PG0.9.5 `adapter_gan`
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
A complete authenticated local pretrained model, actual prompt/cache preparation
and admitted pretrained/GPU pilot remain pending. No pretrained slider weights,
finished pretrained examples or optimized LR are established. The core dependency names the actual merged main PR205 commit
`41f6066f2d2644902670d5eb736de052e1a70553` (core0.14.1/PG0.9.5). The old top-level
preview package and its core0.12/PG0.9.4 identity are unchanged. Use a new
environment, not the old captured environment; do not coinstall incompatible
core/HNDL profiles or reinterpret old packets.

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
`flux-native95 check` for an allocation-free profile description. Framework
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
flux-native95 cache --model-dir MODEL --model-lock-sha256 LOCK_SHA \
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
from ps_flux_native95.lifecycle import open_training
from ps_flux_native95.contracts import INITIALIZATION_POLICY

config = SessionConfig("flux_native_packed95", "my-new-flux-cohort",
    model_dir="MODEL", model_lock_sha256="LOCK_SHA",
    cache_dir="CACHE", cache_index_sha256="CACHE_SHA",
    device="cuda:0", base_precision="bfloat16",
    height=512, width=512, text_tokens=256,
    steps=32, batch=2, seed=7, ema_decay=None,
    pg_version="0.9.5", pg_source_profile="release-0.9.5",
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
omitted or null. No LR winner is selected by this package. Weight EMA is refused.
Fresh initialization uses the current shared scoped helper exactly once before
optimizers: LoRA down keeps its true Kaiming(a=1) RMS law under named QR seeds,
up remains zero/KEEP, and critic leaves retain their declared host RMS. Frozen
pretrained leaves are outside ownership. FULL restore and RAW inference retain
the original receipt and never call that helper; temporary constructor values
are replaced by exact saved state, never used to reset a loaded checkpoint.

Use `open_training(config, resume="FULL.pt")` for exact continuation of the
original fixed budget. The CLI counterpart is `flux-native95 train --config
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
flux-native95 render-pair --artifact RAW.pt --model-dir MODEL \
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
