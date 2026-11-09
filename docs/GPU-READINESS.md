# Flux GPU readiness: blocked, with metadata-only preflight

This preparation is isolated on `readiness/flux-packed-admission`, based on
reviewed migration head `8a80ace99ad5afa3f2b3d779b14cf55ed557ceb1` and exact core
`a794e3b4a8813b012f2c4faa3da82b78d4a90214`. Existing runtime code, graph, source
lock, cohort, version, and historical environments are unchanged. This feature
does not add a training path or relax the CPU/native geometry gates. Full source
and test review through Claude and RETURN is required before any GPU execution
or landing. Root owns scheduling; no card is selected or reserved here.

## First blocking contracts

The shared `validate_pg09_domain(..., response_domain="continuous_field")` and
`ResponseCapabilities` accept genuine fields of rank 4 or 5, while native Flux
predictions are packed rank 3: `[1, image_tokens, 64]`. The actual reviewed
preflight refuses that unchanged shape. `FluxLoRA` also explicitly rejects CUDA,
non-float32 base parameters, and gradient checkpointing. No model loader,
immutable pretrained model binding, frozen response dataset/provider, packed
critic, shared training backend, checkpoint/resume, EMA, or export API exists.
The core owner's synthetic CUDA fixture does not qualify any of those native
Flux contracts. There is no runnable native training command.

The metadata-only command below always emits `ready: false` and exits with
status 2, even if every candidate cache file is present. It makes no network
request, constructs no model, reads no weight payload, calls no CUDA API, and
attaches no hook. Its installed admission probe checks the exact core origin
and invokes the existing refusal before model loading. The supplied token
counts describe an intended case; they are not packing or positional proof.

```sh
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python scripts/preflight_flux_readiness.py --snapshot /approved/cache/snapshots/IMMUTABLE_REVISION --model-id black-forest-labs/FLUX.1-schnell --revision IMMUTABLE_REVISION --image-tokens 1024 --text-tokens 256
```

Replace the placeholder path and revision with an explicitly selected candidate.
Do not interpret this safe readiness command as permission to load that model.
Metadata reads are capped at one MiB per JSON file; shard names must remain local
basenames. Weight files are inspected only through filenames and stat sizes.
Cache file presence is not cryptographic weight integrity or loader qualification.

## Scoped cache and data evidence

Only directly relevant existing cache/project metadata was inspected. The default
HF cache had no named Flux model entries. A separate existing cache contained:

- Schnell revision `741f7c3ce8b383c54771c7003378a50191e9efe9`: pipeline/config,
  both text encoders, tokenizer assets, VAE, and scheduler metadata were present.
  Its transformer directory contained configuration only, with no transformer
  safetensors payload. The cache therefore cannot supply a complete native model.
- Dev `refs/main` named `3de623fc3c33e44ffbe2bad470d0f45bccf2eb21`, but its
  snapshot directory was absent. A ref alone does not establish cached weights.

Schnell metadata SHA256 values:

- `model_index.json`: `24946df21ff25e210486b5f6b14208983a90c9c73f8d48cfa724c0e4e03f7201`.
- `transformer/config.json`: `397cfb92299488013ec3af6142a2a877366f8d2e44efbb4f3e33479e7960d3d0`.

These are candidate cache revisions, not a product-approved model binding. The
historical loader combines a Schnell pipeline with a caller-supplied single-file
transformer and makes immediate CUDA moves. It was inspected as source only and
must not be used as this product's qualification command. No single-file path,
hash, compatible flavor, or modern loader has been bound by the current product.

The historical `prompts-flux.yaml` at upstream source
`f150b28cb25ad55a99b37b06fa0ddc989f3428d9` has SHA256
`5112b4f0011b6a6b5026d1bfbbe7671c117c357bbc1e365f303c828a8c52664c`.
It is an old enhance/green/red prompt configuration with batch size 2 and
resolution 512, not a prepared modern response dataset. No frozen embeddings,
latents, positional IDs, teacher outputs, split identities, or row-cache manifest
are supplied here. It is not silently adopted as a new training objective.

`row_evidence.snapshot_row` is an opt-in CPU preparation aid for one already
staged `ResponseRow`. It clones and fingerprints all native tensor coordinates
and includes an explicitly caller-asserted model SHA256. It does not verify
model weights, install a loader, provide a durable immutable cache, admit a
shared game, or define a new training objective. Snapshot tensors remain mutable;
the fingerprint represents their contents at creation only. The utility rejects
unsupported shapes, gradients, devices, guidance and schedules before producing
evidence. A qualified loader and an authenticated row provider remain necessary.

## Native API and precision proof prerequisites

The locked implementation is Diffusers 0.38.0 `FluxTransformer2DModel`, with
Transformers 4.57.6. The inspected Schnell configuration advertises an older
Diffusers 0.30 development serialization, 19 double blocks, 38 single blocks,
24 heads of width 128, input width 64, text feature width 4096, pooled width 768,
and no guidance embedding. Actual loaded config/weights must be independently
bound and verified; cached config compatibility is not full-model qualification.

The current staged response requires frozen, colocated float32 CPU tensors:
latent/output `[1, image_tokens, 64]`, text `[1, text_tokens, joint_attention_dim]`,
pooled `[1, pooled_projection_dim]`, text IDs `[text_tokens, 3]`, image IDs
`[image_tokens, 3]`, and one raw timestep normalized by 1000. Guidance must be
absent for Schnell-style non-guidance models and match the declared model config
for guidance models. HNDL branches retain fixed native image/combined token
counts. No teacher text encoder is trainable. The response is raw packed velocity,
not the historical scheduler-step latent objective.

Before enabling a shared native game, root/core review must establish:

1. An explicit packed continuous geometry/capability and penalty contract, with
   a new immutable reviewed source/cohort if needed. Existing vector guards and
   historical cohorts must remain intact; an unknown domain label is not proof.
2. Exact model/revision, complete local components and verified weight identity,
   product loader, data/provider identity, teacher convention, and matched native
   forward/VJP behavior for the actual precision and target layout.
3. A declared packed-response critic/condition graph and shared game/lifecycle
   binding, with penalty second derivatives and generator ownership tests.
4. If proposing a real unpacked-field response instead, native pack/unpack
   invertibility, coordinate/channel/patch ordering, positional ID correspondence,
   matched outputs and VJPs, and equivalence at the penalty boundary. Flattening,
   pooling, or merely relabeling axes to obtain admission is not acceptable.
5. A reviewed native GPU/precision path. The present CPU-only guard cannot be
   removed merely because a GPU becomes available.

## Conditional resource estimates and execution cap

No GPU memory measurement was made. Four cached non-transformer weight files
totaled 9,938,459,838 bytes by stat metadata; that is storage size, not resident
memory or a verified loaded dtype. Transformer storage was missing.

Conditional adapter estimate from the inspected configuration and unchanged
target policy: `(19 * 4 + 38 * 3) * (3072 * 4 + 4 * 3072)` gives 4,669,440
parameters, about 17.81 MiB of FP32 adapter weights, or 71.25 MiB for weights,
gradients, and two Adam moments together. Actual loaded target shapes still need
verification. This excludes the frozen transformer, text encoders, VAE, critic,
activations, attention workspaces, allocator/context costs, and other overhead.
For a verified base parameter count P, FP32 parameter storage alone is 4P bytes;
P was not established from the inspected metadata. No total working-set claim is
made, and no card capacity is inferred.

Current execution cap is zero model loads, zero GPU allocations, and zero updates.
After all blockers and source reviews are resolved, the proposed first native
qualification is one fixed-seed case, batch size 1, at the smallest accepted
native geometry, with at most one attempted update and a ten-minute wall-time
cap from first allocation. Root must first set an explicit allocator ceiling
from the selected card's live budget; an unset ceiling is a stop condition.
No automatic OOM retry, offload, quantization, alternate model, or shape/dtype
fallback is part of this plan.

## Ordered first qualification, once separately admitted

1. Read-only source/model/cache/data preflight; stop on any missing component,
   source/revision/hash mismatch, unsupported dtype/shape, or unresolved binding.
2. OFF identity against the exact frozen native forward. Preserve positional and
   text/image token contracts; check raw adapter output and input/parameter VJPs.
3. Only after shared admission, attempt one shared update and verify ownership,
   finite output/gradients, and the declared critic's second derivatives.
4. Test raw/EMA evaluation and exact interrupted checkpoint/resume only when
   those product APIs actually exist. Their current absence is a blocker, not a
   skipped success. Do not relabel old packets or invent a local checkpoint API.
5. Compare modes, flags, existing gradients, hooks, RNG, and borrowed state before
   and after success and injected failure. Exercise cleanup retry and stop if
   restoration is incomplete, a graph is detached, results are nonfinite, or any
   cap is reached. Root owns GPU scheduling and any subsequent execution decision.

This branch adds only the fail-closed metadata tool, tiny CPU tests, and this
plan. Prior CPU fixture receipts remain software evidence only. The historical
source-publication hold was superseded for the public source snapshot only;
Claude/RETURN review, execution admission and later landing decisions remain separate.
