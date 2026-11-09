# Flux particle sliders — phase 1 prerequisite extraction

This is the separate HyperGAN destination for the public modern Flux prerequisite
preview from `HyperGAN/particle-sliders` at
`f150b28cb25ad55a99b37b06fa0ddc989f3428d9`. The active 0.2.0 / v2 cohort applies the coordinated core migration to exact
`a794e3b4a8813b012f2c4faa3da82b78d4a90214`. Native response code and the LoRA
graph retain original bytes; only dependency/profile identity and recursive
source/data admission change. Prior 0.1.0 / v1 remains at local commit
`735977b704c82fe8b48da36c6b432497b0725f6d` and is never relabeled. See
[EXTRACTION-PROVENANCE.json](EXTRACTION-PROVENANCE.json) for original paths,
Git blobs and hashes, and [MIGRATION.md](MIGRATION.md) for the extraction boundary.

**Shared training remains blocked. This is not a migrated trainer or release.**

See [PUBLICATION.md](PUBLICATION.md) for the preserved source lineage, current
focused CPU checks and lack of hosted CI or pretrained/GPU/media qualification.
Native Flux responses stay `[1,image_tokens,64]`; core 0.12 at
`a794e3b4a8813b012f2c4faa3da82b78d4a90214` cannot admit that geometry unchanged.
No training/inference backend or CLI is registered, and no optimizer, resume,
EMA, shared export, media renderer or pretrained/GPU qualification is supplied.
The historical prompt_pair.loss trainer and its helpers remain intact upstream
and are tracked for a later migration rather than copied here.

The opt-in distribution supplies real HyperGAN develop
`5f368a3770cf5bdae0448995c9f76e29af8049f7` / HNDL 0.10 ordinary rank 4/alpha 1
LoRA graphs and the native Diffusers 0.38 packed forward adapter. It checks the
complete recursive 50-file core payload named in the source lock, including
LICENSE and all source/data suffixes, plus the verified builder closure, local
graph and six native Diffusers files. Only bytecode/cache files are omitted
beneath the package root; distribution metadata outside that root is separate.
[QUALIFICATION.md](QUALIFICATION.md) documents the preserved contracts.

Install and qualify from this standalone repository, without an umbrella checkout:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install 'torch==2.14.1+cpu' --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install '.[test]'
.venv/bin/python scripts/stage_historical_reference.py --output /tmp/flux-lora-reference.py
FLUX_REFERENCE_LORA_PATH=/tmp/flux-lora-reference.py CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m pytest -c pyproject.toml tests -q
```

The staging script fetches exactly one immutable public **test-only** reference
file and verifies its SHA256 and Git blob. The historical code is not exported,
installed or imported by the product. Tests execute its verified bytes only for
matched historical LoRA comparisons; there is no fallback to an umbrella path
and pytest performs no network access. The shared library dependency remains an
exact public VCS package dependency, not a copied core implementation.

`FluxLoRA` builds unattached graphs on a staged frozen float32 CPU native model.
Both constructors restore CPU Torch RNG; a private bank stream keeps sequential
branch initializations distinct. `borrowed()` restores hooks/modes, and scaled
calls retain OFF and raw LoRA semantics. Raw nn.Module state roundtrip is a
low-level weight check, not a product checkpoint/export API.

`PackedFluxResponse` forwards native pooled/text embeddings, text/image IDs,
guidance and raw time normalization directly. Its raw packed velocity and shared
frozen-positive teacher convention are distinct from historical scheduler-step
latents. `preflight_training(..., model=staged_model)` probes the existing shared
continuous_field rank gate with the unchanged rank 3 shape and refuses without
native mutation. Wrong PG source/version failures have a distinct chained error.

Synthetic legacy safetensors-container refusal is checked through the shared
loader before backend resolution. Actual historical checkpoint migration,
EMA/resume/export, native training, mixed precision, checkpointed recomputation,
rendering and quality remain unqualified. Root owns Claude RETURN review and
landing decisions; phase 1 introduces no GPU jobs, workflows or package release.

See [CORE-MIGRATION.md](CORE-MIGRATION.md) for exact cohort/packet boundaries,
source identity, reproducible installation and remaining qualification gates.

GPU preparation remains blocked. The separate [readiness plan](docs/GPU-READINESS.md)
and `scripts/preflight_flux_readiness.py` provide bounded cache/source inspection
that always refuses execution. They do not load model weights or add a GPU,
training, EMA, resume, or export path.
