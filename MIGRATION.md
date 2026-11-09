# Phase 1 scope and later historical migration

Approved destination: `HyperGAN/flux-particle-sliders`.
Approved source: `HyperGAN/particle-sliders` commit
`f150b28cb25ad55a99b37b06fa0ddc989f3428d9`, modern product root
`packages/particle-sliders-flux-packed/`.

The default branch is a minimal README/license scaffold. Actual source extraction
is proposed on a separate review branch and draft PR. Only allowlisted modern
source/tests/docs/configuration and the original MIT license are exported; no
upstream Git history is mirrored. Original embedded and historical paths remain
unchanged until replacement verification and root decisions.

The original phase 1 commit 735977b preserved all five runtime files and pyproject.
The active 0.2.0 / v2 follow-up changes the exact shared-core dependency, cohort
identity and recursive source/data lock only. The namespace, native response
implementation, LoRA graph, objectives, architecture and gates stay unchanged.
The prior 0.1.0 / v1 profile remains recorded in that original commit. No shared implementation or
historical trainer is copied. Documentation and sdist inclusion are relocated;
the historical parity helper reads a separately staged, pinned test-only file.
Source→destination Git blobs and both SHA256 values are recorded in
`EXTRACTION-PROVENANCE.json`.

The previous `qualification/` directory contains run receipts with environment
paths; it is intentionally excluded. QUALIFICATION.md is replaced with portable
contract/test documentation. No private logs, recordings, weights, run outputs,
environments, workflow files or unrelated evidence are exported. New installed
receipts are returned separately to root and are not package artifacts.

Historical model code is a later tracked migration, not an installed dependency:

- `conceptmod/textsliders/train_lora_flux.py`
- `conceptmod/textsliders/lora.py` and `dora.py`
- `conceptmod/textsliders/train_util.py` and `model_util.py`
- `conceptmod/textsliders/prompt_util.py`, `config_util.py` and `debug_util.py`
- `conceptmod/textsliders/data/prompts-flux.yaml`

The sole external test reference is `conceptmod/textsliders/lora.py`, pinned in
`tests/historical-reference.json`. Fetching one verified public file for parity
is not an umbrella checkout or historical trainer migration. No model loader or
model weights are fetched. This exception is test preparation only; runtime
imports and package data are fully independent of those historical paths.

External runtime dependencies are declared in pyproject.toml. The active shared library
is the exact core a794e3b package, with PG 0.9.4, HyperGAN develop 5f368a and HNDL 0.10.
The package's local `.hndl` graph and `sources.json` remain adjacent to the Python
binding and are included in wheel/sdist. Core locks inspect the installed shared
package; native locks inspect installed Diffusers, so repository relocation does
not alter lookup roots or require an unrelated native repin.

OFF/raw native outputs and low-level raw weights are testable prerequisites.
EMA, shared resume/export and native-only inference remain absent, because no
shared backend is registered and native packed geometry is still unadmitted.
Legacy container refusal uses a synthetic fixture and does not qualify an actual
historical model checkpoint. No migration, domain bypass or admission is implied.

The coordinated shared-core update is documented separately in CORE-MIGRATION.md.
No umbrella source/PR or old preview artifact is modified by this follow-up.
