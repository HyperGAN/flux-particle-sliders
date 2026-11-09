# Coordinated modern-core migration: v2/a794

Active preview distribution: `particle-sliders-flux-packed-preview==0.2.0`.
Active cohort: `flux-packed-velocity-hndl010-prerequisite-v2`.
Exact shared-core dependency:
`particle-sliders-core[hypergan-hndl] @ git+https://github.com/HyperGAN/particle-sliders.git@a794e3b4a8813b012f2c4faa3da82b78d4a90214#subdirectory=packages/particle-sliders-core`.
Core package version remains 0.12.0, PG 0.9.4, HyperGAN develop 5f368a and HNDL 0.10.

Previous preview 0.1.0/cohort v1 and core 7a8a321 remain at local extraction commit
`735977b704c82fe8b48da36c6b432497b0725f6d` and original upstream f150b28.
Their metadata, wheels and packets are not overwritten or relabeled. Explicit
v1 preflight requests and old core/profile manifests fail closed before hooks.
No old/new packet compatibility is supported. Raw nn.Module tensor dictionaries
are unversioned low-level weights, not product checkpoints or migration evidence.

The recursive source/data lock is regenerated once from an exact a794 VCS-built
wheel. All50 files beneath installed `particle_sliders/` are compared by relative
path and SHA256, including LICENSE and any nested/unknown data suffix. Only cache
paths/bytecode are excluded. Core distribution metadata is outside this payload
scope; installed VCS origin/commit/version are independently asserted by the
installed conformance test and captured in the private review receipt. The wheel
hash in sources.json is the qualification build receipt, not an assertion that
independent build zip timestamps reproduce an identical wheel hash. Runtime
admission depends on exact declared commit/profile/version and complete payload
bytes. Native Diffusers/source and LoRA graph hashes remain unchanged.

README.md gives isolated CPU install commands. Build with `python -m build`, then
install the wheel or sdist (with dependencies) in a fresh CPU venv. The exact core
VCS dependency must resolve to a794; run `python -m pip check` and the full test
suite from outside this checkout with empty PYTHONPATH and disabled user site.
Stage only the same immutable test reference with scripts/stage_historical_reference.py;
pytest remains offline. No umbrella model import or checkout is needed. Complete
installed receipts and artifact/file indices remain outside this source payload.
The historical creation/publication hold was superseded for this public source
snapshot only; model execution and subsequent landing decisions remain separate.

The bank architecture, native.py, LoRA graph, teacher/objective, token axes and
native admission gates are unchanged. The shared synthetic CUDA fixture does
not qualify native Flux, widths, media quality or a product training path.
There is still no product training/inference/CLI registration, shared optimizer,
EMA, resume or export API. OFF/raw native conformance and unsupported old-container
refusal remain testable; shared-native behavior remains blocked/unexecuted.

Only this active extracted product is changed. Historical/default/stable cohorts,
umbrella PR157, licenses, workflows, access settings and deployment/release state
are untouched. Root owns Claude RETURN/final-core/Ada and all landing decisions.
