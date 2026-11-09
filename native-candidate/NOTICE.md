The ordinary LoRA adapter, two-line HNDL graph and native response validation
derive from HyperGAN/flux-particle-sliders commit
11666361145d175e04ad571ae07100de84a98904 (src/ps_flux_packed_preview/native.py,
lora.hndl). Its MIT license is retained at ../LICENSE; original repository
copyright is Rohit Gandikota (2023).

General external builder source/provenance checks derive from
HyperGAN/particle-sliders' MIT general verified_builder statements. Only the
HNDL/HyperGAN builder closure is carried; no removed product projection, critic,
weights, old ParticleGAN profile or old core-package payload is bundled.

Native finished scheduling/packing/unpacking/VAE execution is delegated to the
public Diffusers0.38.0 FluxPipeline (Apache2.0). Native Source pins record its
operative leaves; this package does not reproduce its numerical algorithm.

The new packed critic, lifecycle/cache/RAW adapter and initialization policy
belong to this explicitly new native product cohort. Public ParticleGAN losses,
optimizer formula, scalar statistics and penalty mathematics remain upstream.
