# Current optional native Flux package

The separately installable [native-candidate](native-candidate/README.md) adds
native packed Flux training with ParticleGAN 0.11.1, exact FULL restore, RAW-only
inference and a native finished-image renderer. The earlier random-tiny CPU
checks are recorded separately; [the evidence](native-candidate/docs/CPU17-VALIDATION.md) states
the exact Source, dependencies and limits. Separately retained actual pretrained
RAW32 execution now has a [completed provisional zero tie](docs/finished-output-learning-rates.md)
for G0.012/G0.001 on one saturated case. Both FULL32 catastrophic guards pass;
`trait_qualified:false`, differing critic execution histories and no default,
global optimum, released weights or ordinary Generate remain explicit.

The [original phase-one preview](https://github.com/HyperGAN/flux-particle-sliders/tree/11666361145d175e04ad571ae07100de84a98904) retains its blocked-training boundary. Use a fresh environment for the current native candidate and keep saved artifacts with their captured source.
