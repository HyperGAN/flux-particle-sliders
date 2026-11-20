# Current optional native Flux package

The separately installable [native-candidate](native-candidate/README.md) adds
native packed Flux training with ParticleGAN 0.11.0, exact FULL restore, RAW-only
inference and a native finished-image renderer. The earlier random-tiny CPU
checks are recorded separately; [the evidence](native-candidate/docs/CPU17-VALIDATION.md) states
the exact Source, dependencies and limits. Separately retained actual pretrained
RAW32 execution now has a [completed provisional zero tie](docs/finished-output-learning-rates.md)
for G0.012/G0.001 on one saturated case. Both FULL32 catastrophic guards pass;
`trait_qualified:false`, differing critic execution histories and no default,
global optimum, released weights or ordinary Generate remain explicit.

The root README and original package document the preserved phase1 prerequisite
at11666361145d175e04ad571ae07100de84a98904. Earlier locks and evidence retain their original source identity. Its core0.12/PG0.9.4 blocked-training boundary remains true for
that old package. The new package lives entirely in native-candidate/, pins
core 0.15.1 at `c13890f8e2a1ecf1dc8eb9f2a37f9187cb9fea78` and ParticleGAN 0.11.0, and needs a
separate compatible environment. Old artifacts are not migrated or retagged.
