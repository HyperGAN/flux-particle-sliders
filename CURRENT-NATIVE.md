# Current optional native Flux package

The separately installable [native-candidate](native-candidate/README.md) adds
full native packed Flux ParticleGAN0.9.5 training, exact FULL restore, RAW-only
inference and a native finished-image renderer. Its17 actual random-tiny CPU
checks passed; [the evidence](native-candidate/docs/CPU17-VALIDATION.md) states
the exact Source, dependencies and limits. No pretrained/GPU/LR or image-quality
qualification, trained weights, version bump or package release is claimed.

The root README and original package document the preserved phase1 prerequisite
at11666361145d175e04ad571ae07100de84a98904. Every existing tracked body is retained
byte-for-byte. Its core0.12/PG0.9.4 blocked-training boundary remains true for
that old package. The new package lives entirely in native-candidate/, pins
core0.14.1 at41f6066f2d2644902670d5eb736de052e1a70553 and PG0.9.5, and needs a
separate compatible environment. Old artifacts are not migrated or retagged.
