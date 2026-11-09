# Actual native CPU validation

On2026-11-09, source commit `01ac3bb2166656427252776b591ba47a82ba35d7`
passed all17 random-tiny native CPU checks in1.546 seconds:0 failures,0 errors,
0 skips. The original8 checks plus9 HNDL initialization regressions ran once
against the repaired Source. [cpu17-evidence.json](cpu17-evidence.json) records
the measured file hashes, dependency identities, scope and sanitized evidence
identities. The original failed8-check result (5pass/3fail) remains retained.
Test-module authorship notes remain byte-identical to the measured Source;
this receipt records their later actual execution.

The tests exercise actual tiny Diffusers Flux/VAE and Transformers CLIP/T5
components, the exact HNDL builder and official ParticleGAN0.9.5. They cover
native packed responses, fresh-before-optimizer initialization, exact FULL
resume, RAW export/inference, cache reuse/tamper refusal, real tiny pipeline PIL
rendering and cleanup. Actual HNDL Linear subclasses are admitted through plain
parameter-only holders with complete explicit declarations. Original Parameter
identity, host forward, Kaiming(a=1) down distribution and KEEP zero-up are
preserved; the shared initializer is unchanged.

The execution reused the existing repaired private environment and explicitly
loaded the new worktree's `src` before the installed product. Every imported
Flux module resolved to that new Source. The installed155a package remained
untouched; this result does not claim it was replaced. No pip/install/resolution
was repeated. Root observed a genuine outer0/reap62be3e and worker0, released
only the owned8GiB CPU allocation, and retained the original failure metadata.
CUDA was never initialized, no pretrained model was consumed, and blocked
model/network attempts were0.

The original top-level prerequisite distribution and every file from public
baseline116663 remain unchanged. This is a separately installable native
candidate, with no version/tag or weight release. Pretrained model preparation,
actual GPU fit/execution, finished pretrained outputs and intended-effect LR
selection remain separate work. No trained pretrained slider, published weight,
LR winner or image-quality success is claimed by these CPU tests.

For a fresh compatible environment, install `native-candidate[test]` from this
checkout (for example `python -m pip install './native-candidate[test]'`) and
run `python -m pytest -c native-candidate/pyproject.toml native-candidate/tests`.
The dependency pins are in that subdirectory. Keep the old core0.12 preview in
its own environment. Running these tests is an explicit CPU action; it is not
part of pretrained inference or permission to rerun completed scientific work.
