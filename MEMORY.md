# Memory requirements

| Operation / profile | GPU count | Calculated VRAM per GPU (GiB) | Calculated host RAM (GiB) | Input limits / precision |
| --- | ---: | --- | ---: | --- |
| Native cached-response training | 1 | 80 | 40 | B2; 512²; text256; BF16; no checkpointing |
| Native cached-response training, largest declared canvas | 1 | 160 | 40 | B2; 1024²; text256; BF16; no checkpointing |
| Finished-image sampling | 1 | 48 | 48 | B1; up to 1024²; text256; BF16; whole pipeline resident |
| Conditioning / teacher preparation | 1 | 32 | 48 | 512²; one native teacher call at a time; encoders on CPU |


These are **calculated planning budgets, not measured peaks or hard runtime ceilings**.
They apply only to the bounded profiles in the table, for one process/slider at a
time. GiB means 2^30 bytes. Add concurrent jobs' resident allocations; do not sum
mutually exclusive loading, conditioning, training and decoding phases as if all
their models were resident together. Allow the stated host RAM independently of
VRAM. Larger batches, longer inputs, extra references, FP32 bases, different
attention kernels or allocator fragmentation can require more. There is no finite
universal maximum for arbitrary inputs. These calculations start no model jobs.

## Native profile and calculation

The native candidate uses `black-forest-labs/FLUX.1-schnell@741f7c3ce8b383c54771c7003378a50191e9efe9`. Its
[contracts](native-candidate/src/ps_flux_native95/contracts.py),
[loader](native-candidate/src/ps_flux_native95/native.py),
[attachment selector](native-candidate/src/ps_flux_native95/adapter.py),
[cache](native-candidate/src/ps_flux_native95/cache.py) and
[critic](native-candidate/src/ps_flux_native95/critic.py) define these profiles.
The older bootstrap is not a smaller native training configuration.

* Frozen DiT: 23,782,357,120 tensor bytes = **22.149 GiB**. T5 is
  8.870 GiB; CLIP and VAE serialized upper bounds add
  0.385 GiB. The renderer moves **all four components** to its
  selected GPU, giving 31.405 GiB of frozen residency.
* Ordinary rank4 LoRA has 4,669,440 FP32 scalars: 19 double blocks ×4
  image Q/K/V/output hosts plus 38 single blocks ×3 Q/K/V hosts, each
  `4*(3072+3072)`. Text `add_*` hosts and single-block output outside attention
  are excluded by the literal selector. No particle bank is allocated.
* The critic concatenates **all three full contexts**, rather than a mean text
  vector: `C=4+3*(256*4096+768+256*3+image_tokens*3)`. At 512²,
  C=3,159,556, so its condition→32 projection
  dominates the 101,107,201-scalar critic. Adapter +
  critic/update allowance is 7.881 GiB.
* No checkpointing is supported. With B2, T=image_tokens+256, the retained
  envelope is `2*B*T*(2*19+38)*(12*3072+3*12288+4*3072)` bytes.
  This deliberately bounds the two double-block streams separately: **31.172 GiB
  at 512²; 105.984 GiB at 1024²**. Add frozen residency and the trainable allowance,
  then 8/12 GiB for penalty/attention/allocator workspace; round upward to 80/160.
* Sampling retains one block's working set, not every backward activation.
  Frozen 31.405 +7 live/decode +7 workspace rounds to48.
  Conditioning keeps T5/CLIP/VAE on CPU and only the DiT on GPU.
* Host loading: complete pipeline 31.405 GiB + largest
  transformer shard 9.278
  + up to 128 frozen context/response rows (under2 GiB) +4 runtime GiB is rounded
  to48. Cached-training DiT loading + one largest shard +2 cache +4 runtime is
  rounded to40. More cache rows increase disk use and per-row loading costs.

These are BF16 profiles. Full FP32 doubles the frozen and most native activation
entries; it is not covered by the table. Native geometry validation permits
64..1024 pixels aligned to16; batch/cache/critic widths can grow, so there is no
single finite maximum across every accepted option. The table binds critic width16
and up to128 cache rows. Quantized base loading is not implemented by this candidate.
Pinned config/index/size metadata is captured in [docs/memory](docs/memory/inputs.json).


## Optimizer and estimate policy

Training figures freeze pretrained weights, use RAW current adapters and explicitly
disable weight EMA. They include no full-base gradients or EMA shadow model.
The PG 0.9.5 `bispf_optimized` recipe uses zero-momentum DualNorm, not Adam's two
full-size moment tensors. Its direction blend still retains protected gradients,
clones originals/displacements, and converts up to two protected normals to
FP64. For a conservative planning allowance, the tables reserve **80 bytes per
trainable scalar in total**: 8 for FP32 weights+gradients, 16 for two FP64 normals,
and 56 for protected FP32 gradients, FP32/FP64 direction/projection/polar,
rollback and serialization temporaries. This is an explicit allowance, not a claim
that every temporary is simultaneously allocated or that it is an exact peak.
Small scalar optimizer statistics remain enabled. Recipe changes need a new ledger.
See released [PG 0.9.5 recipes](https://github.com/255BITS/ParticleGAN/blob/v0.9.5/particlegan/recipes.py),
[DualNorm](https://github.com/255BITS/ParticleGAN/blob/v0.9.5/particlegan/optim/dualnorm.py)
and [direction blend](https://github.com/255BITS/ParticleGAN/blob/v0.9.5/particlegan/optim/direction_blend.py).

Activation entries below are architecture-based envelopes. Fused/memory-efficient
SDPA is assumed for the pretrained model; a math attention fallback that materializes
all score matrices can exceed them. No model benchmark or runtime measurement was
performed for this documentation. Machine-readable inputs, arithmetic and all table
rows are in [docs/memory/inputs.json](docs/memory/inputs.json).

## Explicit rounding ledger

| Operation | Calculated GPU subtotal (GiB) → planning budget | Host subtotal (GiB) → planning budget |
| --- | --- | --- |
| Native cached-response training | 69.202 → 80 | 37.428 → 40 |
| Native cached-response training, largest declared canvas | 148.080 → 160 | 37.428 → 40 |
| Finished-image sampling | 45.405 → 48 | 46.684 → 48 |
| Conditioning / teacher preparation | 30.149 → 32 | 46.684 → 48 |

The detailed host terms and per-device arithmetic are included in the machine-readable inputs.
