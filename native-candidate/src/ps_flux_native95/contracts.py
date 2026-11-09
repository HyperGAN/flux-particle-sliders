"""Allocation-free options and local model-lock checks for a new Flux cohort."""
import hashlib
import json
import math
from pathlib import Path

# Actual main PR205 packed-admission Source; package/core version remains0.14.1.
CORE_REVISION = "41f6066f2d2644902670d5eb736de052e1a70553"
MODEL = "black-forest-labs/FLUX.1-schnell"
MODEL_REVISION = "741f7c3ce8b383c54771c7003378a50191e9efe9"
BACKEND_NAME = "flux_native_packed95"
SOURCE_PROFILE = "flux-native-packed64-ordinary-hndl010-pg095-v1"
PROTOCOL = "flux-native-packed64-positive-velocity-v1"
INITIALIZATION_POLICY = "flux-hndl-kaiming-a1-QR-zero-up-critic-host-RMS-v1"
WEIGHT_POLICY = "raw-current-no-weight-ema"
EXPECTED = dict(in_channels=64, num_layers=19, num_single_layers=38,
                attention_head_dim=128, num_attention_heads=24,
                joint_attention_dim=4096, pooled_projection_dim=768,
                guidance_embeds=False)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def record(path):
    path = Path(path)
    return {"path": str(path.absolute()), "bytes": path.stat().st_size, "sha256": file_hash(path)}


def positive(value, key):
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError(key + " must be finite, positive and non-bool")
    return value


def rate_overrides(options):
    """Ordinary LoRA has G/D only; there is no bank or bank LR to invent."""
    if options.get("prior_lr_mult") is not None:
        raise ValueError("ordinary Flux adapter_gan has no particle prior/bank LR")
    return {key: positive(options[key], key) for key in ("lr", "d_lr_mult") if key in options}


def validate_policy(options):
    if options.get("ema_decay") is not None or options.get("ema") is not None:
        raise ValueError("RAW-only Flux requires explicit ema_decay=None; weight EMA is refused")
    if (options.get("pg_version"), options.get("pg_source_profile"),
            options.get("penalty_input_domain"), options.get("initialization_policy")) != (
            "0.9.5", "release-0.9.5", "auto", INITIALIZATION_POLICY):
        raise ValueError("explicit native packed PG0.9.5/auto/fresh-init source cohort required")
    rate_overrides(options)


def geometry(options):
    height, width, length = (options.get(key, default) for key, default in
                            (("height", 512), ("width", 512), ("text_tokens", 256)))
    if any(type(v) is not int for v in (height, width, length)):
        raise ValueError("integer native dimensions required")
    if not (64 <= height <= 1024 and 64 <= width <= 1024 and height % 16 == width % 16 == 0):
        raise ValueError("native dimensions must be 64..1024 and divisible by16")
    if not 1 <= length <= 256:
        raise ValueError("this Schnell cohort declares a fixed native text length in1..256")
    return {"height": height, "width": width, "text_tokens": length,
            "image_tokens": (height // 16) * (width // 16)}


def validate_options(options):
    validate_policy(options)  # Policy and scalars before any path access/allocation.
    geometry(options)
    allowed = {"model_dir", "model_lock_sha256", "cache_dir", "cache_index_sha256",
               "device", "base_precision", "height", "width", "text_tokens", "rank", "alpha",
               "steps", "batch", "seed", "ema_decay", "critic_width", "sigma", "response_scale",
               "pg_version", "pg_source_profile", "penalty_input_domain", "initialization_policy",
               "lr", "d_lr_mult", "prior_lr_mult"}
    if set(options) - allowed:
        raise ValueError("unknown native Flux options: " + str(sorted(set(options) - allowed)))
    if options.get("device", "cpu") not in ("cpu", "cuda:0", "cuda:1"):
        raise ValueError("declare one CPU or visible CUDA device explicitly")
    if options.get("base_precision", "bfloat16") not in ("float32", "bfloat16"):
        raise ValueError("frozen native base supports explicitly declared FP32/BF16 only")
    if (options.get("rank", 4) != 4 or type(options.get("rank", 4)) is not int
            or type(options.get("alpha", 1.)) not in (int, float) or options.get("alpha", 1.) != 1.):
        raise ValueError("this ordinary native adapter cohort preserves rank4/alpha1")
    for key in ("sigma", "response_scale"):
        positive(options.get(key, 1.), key)
    if type(options.get("critic_width", 16)) is not int or not 4 <= options.get("critic_width", 16) <= 64:
        raise ValueError("critic_width must be an integer in4..64")
    for key, filename, sha_key in (("model_dir", "flux-model.lock.json", "model_lock_sha256"),
                                  ("cache_dir", "index.json", "cache_index_sha256")):
        if not isinstance(options.get(key), str) or not Path(options[key]).is_dir():
            raise ValueError(key + " must be a staged local directory")
        if file_hash(Path(options[key]) / filename) != options.get(sha_key):
            raise ValueError(sha_key + " must bind the exact staged metadata")


def verify_model(directory, expected_lock_sha256):
    """Authenticate the explicitly staged complete pipeline, never download it.

    A lock is a preparation receipt, not proof that a pretrained pilot ran.
    Required weight-index shard references and native component files must all
    be present in its full byte/SHA manifest. Runtime reads only named files.
    """
    root = Path(directory).resolve()
    path = root / "flux-model.lock.json"
    if file_hash(path) != expected_lock_sha256:
        raise ValueError("model lock checksum differs")
    lock = json.loads(path.read_text())
    if (lock.get("schema"), lock.get("model"), lock.get("revision")) != (
            "flux-native-local-model-lock-v1", MODEL, MODEL_REVISION):
        raise ValueError("requires the exact declared FLUX.1-schnell model revision")
    files = lock.get("files")
    required = {"model_index.json", "transformer/config.json", "scheduler/scheduler_config.json",
                "vae/config.json", "text_encoder/config.json", "text_encoder_2/config.json",
                "tokenizer/tokenizer_config.json", "tokenizer_2/tokenizer_config.json"}
    if not isinstance(files, dict) or not required <= set(files):
        raise ValueError("complete native pipeline file manifest required")
    components = ("transformer", "scheduler", "vae", "text_encoder", "text_encoder_2", "tokenizer", "tokenizer_2")
    actual_names = {str(path.relative_to(root)) for component in components
                    for path in (root / component).rglob("*") if path.is_file()}
    actual_names.add("model_index.json")
    if set(files) != actual_names:
        raise ValueError("native staged component inventory differs from the complete lock")
    for name, value in files.items():
        relative = Path(name)
        local = root / relative
        if (relative.is_absolute() or ".." in relative.parts or not isinstance(value, dict)
                or set(value) != {"bytes", "sha256"} or type(value["bytes"]) is not int
                or not local.is_file() or local.stat().st_size != value["bytes"]
                or file_hash(local) != value["sha256"]):
            raise ValueError("staged native file identity differs: " + str(name))
    for component in ("transformer", "vae", "text_encoder", "text_encoder_2"):
        weights = [name for name in files if name.startswith(component + "/") and name.endswith(".safetensors")]
        indices = [name for name in files if name.startswith(component + "/") and name.endswith(".safetensors.index.json")]
        if not weights:
            raise ValueError("native component has no authenticated weights: " + component)
        for name in indices:
            mapping = json.loads((root / name).read_text()).get("weight_map", {})
            if not mapping or any(str(Path(component) / shard) not in files for shard in mapping.values()):
                raise ValueError("native index references missing unauthenticated shards")
    config = json.loads((root / "transformer/config.json").read_text())
    if any(config.get(key) != value for key, value in EXPECTED.items()) or config.get("out_channels") not in (None, 64):
        raise ValueError("requires full native Schnell packed64 architecture")
    if json.loads((root / "model_index.json").read_text()).get("_class_name") != "FluxPipeline":
        raise ValueError("requires the original native FluxPipeline")
    vae = json.loads((root / "vae/config.json").read_text())
    if (len(vae.get("block_out_channels", ())) != 4 or vae.get("latent_channels") != 16):
        raise ValueError("this packed64 geometry requires original VAE factor8/latent16")
    return {"model": MODEL, "revision": MODEL_REVISION, "files_sha256": digest(files),
            "model_lock_sha256": expected_lock_sha256}
