"""Require the exact installed cohort before touching a staged native model."""
import hashlib
from importlib.metadata import version
import json
from pathlib import Path

from . import COHORT

ROOT = Path(__file__).parent
CORE_COMMIT = "a794e3b4a8813b012f2c4faa3da82b78d4a90214"
CORE_VERSION = "0.12.0"
CORE_PAYLOAD_POLICY = "recursive-files-except-pycache-and-bytecode-v1"


def checked_builder():
    import particle_sliders
    from particle_sliders.fibo_hndl import verified_builder
    expected = json.loads((ROOT / "sources.json").read_text())
    if version("particle-sliders-core") != CORE_VERSION or expected.get("core_version") != CORE_VERSION:
        raise ValueError("Flux prerequisite requires exact core0.12 source cohort")
    if (expected.get("cohort"), expected.get("core_payload_policy")) != (COHORT, CORE_PAYLOAD_POLICY):
        raise ValueError("Flux prerequisite shared-core profile differs")
    core_root = Path(particle_sliders.__file__).parent
    actual = {str(p.relative_to(core_root)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in core_root.rglob("*") if p.is_file() and "__pycache__" not in p.relative_to(core_root).parts
              and p.suffix not in (".pyc", ".pyo")}
    if actual != expected["core_sources"] or expected["core_commit"] != CORE_COMMIT:
        raise ValueError("Flux prerequisite shared-core source closure differs")
    if hashlib.sha256((ROOT / "lora.hndl").read_bytes()).hexdigest() != expected["lora_sha256"]:
        raise ValueError("Flux prerequisite LoRA graph differs")
    builder, identity = verified_builder()
    return builder, dict(identity, flux_core_commit=CORE_COMMIT,
                         flux_cohort=COHORT, flux_core_version=CORE_VERSION,
                         flux_core_payload_policy=CORE_PAYLOAD_POLICY,
                         flux_lora_sha256=expected["lora_sha256"])


def check_native(model):
    import diffusers
    from diffusers import FluxTransformer2DModel
    expected = json.loads((ROOT / "sources.json").read_text())
    if type(model) is not FluxTransformer2DModel or version("diffusers") != "0.38.0":
        raise ValueError("exact native Diffusers0.38 FluxTransformer2DModel required")
    from diffusers.models.transformers.transformer_flux import FluxAttnProcessor
    if (any(type(p) is not FluxAttnProcessor or p._attention_backend is not None
            for p in model.attn_processors.values())
            or any(getattr(m, "fused_projections", False) or hasattr(m, "_hf_hook") for m in model.modules())):
        raise ValueError("native unfused Flux attention processors required")
    root = Path(diffusers.__file__).parent
    if any(hashlib.sha256((root / name).read_bytes()).hexdigest() != digest
           for name, digest in expected["native_sources"].items()):
        raise ValueError("native Flux forward source closure differs")
