"""Finite operative native Source pins; no old core/Fibo package dependency."""
from importlib.metadata import version
import json
from pathlib import Path
from .contracts import file_hash

ROOT = Path(__file__).parent


def check_native(model):
    import diffusers
    from diffusers import FluxTransformer2DModel
    from diffusers.models.transformers.transformer_flux import FluxAttnProcessor
    expected = json.loads((ROOT / "native-sources.json").read_text())
    if type(model) is not FluxTransformer2DModel or version("diffusers") != expected["diffusers_version"]:
        raise ValueError("exact Diffusers0.38.0 native FluxTransformer2DModel required")
    if (any(type(p) is not FluxAttnProcessor or p._attention_backend is not None
            for p in model.attn_processors.values())
            or any(getattr(m, "fused_projections", False) or hasattr(m, "_hf_hook") for m in model.modules())):
        raise ValueError("native unfused attention and no offload/replaced hooks required")
    root = Path(diffusers.__file__).parent
    if any(file_hash(root / name) != sha for name, sha in expected["native_sources"].items()):
        raise ValueError("native Flux operative Source differs")
    if file_hash(ROOT / "lora.hndl") != expected["lora_sha256"]:
        raise ValueError("ordinary LoRA graph differs")
