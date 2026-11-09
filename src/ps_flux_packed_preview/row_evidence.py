"""Snapshot native CPU response-row evidence without admitting a training game.

The model digest is a caller assertion, not a verified loader or weight identity.
The fingerprint describes the snapshot at creation; tensors remain mutable.
This is not a cache admission or training interface.
"""
from dataclasses import dataclass, replace
import hashlib
import json
import math
import re

import torch


@dataclass(frozen=True)
class RowEvidence:
    row: object
    identity: dict
    fingerprint: str


def snapshot_row(row, *, image_tokens, text_tokens, text_width, pooled_width,
                 guidance_embeds, asserted_model_sha256):
    """Clone and identify one raw-velocity row without running a native model."""
    dimensions = (image_tokens, text_tokens, text_width, pooled_width)
    if any(type(v) is not int or v < 1 for v in dimensions) or type(guidance_embeds) is not bool:
        raise ValueError("Positive native dimensions and an explicit guidance flag are required")
    if not isinstance(asserted_model_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", asserted_model_sha256):
        raise ValueError("An explicitly asserted model SHA256 is required")
    if row.schedule is not None or row.timestep is None:
        raise ValueError("Only an explicit raw-velocity timestep is supported")
    if row.scale != 1 or row.strength != 1 or not math.isfinite(row.guidance) or row.guidance < 0:
        raise ValueError("The existing frozen-positive target requires unit scale and strength")
    records = {}

    def tensor(name, value, shape):
        if (not isinstance(value, torch.Tensor) or value.device.type != "cpu" or value.dtype != torch.float32
                or tuple(value.shape) != shape or value.requires_grad or not torch.isfinite(value).all()):
            raise ValueError("Frozen finite CPU float32 tensor with native shape required: " + name)
        clone = value.detach().clone().contiguous()
        records[name] = {"shape": list(shape), "dtype": "float32",
                         "sha256": hashlib.sha256(clone.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()}
        return clone

    latent = tensor("latent", row.latent, (1, image_tokens, 64))
    if not isinstance(row.timestep, torch.Tensor):
        raise ValueError("The raw timestep must be a tensor")
    timestep = tensor("timestep", row.timestep, tuple(row.timestep.shape))
    if timestep.numel() != 1 or not 0 <= float(timestep.item()) <= 1000:
        raise ValueError("The raw timestep must contain one value in [0, 1000]")
    conditions = {}
    expected = {"embeds", "pooled", "text_ids", "image_ids", "guidance"}
    for branch in ("neutral", "positive", "negative"):
        source = getattr(row, branch)
        if not isinstance(source, dict) or set(source) != expected:
            raise ValueError("Complete native conditioning is required: " + branch)
        if (guidance_embeds and not isinstance(source["guidance"], torch.Tensor)
                or not guidance_embeds and source["guidance"] is not None):
            raise ValueError("Guidance presence contradicts the declared model configuration")
        conditions[branch] = {
            "embeds": tensor(branch + ".embeds", source["embeds"], (1, text_tokens, text_width)),
            "pooled": tensor(branch + ".pooled", source["pooled"], (1, pooled_width)),
            "text_ids": tensor(branch + ".text_ids", source["text_ids"], (text_tokens, 3)),
            "image_ids": tensor(branch + ".image_ids", source["image_ids"], (image_tokens, 3)),
            "guidance": tensor(branch + ".guidance", source["guidance"], (1,)) if guidance_embeds else None,
        }
    snapshot = replace(row, latent=latent, timestep=timestep, **conditions)
    identity = {"schema": "flux-cpu-row-evidence-v1", "model_sha256_asserted_not_verified": asserted_model_sha256,
                "native_shape": [1, image_tokens, 64], "text_tokens": text_tokens,
                "text_width": text_width, "pooled_width": pooled_width, "guidance_embeds": guidance_embeds,
                "row_guidance": row.guidance, "row_scale": row.scale, "row_strength": row.strength,
                "tensors": records}
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True, allow_nan=False).encode()).hexdigest()
    return RowEvidence(snapshot, identity, fingerprint)


def verify_snapshot(evidence):
    """Fail if a snapshot has changed since its recorded fingerprint.

    This detects accidental mutation only. The identity and fingerprint are
    caller-controlled and do not authenticate model weights or their origin.
    Verification does not admit a row to the shared game.
    """
    try:
        if not isinstance(evidence, RowEvidence) or not isinstance(evidence.identity, dict):
            raise ValueError("A row snapshot is required")
        identity = evidence.identity
        native_shape = identity["native_shape"]
        if (not isinstance(native_shape, list) or len(native_shape) != 3
                or native_shape[0] != 1 or native_shape[2] != 64):
            raise ValueError("Invalid native shape")
        checked = snapshot_row(
            evidence.row, image_tokens=native_shape[1],
            text_tokens=identity["text_tokens"], text_width=identity["text_width"],
            pooled_width=identity["pooled_width"], guidance_embeds=identity["guidance_embeds"],
            asserted_model_sha256=identity["model_sha256_asserted_not_verified"],
        )
        if checked.identity != identity or checked.fingerprint != evidence.fingerprint:
            raise ValueError("Row snapshot fingerprint mismatch")
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise ValueError("Invalid or modified row snapshot") from exc
