"""Content-bound full native teacher rows. No regeneration during training."""
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path

import torch
from particle_sliders.pg09_backends import PreparedResponse, ResponseRow
from .contracts import PROTOCOL, digest, file_hash

CONDITION_POLICY = "all-native-context-coordinates-with-time-v1-no-sketch"


def cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().contiguous().clone()
    if isinstance(value, dict):
        return {key: cpu(child) for key, child in value.items()}
    return value


def context_vector(row):
    """No information-reducing condition sketch; original response stays rank3."""
    values = [row.timestep.detach().cpu().float().reshape(-1),
              torch.tensor([row.scale, row.strength, row.guidance], dtype=torch.float32)]
    for branch in (row.neutral, row.positive, row.negative):
        if not isinstance(branch, dict) or set(branch) != {"embeds", "pooled", "text_ids", "image_ids", "guidance"}:
            raise ValueError("complete original Flux conditioning required")
        for name in ("embeds", "pooled", "text_ids", "image_ids"):
            tensor = branch[name]
            if not isinstance(tensor, torch.Tensor) or not torch.isfinite(tensor).all():
                raise ValueError("native context coordinates must be finite tensors")
            values.append(tensor.detach().cpu().float().reshape(-1))
        if branch["guidance"] is not None:
            raise ValueError("Schnell native context has no guidance embedding")
    return torch.cat(values)


def tensor_identity(value):
    value = value.detach().cpu().contiguous()
    return {"shape": list(value.shape), "dtype": str(value.dtype),
            "sha256": hashlib.sha256(value.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()}


def context_identity(row):
    return digest({name: {key: tensor_identity(value) if isinstance(value, torch.Tensor) else value
                         for key, value in getattr(row, name).items()}
                   for name in ("neutral", "positive", "negative")} | {
                       "timestep": tensor_identity(row.timestep), "guidance": row.guidance,
                       "scale": row.scale, "strength": row.strength})


def validate_row(prepared, *, shape, text_tokens, text_width, pooled_width):
    row = prepared.row
    if (row.schedule is not None or row.scale != 1 or row.strength != 1 or row.guidance != 1
            or not isinstance(row.timestep, torch.Tensor) or row.timestep.numel() != 1
            or row.timestep.requires_grad or row.timestep.dtype != torch.float32
            or not torch.isfinite(row.timestep).all() or not 0 <= float(row.timestep) <= 1000):
        raise ValueError("frozen unipolar raw-velocity row required")
    for value in (row.latent, prepared.target, prepared.neutral):
        if (not isinstance(value, torch.Tensor) or tuple(value.shape) != shape
                or value.dtype != torch.float32 or value.requires_grad or not torch.isfinite(value).all()):
            raise ValueError("full frozen FP32 native packed response/latent required")
    expected = {"embeds": (1, text_tokens, text_width), "pooled": (1, pooled_width),
                "text_ids": (text_tokens, 3), "image_ids": (shape[1], 3)}
    context_vector(row)
    for branch in (row.neutral, row.positive, row.negative):
        if any(tuple(branch[key].shape) != value or branch[key].requires_grad
               or branch[key].dtype != torch.float32 for key, value in expected.items()):
            raise ValueError("frozen native context shape differs")
        if not torch.equal(branch["image_ids"], row.neutral["image_ids"]):
            raise ValueError("matched teacher branches must share exact native image position IDs")


def write_row(directory, index, prepared):
    from dataclasses import asdict
    stored = {"row": cpu(asdict(prepared.row)), "target": cpu(prepared.target), "neutral": cpu(prepared.neutral)}
    path = Path(directory) / f"row-{index:05d}.pt"
    with path.open("xb") as stream:
        torch.save(stored, stream)
    row = ResponseRow(**stored["row"])
    return {"path": path.name, "sha256": file_hash(path), "context_sha256": context_identity(row),
            "condition_sha256": tensor_identity(context_vector(row))["sha256"]}


class CachedResponses:
    def __init__(self, directory, runtime, index_sha256, geometry):
        self.root, self.runtime, self.index_sha256 = Path(directory).resolve(), runtime, index_sha256
        if file_hash(self.root / "index.json") != index_sha256:
            raise ValueError("frozen cache index checksum differs")
        self.index = json.loads((self.root / "index.json").read_text())
        self.geometry = deepcopy(geometry)
        self.response_shape = (1, geometry["image_tokens"], 64)
        if (self.index.get("protocol") != PROTOCOL or self.index.get("model") != runtime.identity
                or self.index.get("geometry") != geometry or self.index.get("condition_policy") != CONDITION_POLICY
                or self.index.get("response_shape") != list(self.response_shape)):
            raise ValueError("cache protocol/native model/geometry/conditioning differs")
        self.records = deepcopy(self.index.get("rows"))
        if not isinstance(self.records, list) or not 2 <= len(self.records) <= 4096:
            raise ValueError("requires2..4096 actual frozen native rows")
        self._identity = deepcopy(self.index)
        self.condition_width = None
        self.image_ids = None
        for index in range(len(self.records)):
            prepared = self._load(index, move=False)
            feature_width = context_vector(prepared.row).numel()
            if self.condition_width is None:
                self.condition_width = feature_width
            elif self.condition_width != feature_width:
                raise ValueError("all rows require the same complete native conditioning dimensions")
            ids = prepared.row.neutral["image_ids"]
            if self.image_ids is None:
                self.image_ids = ids.clone()
            elif not torch.equal(ids, self.image_ids):
                raise ValueError("this fixed native geometry requires one exact image-ID grid")
        expected_ids = runtime.helper._prepare_latent_image_ids(1, geometry["height"] // 16,
            geometry["width"] // 16, "cpu", torch.float32)
        if not torch.equal(self.image_ids.cpu(), expected_ids):
            raise ValueError("frozen image position IDs differ from the actual native grid")

    def __len__(self):
        return len(self.records)

    def identity(self):
        if (file_hash(self.root / "index.json") != self.index_sha256 or self.index != self._identity
                or self.records != self._identity["rows"] or self.runtime.identity != self._identity["model"]):
            raise ValueError("frozen cache identity changed")
        return deepcopy(self._identity)

    def _load(self, index, *, move):
        record = self.records[index]
        relative = Path(record["path"])
        path = (self.root / relative).resolve()
        if relative.is_absolute() or ".." in relative.parts or not path.is_relative_to(self.root):
            raise ValueError("cache row must stay inside its frozen directory")
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != record["sha256"]:
            raise ValueError("cache row bytes differ")
        stored = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
        if not isinstance(stored, dict) or set(stored) != {"row", "target", "neutral"}:
            raise ValueError("malformed frozen teacher row")
        prepared = PreparedResponse(ResponseRow(**stored["row"]), stored["target"], stored["neutral"])
        validate_row(prepared, shape=self.response_shape, text_tokens=self.geometry["text_tokens"],
            text_width=self.runtime.transformer.config.joint_attention_dim,
            pooled_width=self.runtime.transformer.config.pooled_projection_dim)
        if (context_identity(prepared.row) != record["context_sha256"]
                or tensor_identity(context_vector(prepared.row))["sha256"] != record["condition_sha256"]):
            raise ValueError("cache full conditioning identity differs")
        if not move:
            return prepared
        def to_device(value):
            if isinstance(value, torch.Tensor):
                return value.to(self.runtime.device)
            if isinstance(value, dict):
                return {key: to_device(child) for key, child in value.items()}
            return value
        return PreparedResponse(ResponseRow(**to_device(stored["row"])),
                                stored["target"].to(self.runtime.device), stored["neutral"].to(self.runtime.device))

    def __getitem__(self, index):
        self.identity()
        return self._load(index, move=True)

    def conditions(self, indices):
        self.identity()
        # Bind the exact same authenticated full contexts as row fetches. Do not
        # rely on a caller-mutable feature cache or derive features from row IDs.
        return torch.stack([context_vector(self._load(int(index), move=False).row)
                            for index in indices]).to(self.runtime.device)


@torch.no_grad()
def prepare_cache(runtime, adapter, pairs, directory, *, geometry, seeds=(7, 19), steps=4):
    """Frozen positive velocities on native matched neutral trajectories, once."""
    import numpy as np
    from diffusers.pipelines.flux.pipeline_flux import calculate_shift, retrieve_timesteps
    from .native import PackedFluxResponse
    if runtime.pipe is None or type(steps) is not int or not 1 <= steps <= 50:
        raise ValueError("cache preparation needs the full frozen native pipeline and explicit steps1..50")
    if (not isinstance(pairs, list) or len(pairs) < 1 or any(not isinstance(pair, dict)
            or set(pair) != {"neutral", "positive"} or any(not isinstance(v, str) or not v.strip()
            for v in pair.values()) or pair["neutral"] == pair["positive"] for pair in pairs)):
        raise ValueError("distinct nonempty neutral/positive prompt pairs required")
    if not seeds or any(type(seed) is not int or not 0 <= seed < 2**63 - 100 for seed in seeds):
        raise ValueError("explicit fixed native seeds required")
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=False)
    pipe, records = runtime.pipe, []
    backend = PackedFluxResponse(runtime, adapter)
    for pair in pairs:
        neutral = runtime.encode(pair["neutral"], geometry["text_tokens"])
        positive = runtime.encode(pair["positive"], geometry["text_tokens"])
        for seed in seeds:
            generator = torch.Generator(device=runtime.device).manual_seed(seed)
            packed, ids = pipe.prepare_latents(1, 16, geometry["height"], geometry["width"],
                                               runtime.dtype, runtime.device, generator)
            if tuple(packed.shape) != (1, geometry["image_tokens"], 64):
                raise ValueError("actual native packing differs from declared geometry")
            conditions = [{**value, "image_ids": ids.detach().float(), "guidance": None}
                          for value in (neutral, positive)]
            scheduler = deepcopy(pipe.scheduler)
            sigmas = np.linspace(1.0, 1 / steps, steps)
            if getattr(scheduler.config, "use_flow_sigmas", False):
                sigmas = None
            mu = calculate_shift(packed.shape[1], scheduler.config.get("base_image_seq_len", 256),
                scheduler.config.get("max_image_seq_len", 4096), scheduler.config.get("base_shift", .5),
                scheduler.config.get("max_shift", 1.15))
            times, _ = retrieve_timesteps(scheduler, steps, runtime.device, sigmas=sigmas, mu=mu)
            for timestep in times:
                row = ResponseRow(conditions[0], conditions[1], conditions[0], packed.detach().float(),
                                  timestep=timestep.detach().float())
                prepared = backend.prepare(row)
                records.append(write_row(root, len(records), prepared))
                packed = scheduler.step(prepared.neutral.to(runtime.dtype), timestep, packed, return_dict=False)[0]
    index = {"protocol": PROTOCOL, "model": runtime.identity, "geometry": geometry,
             "response_shape": [1, geometry["image_tokens"], 64], "pairs": pairs, "seeds": list(seeds),
             "scheduler_steps": steps, "teacher": "frozen positive native raw velocity on neutral trajectory",
             "condition_policy": CONDITION_POLICY, "rows": records}
    (root / "index.json").write_text(json.dumps(index, indent=2, allow_nan=False) + "\n")
    return file_hash(root / "index.json")
