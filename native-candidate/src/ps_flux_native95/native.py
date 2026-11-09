"""Local frozen Flux loader and unchanged packed velocity boundary."""
from contextlib import contextmanager
from copy import deepcopy
from importlib.metadata import version
from pathlib import Path

import torch
from particle_sliders.pg09_backends import FrozenResponseAdapter
from .contracts import PROTOCOL, SOURCE_PROFILE, file_hash, verify_model
from .provenance import check_native


class NativeRuntime:
    def __init__(self, directory, *, model_lock_sha256, device="cpu", precision="bfloat16", encoder=False):
        from diffusers import FluxPipeline, FluxTransformer2DModel
        if device not in ("cpu", "cuda:0", "cuda:1") or precision not in ("float32", "bfloat16"):
            raise ValueError("explicit FP32/BF16 and one CPU/visible CUDA device required")
        self.identity = verify_model(directory, model_lock_sha256)
        self.model_directory = str(Path(directory).absolute())
        self.device, self.dtype = torch.device(device), getattr(torch, precision)
        self.helper, self.pipe, self.transformer = FluxPipeline, None, None
        try:
            if encoder:
                # All pipeline weights first load on CPU. Move only the one
                # DiT here; frozen text encoders/VAE stay on CPU until render.
                self.pipe = FluxPipeline.from_pretrained(directory, torch_dtype=self.dtype, local_files_only=True)
                self.transformer = self.pipe.transformer
                for component in (self.pipe.text_encoder, self.pipe.text_encoder_2, self.pipe.vae):
                    component.requires_grad_(False).eval()
            else:
                self.transformer = FluxTransformer2DModel.from_pretrained(directory, subfolder="transformer",
                    torch_dtype=self.dtype, local_files_only=True)
            self.transformer.requires_grad_(False).eval().to(self.device)
            check_native(self.transformer)
            if self.pipe is not None and self.pipe.vae_scale_factor != 8:
                raise ValueError("native packed64 geometry requires the original factor8 VAE")
            self.identity.update(self._execution_identity())
        except BaseException:
            self.close()
            raise

    def _execution_identity(self):
        return {"source_profile": SOURCE_PROFILE,
                "product_sources": {name: file_hash(Path(__file__).with_name(name)) for name in
                    ("native.py", "adapter.py", "builder.py", "builder-sources.json", "native-sources.json", "lora.hndl", "render.py")},
                "base_precision": str(self.dtype), "device_type": self.device.type,
                "runtime_versions": {key: version(key) for key in ("torch", "diffusers", "transformers", "hndl")},
                "numerics": {"deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                    "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
                    "tf32_cudnn": torch.backends.cudnn.allow_tf32}}

    @torch.no_grad()
    def encode(self, text, text_tokens):
        if self.pipe is None:
            raise ValueError("cache preparation requires a separately staged full pipeline")
        embeddings, pooled, text_ids = self.pipe.encode_prompt(prompt=text, prompt_2=None, device="cpu",
                                                              max_sequence_length=text_tokens)
        return {"embeds": embeddings.detach().float(), "pooled": pooled.detach().float(),
                "text_ids": text_ids.detach().float()}

    def load_renderer(self):
        if self._execution_identity() != {key: self.identity[key] for key in self._execution_identity()}:
            raise RuntimeError("native runtime/operative Source changed after identity construction")
        if self.pipe is None:
            self.pipe = self.helper.from_pretrained(self.model_directory, transformer=self.transformer,
                                                    torch_dtype=self.dtype, local_files_only=True)
        if self.pipe.transformer is not self.transformer or self.pipe.vae_scale_factor != 8:
            raise RuntimeError("renderer must reuse the exact ONE bound native DiT and original VAE geometry")
        for component in (self.pipe.transformer, self.pipe.text_encoder, self.pipe.text_encoder_2, self.pipe.vae):
            component.requires_grad_(False).eval()
        # Caller must budget the full DiT+two encoders+VAE render placement.
        self.pipe.to(self.device)
        return self.pipe

    def close(self):
        self.pipe, self.transformer = None, None


class PackedFluxResponse(FrozenResponseAdapter):
    family = "flux-native-packed-ordinary-lora"
    time_coordinates = "native raw scheduler t/1000; raw packed velocity, no scheduler.step"

    def __init__(self, runtime, network):
        self.runtime, self.transformer = runtime, runtime.transformer
        if network._model is not self.transformer:
            raise ValueError("bind the actual owned adapter and native model")
        self._config = deepcopy(dict(self.transformer.config))
        super().__init__(network, model_identity=runtime.identity)

    def _velocity(self, condition, latent, timestep):
        c = condition
        if dict(self.transformer.config) != self._config:
            raise ValueError("native model configuration changed")
        if (not isinstance(latent, torch.Tensor) or tuple(latent.shape) != (1, self.network.image_tokens, 64)
                or not isinstance(c, dict) or set(c) != {"embeds", "pooled", "text_ids", "image_ids", "guidance"}
                or c["guidance"] is not None):
            raise ValueError("full native Schnell packed64 latent and conditioning required")
        shapes = {"embeds": (1, self.network.text_tokens, self._config["joint_attention_dim"]),
                  "pooled": (1, self._config["pooled_projection_dim"]),
                  "text_ids": (self.network.text_tokens, 3), "image_ids": (self.network.image_tokens, 3)}
        if (not isinstance(timestep, torch.Tensor) or timestep.numel() != 1
                or not torch.isfinite(timestep).all() or not 0 <= float(timestep) <= 1000):
            raise ValueError("one finite original raw timestep in[0,1000] required")
        for name, shape in shapes.items():
            value = c[name]
            if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape or not torch.isfinite(value).all():
                raise ValueError("native conditioning geometry/value differs: " + name)
        if not latent.is_floating_point() or not torch.isfinite(latent).all():
            raise ValueError("finite native floating latent required")
        z = latent.to(device=self.runtime.device, dtype=self.runtime.dtype)
        result = self.transformer(hidden_states=z, timestep=timestep.reshape(1).to(z) / 1000,
            encoder_hidden_states=c["embeds"].to(z), pooled_projections=c["pooled"].to(z),
            txt_ids=c["text_ids"].to(z), img_ids=c["image_ids"].to(z), guidance=None,
            return_dict=False)[0]
        if result.shape != latent.shape or not torch.isfinite(result).all():
            raise ValueError("native Flux velocity geometry/values differ")
        return result.float()

    def prepare(self, row):
        if row.schedule is not None or row.guidance != 1 or row.scale != 1 or row.strength != 1:
            raise ValueError("new cohort is unipolar frozen positive raw velocity; native CFG=1")
        return super().prepare(row)

    def metadata(self):
        return dict(super().metadata(), protocol=PROTOCOL, source_profile=SOURCE_PROFILE,
            model_config=self._config, native_geometry=[1, self.network.image_tokens, 64],
            text_tokens=self.network.text_tokens, target_names=list(self.network.target_names),
            ordinary_lora=True, particle_bank=None, gradient_checkpointing=False,
            fresh_initialization=getattr(self, "fresh_initialization", None))


@contextmanager
def observation(runtime, adapter):
    from particle_sliders.lifecycle import native_cleanup
    restore = native_cleanup(runtime.transformer)
    old_strength = adapter.strength
    buffers = [(value, value.detach().clone()) for value in runtime.transformer.buffers()]
    try:
        yield
    finally:
        from particle_sliders.restoration import _restore_all
        def restore_buffer(value, saved):
            with torch.no_grad():
                value.copy_(saved)
        _restore_all([*(lambda value=value, saved=saved: restore_buffer(value, saved) for value, saved in buffers),
                      lambda: setattr(adapter, "strength", old_strength), restore])
