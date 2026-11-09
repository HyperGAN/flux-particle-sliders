"""Product factory over shared lifecycle/official PG; no product GAN update loop."""
from copy import deepcopy
from pathlib import Path
import torch

from particle_sliders.lifecycle import Binding, Capabilities, UnsupportedOperation, native_cleanup
from particle_sliders.response import ResponseBackend, ResponseCapabilities
from particle_sliders.inference import InferenceBackend, InferenceBinding, native_identity
from particle_sliders.pg095 import ConditionalPG095, engine_options
from particle_sliders.restoration import _restore_all
from .contracts import (BACKEND_NAME, PROTOCOL, SOURCE_PROFILE, INITIALIZATION_POLICY, WEIGHT_POLICY,
                        digest, geometry, rate_overrides, validate_options)
from .initialization import Construction, construction_for, initialize_owned, validate_receipt
from .adapter import FluxLoRA
from .native import NativeRuntime, PackedFluxResponse, observation
from .critic import PackedCritic
from .cache import CachedResponses


def capabilities(options):
    shape = geometry(options)
    return ResponseCapabilities("adapter_gan", "image", "native-packed-velocity", "continuous_packed",
        (1, shape["image_tokens"], 64), "0.9.5", "release-0.9.5", penalty_input_domain="auto",
        base_precisions=("float32", "bfloat16"), qualification="New native Source; matching CPU/pretrained/GPU checks pending")


INFERENCE_CAPABILITIES = Capabilities("adapter_gan", render=True, render_outputs=("pil",),
    base_precisions=("float32", "bfloat16"), operations=("response", "render"),
    qualification="Native full pipeline/RAW Source; matching CPU and pretrained finished-render checks pending")


def preflight(options, request, staged):
    if set(staged) - {"runtime", "construction"}:
        raise ValueError("only explicit local runtime and fresh/resume intent may be staged")
    construction = staged.get("construction")
    if (not isinstance(construction, Construction) or construction.options_sha256 != digest(options)
            or construction.mode not in ("fresh", "resume")):
        raise ValueError("use open_training to bind actual fresh/resume intent before native allocation")
    if request["native"]:
        raise UnsupportedOperation("training accepts its content-bound full native velocity cache only")
    runtime = staged.get("runtime")
    if runtime is not None and (str(runtime.device) != options.get("device", "cpu")
            or runtime.dtype != getattr(torch, options.get("base_precision", "bfloat16"))):
        raise ValueError("borrowed runtime placement/precision differs from session")


def prepare(options, *, runtime=None, construction=None):
    validate_options(options)
    preflight(options, {"native": {}}, {"runtime": runtime, "construction": construction})
    owns = runtime is None
    if owns:
        runtime = NativeRuntime(options["model_dir"], model_lock_sha256=options["model_lock_sha256"],
            device=options.get("device", "cpu"), precision=options.get("base_precision", "bfloat16"))
    restore, adapter, closed = native_cleanup(runtime.transformer), None, False
    def close():
        nonlocal closed
        if closed:
            return
        _restore_all([*([] if adapter is None else [adapter.close]), restore,
                      *([runtime.close] if owns else [])])
        closed = True
    try:
        shape = geometry(options)
        rows = CachedResponses(options["cache_dir"], runtime, options["cache_index_sha256"], shape)
        devices = [] if runtime.device.type == "cpu" else [runtime.device.index]
        with torch.random.fork_rng(devices=devices):
            from particle_sliders.field_critic import seed_initialization_rng
            seed_initialization_rng(options.get("seed", 7), devices)
            adapter = FluxLoRA(runtime.transformer, image_tokens=shape["image_tokens"], text_tokens=shape["text_tokens"],
                               rank=4, alpha=1.).to(device=runtime.device, dtype=torch.float32)
            critic = PackedCritic(rows.image_ids, rows.condition_width, options.get("critic_width", 16)).to(runtime.device)
            receipt = initialize_owned(adapter, critic, construction, options)  # Before optimizers.
            adapter.attach()
            backend = PackedFluxResponse(runtime, adapter)
            backend.fresh_initialization = receipt.as_dict()
        game = ConditionalPG095(backend, rows, critic, particles=None, response_domain="continuous_packed",
            batch_size=options.get("batch", 2), total_steps=options.get("steps", 4), seed=options.get("seed", 7),
            sigma=options.get("sigma", 1.), response_scale=options.get("response_scale", 1.),
            recipe_overrides=rate_overrides(options), **engine_options(options))
        configuration = {"geometry": shape, "rank": 4, "alpha": 1., "weight_policy": WEIGHT_POLICY,
                         "source_profile": SOURCE_PROFILE, "initialization_policy": INITIALIZATION_POLICY,
                         "initialization": receipt.as_dict()}
        spec = NATIVE_BACKEND.specification(backend, model_identity=runtime.identity,
                                           protocol=PROTOCOL, configuration=configuration)
        return Binding(game, close, inference=spec, before_update=lambda request: bool(rows.identity()),
                       observation=lambda: observation(runtime, adapter))
    except BaseException as primary:
        _restore_all([close], primary=primary)
        raise


def validate_inference(options, spec):
    if set(options) - {"model_dir", "model_lock_sha256", "device", "base_precision"}:
        raise ValueError("RAW inference accepts local model identity/placement only; no cache or game")
    if (not isinstance(options.get("model_dir"), str) or not Path(options["model_dir"]).is_dir()
            or options.get("device", "cpu") not in ("cpu", "cuda:0", "cuda:1")
            or options.get("base_precision", "bfloat16") not in ("float32", "bfloat16")):
        raise ValueError("explicit local native model and supported placement/precision required")
    configuration = spec.configuration
    if (spec.identity["protocol"] != PROTOCOL or set(configuration) != {
            "geometry", "rank", "alpha", "weight_policy", "source_profile", "initialization_policy", "initialization"}
            or configuration["source_profile"] != SOURCE_PROFILE or configuration["weight_policy"] != WEIGHT_POLICY
            or configuration["initialization_policy"] != INITIALIZATION_POLICY
            or configuration["rank"] != 4 or configuration["alpha"] != 1.
            or geometry(configuration["geometry"]) != configuration["geometry"]):
        raise ValueError("requires this exact new native packed Flux RAW cohort; no old artifact retagging")
    validate_receipt(configuration["initialization"], configuration["initialization"].get("cohort_id"))


def preflight_inference(options, request, spec):
    if request["operation"] == "render":
        from .render import validate_render_request
        native = request["native"]
        if set(native) - {"prompt", "height", "width", "steps", "images"}:
            raise ValueError("unknown native finished render arguments")
        if native.get("images") is not None:
            raise ValueError("native Flux finished render is text-only; reference images are unsupported")
        validate_render_request(geometry=spec.configuration["geometry"], prompt=native.get("prompt"),
            height=native.get("height"), width=native.get("width"), steps=native.get("steps", 4),
            strength=request["strength"], seed=0, output_type=request["output_type"])
    elif request["native"] and set(request["native"]) != {"condition", "latent", "timestep"}:
        raise ValueError("native response requires exact condition/latent/timestep together")


def prepare_inference(options, spec):
    validate_inference(options, spec)
    runtime = NativeRuntime(options["model_dir"], model_lock_sha256=options["model_lock_sha256"],
        device=options.get("device", "cpu"), precision=options.get("base_precision", "bfloat16"))
    adapter = None
    def close():
        _restore_all([*([] if adapter is None else [adapter.close]), runtime.close])
    try:
        shape = spec.configuration["geometry"]
        with torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed(0)
            adapter = FluxLoRA(runtime.transformer, image_tokens=shape["image_tokens"], text_tokens=shape["text_tokens"],
                               rank=4, alpha=1.).to(device=runtime.device, dtype=torch.float32).attach()
        backend = PackedFluxResponse(runtime, adapter)
        # Restore ORIGINAL initialization lineage without invoking any initializer.
        backend.fresh_initialization = spec.configuration["initialization"]
        identity = native_identity(backend, model_identity=runtime.identity, protocol=PROTOCOL)
        def response(*, strength, seed, condition, latent, timestep):
            with adapter.scaled(strength):
                return backend._velocity(condition, latent, torch.as_tensor(timestep)).detach().clone()
        def render(*, strength, seed, images=None, output_type="pil", **native):
            if images is not None:
                raise ValueError("native Flux finished render is text-only; reference images are unsupported")
            from .render import render_image
            return render_image(runtime, adapter, geometry=shape, strength=strength, seed=seed,
                                output_type=output_type, **native)
        return InferenceBinding(adapter, identity, close, response=response, render=render,
                                observation=lambda: observation(runtime, adapter))
    except BaseException as primary:
        _restore_all([close], primary=primary)
        raise


NATIVE_BACKEND = InferenceBackend(BACKEND_NAME, INFERENCE_CAPABILITIES, prepare_inference, validate_inference,
                                  preflight=preflight_inference)
BACKEND = ResponseBackend(BACKEND_NAME, capabilities({}), prepare, validate_options,
                          configure_capabilities=capabilities, preflight=preflight)


def open_training(config, *, resume=None, **staged):
    """Bind true fresh/restore intent before prepare and any optimizer construction."""
    from particle_sliders.lifecycle import read_packet
    validate_options(config.options)
    packet = deepcopy(resume) if isinstance(resume, dict) else read_packet(resume) if resume is not None else None
    construction = construction_for(config, packet)
    session = BACKEND.open(config, operation="resume" if packet is not None else "train", construction=construction, **staged)
    try:
        if packet is not None:
            session.restore(packet)
        return session
    except BaseException as primary:
        _restore_all([session.close], primary=primary)
        raise
