"""Real native Flux finished PIL/PNG output from exact RAW exports."""
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import io
import json
import math
import os
from pathlib import Path

RENDER_PROTOCOL = "flux-native-packed95-matched-OFF-ON-images-v1"


def validate_render_request(*, geometry, prompt, height=None, width=None, steps=4,
                            seed=7, strength=1., output_type="pil"):
    if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 16384:
        raise ValueError("a nonempty native prompt of at most16384 characters is required")
    height = geometry["height"] if height is None else height
    width = geometry["width"] if width is None else width
    if type(height) is not int or type(width) is not int or (height, width) != (geometry["height"], geometry["width"]):
        raise ValueError("HNDL fixed native token graph requires exact trained height/width; no resize/reshape")
    if type(steps) is not int or not 1 <= steps <= 50:
        raise ValueError("native scheduler steps must be an integer1..50")
    if type(seed) is not int or not 0 <= seed < 2**63 - 100:
        raise ValueError("explicit native seed required")
    if type(strength) not in (int, float) or not math.isfinite(strength) or not 0 <= strength <= 1:
        raise ValueError("native unipolar adapter strength must be finite in[0,1]")
    if output_type != "pil":
        raise ValueError("finished native output is a true PIL image")


@contextmanager
def native_pipeline_state(pipe):
    scheduler = pipe.scheduler
    saved = {key: (key in vars(pipe), vars(pipe).get(key)) for key in
             ("_guidance_scale", "_joint_attention_kwargs", "_current_timestep", "_interrupt", "_num_timesteps")}
    pipe.scheduler = deepcopy(scheduler)
    try:
        yield
    finally:
        pipe.scheduler = scheduler
        for key, (present, value) in saved.items():
            if present:
                setattr(pipe, key, value)
            else:
                vars(pipe).pop(key, None)


def render_image(runtime, adapter, *, geometry, prompt, height=None, width=None,
                 steps=4, seed=7, strength=1., output_type="pil"):
    validate_render_request(geometry=geometry, prompt=prompt, height=height, width=width,
                            steps=steps, seed=seed, strength=strength, output_type=output_type)
    import torch
    pipe = runtime.load_renderer()
    if pipe.transformer is not runtime.transformer or adapter._model is not runtime.transformer:
        raise RuntimeError("RAW adapter and native finished renderer must share the actual ONE bound DiT")
    generator = torch.Generator(device=runtime.device).manual_seed(seed)
    with native_pipeline_state(pipe), adapter.scaled(strength):
        result = pipe(prompt=prompt, prompt_2=None, negative_prompt=None,
            height=geometry["height"], width=geometry["width"], num_inference_steps=steps,
            guidance_scale=0., true_cfg_scale=1., num_images_per_prompt=1,
            generator=generator, max_sequence_length=geometry["text_tokens"],
            output_type="pil", return_dict=True)
    from PIL.Image import Image
    if (not isinstance(result.images, list) or len(result.images) != 1 or not isinstance(result.images[0], Image)
            or result.images[0].size != (geometry["width"], geometry["height"])):
        raise ValueError("native pipeline must produce exactly one full requested-size PIL image")
    return result.images[0]


def _record(path, raw=None):
    path = Path(path)
    raw = path.read_bytes() if raw is None else raw
    return {"path": str(path.absolute()), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def _write(path, value):
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


def render_pair(*, artifact, model_dir, model_lock_sha256, output, prompt, steps=4, seed=7,
                strength=1., device="cpu", precision="bfloat16"):
    """No training factory/cache/critic/optimizer; two genuine same-seed PNGs.

    Core validates the exact original RAW bytes/spec/native identity, and owns
    its network/RNG observation. Failure keeps completed outputs; never retry.
    This is an execution entry point, not a claim that a native pilot ran.
    """
    output = Path(output)
    if output.exists():
        raise FileExistsError("use a fresh native render output directory")
    raw = Path(artifact).read_bytes()
    artifact_record = _record(artifact, raw)
    import torch
    from particle_sliders.inference import open_inference
    from .lifecycle import NATIVE_BACKEND
    packet = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    del raw
    if not isinstance(packet, dict) or packet.get("selection") != "raw":
        raise ValueError("original RAW current weights required")
    shape = packet["specification"]["configuration"]["geometry"]
    validate_render_request(geometry=shape, prompt=prompt, steps=steps, seed=seed, strength=strength)
    native = {"prompt": prompt, "height": shape["height"], "width": shape["width"], "steps": steps}
    options = {"model_dir": str(Path(model_dir).absolute()), "model_lock_sha256": model_lock_sha256,
               "device": device, "base_precision": precision}
    with open_inference(packet, backend=NATIVE_BACKEND, operation="render", strength=strength,
                        options=options, output_type="pil", native_request=native) as session:
        output.mkdir(parents=True, exist_ok=False)
        receipt = {"protocol": RENDER_PROTOCOL, "status": "running", "artifact": artifact_record,
                   "original_artifact_internal_sha256": packet["sha256"], "origin": packet["origin"],
                   "specification": packet["specification"], "selection": "raw", "ema_decay": None,
                   "request": dict(native, seed=seed, ON_strength=strength), "options": options,
                   "geometry": shape, "matched_OFF_strength": 0., "native_guidance_scale": 0.,
                   "true_cfg_scale": 1., "images": [], "training_updates": 0, "teacher_cache_used": False,
                   "native_qualification": None}
        _write(output / "render-receipt.json", receipt)
        try:
            for label, alpha in (("OFF", 0.), ("ON", strength)):
                image = session.render(strength=alpha, seed=seed, output_type="pil", **native)
                path = output / (label + ".png")
                with path.open("xb") as stream:
                    image.save(stream, format="PNG"); stream.flush(); os.fsync(stream.fileno())
                receipt["images"].append({"label": label, "strength": alpha, "output": _record(path),
                                          "width": image.width, "height": image.height, "mode": image.mode})
                _write(output / "render-receipt.json", receipt)
            receipt["status"] = "complete"
        except BaseException as error:
            receipt.update(status="failed", error=repr(error), completed_outputs_retained=True)
            raise
        finally:
            _write(output / "render-receipt.json", receipt)
    return receipt
