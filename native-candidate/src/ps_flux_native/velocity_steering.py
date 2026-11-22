"""Apply a fitted shared S-PSR atom to Flux's native Euler velocity."""

from collections.abc import Mapping
from hashlib import sha256
from io import BytesIO
from itertools import pairwise
from math import fsum, isfinite, sqrt
from pathlib import Path
from types import ModuleType
from zipfile import ZipFile

import torch

from .contracts import digest, geometry


CORE_REVISION = "7b26e5b06f5db91e8d3f51a1602d4e459f0f6a22"
CORE_MODULE = "particle_sliders/prompt_steering.py"
CORE_MODULE_SHA256 = "67ee03066e255fe105735820b0edba53757d943b93239b1279de707d23dbae59"


def _shared_prompt_steering(core_wheel):
    # Use the standalone verified equations without changing the RAW owner's core.
    with ZipFile(core_wheel) as archive:
        try:
            source = archive.read(CORE_MODULE)
        except KeyError as error:
            raise ValueError("Flux S-PSR requires the published shared prompt-steering module") from error
    if sha256(source).hexdigest() != CORE_MODULE_SHA256:
        raise ValueError("Flux S-PSR requires the exact shared prompt-steering module")
    core = ModuleType(f"{__name__}._shared")
    exec(compile(source, f"{core_wheel}/{CORE_MODULE}", "exec"), core.__dict__)
    return core.PromptSteering


class VelocitySteeringPlan:
    """Bind a measured, raw PromptSteering fit to one Flux velocity coordinate."""

    def __init__(self, *, core_wheel, fit_path):
        payload = Path(fit_path).read_bytes()
        fit = torch.load(BytesIO(payload), map_location="cpu", weights_only=True)
        if (not isinstance(fit, Mapping) or fit.get("weights") != "raw"
                or "ema_decay" not in fit or fit["ema_decay"] is not None):
            raise ValueError("Flux S-PSR requires fitted raw weights with EMA disabled")
        strength = fit.get("native_raw_strength")
        if type(strength) not in (int, float) or not 0 <= strength <= 1:
            raise ValueError("Flux S-PSR fit must record its fixed native RAW strength")
        measured_geometry = fit.get("geometry")
        if not isinstance(measured_geometry, dict) or geometry(measured_geometry) != measured_geometry:
            raise ValueError("Flux S-PSR fit must record its native geometry")
        sigmas = fit.get("sigmas")
        if (not isinstance(sigmas, (list, tuple)) or not 2 <= len(sigmas) <= 51
                or any(type(sigma) not in (float, int) or not 0 <= sigma <= 1 for sigma in sigmas)
                or any(a <= b for a, b in pairwise(sigmas))):
            raise ValueError("Flux S-PSR fit must record its descending native sigma grid")
        raw_sha, model_sha = (fit.get(key) for key in ("raw_packet_sha256", "model_identity_sha256"))
        if any(not isinstance(sha, str) or len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha)
               for sha in (raw_sha, model_sha)):
            raise ValueError("Flux S-PSR fit must identify its exact RAW packet and native model")
        state = fit.get("probe")
        if not isinstance(state, Mapping) or set(state) != {"probe.weight", "probe.bias", "direction.weight"}:
            raise ValueError("Provide a fitted shared PromptSteering state")
        if any(not isinstance(value, torch.Tensor) or value.dtype != torch.float32
               or not torch.isfinite(value).all() for value in state.values()):
            raise ValueError("Flux S-PSR fit needs finite FP32 parameters")
        direction = state["direction.weight"]
        if (direction.ndim != 2 or direction.shape[0] != 1 or not direction.shape[1]
                or state["probe.weight"].shape != direction.shape or state["probe.bias"].shape != (1,)):
            raise ValueError("Flux S-PSR fit must have one coefficient and one packed-channel direction")
        self._module = _shared_prompt_steering(core_wheel)
        self._state = {name: value.detach().cpu().clone() for name, value in state.items()}
        self._channels = direction.shape[1]
        self._strength = strength
        self._geometry = measured_geometry
        self._sigmas = tuple(sigmas)
        self._raw_sha256 = raw_sha
        self._model_sha256 = model_sha
        self._fit_sha256 = sha256(payload).hexdigest()

    def _probe_for(self, device):
        # Linear construction initializes weights even though these are loaded immediately.
        with torch.random.fork_rng(devices=[]):
            probe = self._module(self._channels, device="cpu")
        probe.load_state_dict(self._state, strict=True)
        return probe.eval().requires_grad_(False).to(device)


def _render_velocity(session, runtime, adapter, plan, *, strength, steps, constant_coefficient, request):
    if runtime.transformer.config.in_channels != plan._channels:
        raise ValueError("Flux S-PSR direction must match the native packed velocity channels")
    if plan._strength != strength:
        raise ValueError("Use the measured Flux S-PSR fit's fixed native RAW strength")
    manifest = session.manifest
    if manifest["specification"]["configuration"]["geometry"] != plan._geometry:
        raise ValueError("Use the Flux S-PSR fit's original native geometry")
    if manifest["sha256"] != plan._raw_sha256 or digest(runtime.identity) != plan._model_sha256:
        raise ValueError("Use the Flux S-PSR fit's exact RAW packet and native model")
    if len(plan._sigmas) != steps + 1:
        raise ValueError("Use the Flux S-PSR fit's native step count")
    image, receipt = _render_native_velocity(
        lambda: session.render(strength=strength, steps=steps, **request),
        transformer=runtime.transformer, current_scheduler=lambda: runtime.pipe.scheduler,
        current_strength=lambda: adapter.strength, strength=strength, sigmas=plan._sigmas,
        shape=(1, plan._geometry["image_tokens"], plan._channels), probe_for=plan._probe_for,
        constant_coefficient=constant_coefficient,
    )
    receipt.update(core_revision=CORE_REVISION, core_module_sha256=CORE_MODULE_SHA256, fit_sha256=plan._fit_sha256,
                   raw_packet_sha256=plan._raw_sha256, model_identity_sha256=plan._model_sha256,
                   geometry=plan._geometry.copy(), weights="raw", ema_decay=None)
    image.info["strength_schedule"] = dict(sigmas=list(plan._sigmas), effective_strengths=[float(strength)] * steps,
                                          velocity_steering=receipt)
    return image


def _render_native_velocity(render, *, transformer, current_scheduler, current_strength,
                            strength, sigmas, shape, probe_for, constant_coefficient):
    """Execute an edit after the native owner authenticates model, RAW and fit provenance."""
    if constant_coefficient is not None and (
        type(constant_coefficient) not in (float, int) or not 0 <= constant_coefficient <= torch.finfo(torch.float32).max
    ):
        raise ValueError("Provide a finite nonnegative same-atom constant coefficient")
    sigmas = tuple(sigmas)
    widths = [a - b for a, b in pairwise(sigmas)]
    steps = len(widths)
    scheduler = None
    probe = None
    steps_trace = []
    use_adaptive_coefficient = constant_coefficient is None
    fp32_constant_coefficient = None
    if constant_coefficient is not None:
        constant_coefficient = float(constant_coefficient)
        fp32_constant_coefficient = torch.tensor(constant_coefficient, dtype=torch.float32, device="cpu").item()

    def edit(_model, _args, output):
        nonlocal scheduler, probe
        position = len(steps_trace)
        if position == 0:
            scheduler = current_scheduler()
            if tuple(scheduler.sigmas.detach().cpu().tolist()) != sigmas:
                raise RuntimeError("Actual Flux Euler sigmas differ from the measured S-PSR grid")
        if (position >= steps or current_scheduler() is not scheduler or scheduler.begin_index != 0
                or scheduler.step_index != (position if position else None) or current_strength() != strength):
            raise RuntimeError("Expected one native Flux velocity per Euler interval at the fixed RAW strength")
        # FluxPipeline calls the transformer with return_dict=False before scheduler.step.
        if not isinstance(output, tuple) or len(output) != 1:
            raise RuntimeError("Expected Flux's native one-tensor velocity tuple")
        velocity = output[0]
        if (not isinstance(velocity, torch.Tensor) or not velocity.is_floating_point()
                or velocity.shape != shape or not torch.isfinite(velocity).all()):
            raise RuntimeError("Expected a finite packed Flux output velocity")
        if probe is None:
            probe = probe_for(velocity.device)
        pooled = velocity.mean(dim=(0, 1), dtype=torch.float32)
        coefficient = float(probe.coefficients(pooled)) if use_adaptive_coefficient else constant_coefficient
        applied = coefficient if use_adaptive_coefficient else fp32_constant_coefficient
        if not isfinite(coefficient):
            raise RuntimeError("Nonfinite Flux S-PSR coefficient")
        if applied:
            edit_field = applied * probe.direction.weight[0]
            changed = (velocity.float() + edit_field).to(velocity.dtype)
            realized = float(torch.linalg.vector_norm(changed.float() - velocity.float(), dim=-1).mean())
            if not isfinite(realized):
                raise RuntimeError("Nonfinite Flux S-PSR velocity edit")
        else:
            changed, realized = velocity, 0.
        steps_trace.append(dict(interval=position, raw_strength=float(strength), coefficient=coefficient,
                                applied_coefficient=applied, velocity_channel_mean=pooled.tolist(),
                                realized_mean_edit_norm=realized))
        return (changed,)

    with transformer.register_forward_hook(edit):
        image = render()
    if scheduler is None or len(steps_trace) != steps or scheduler.step_index != steps:
        raise RuntimeError("Flux did not complete the S-PSR native Euler intervals")
    direction_norm = sqrt(fsum(value * value for value in probe.direction.weight[0].detach().cpu().tolist()))
    coefficient_dose = fsum(width * row["coefficient"] for width, row in zip(widths, steps_trace, strict=True))
    fp32_dose = fsum(width * row["applied_coefficient"] for width, row in zip(widths, steps_trace, strict=True))
    realized_dose = fsum(width * row["realized_mean_edit_norm"] for width, row in zip(widths, steps_trace, strict=True))
    return image, dict(
        method="s-psr" if use_adaptive_coefficient else "same-atom-constant",
        native_raw_strength=float(strength), direction_norm=direction_norm, steps=steps_trace,
        coefficient_dose=coefficient_dose, fp32_coefficient_dose=fp32_dose,
        matched_constant_coefficient=coefficient_dose / fsum(widths),
        velocity_edit_dose=coefficient_dose * direction_norm, realized_velocity_edit_dose=realized_dose,
    )
