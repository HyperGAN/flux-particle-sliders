"""Opt-in strength changes on the original Flux finished-image consumer."""

from contextlib import ExitStack
from itertools import pairwise
from math import isfinite
from sys import float_info

from .activation_gate import LatentGatePlan
from .adapter import FluxLoRA
from .native import NativeRuntime


def resolve(strength, steps, step_gains):
    if not isinstance(step_gains, (list, tuple)) or len(step_gains) != steps:
        raise ValueError("Flux requires one gain per native scheduler step")
    if any(type(gain) not in (float, int) or not 0 <= gain <= float_info.max for gain in step_gains):
        raise ValueError("Flux step gains must be finite and nonnegative")
    strengths = [strength * gain for gain in step_gains]
    if any(not 0 <= value <= 1 for value in strengths):
        raise ValueError("Flux effective slider strengths must be in [0,1]")
    return strengths


def render(session, *, step_gains=None, gate_plan=None, strength=1.0, steps=4, **request):
    """Apply fixed gains or a calibrated gate on the original strict RAW renderer."""
    session._open()
    if session.selection != "raw" or session.binding.render is None:
        raise ValueError("Flux scheduling requires an open RAW render session")
    # The original factory retains the actual native owners in this callback.
    # Use those same objects so its packet and factory source hashes stay intact.
    owners = [cell.cell_contents for cell in session.binding.close.__closure__ or ()]
    runtimes = [owner for owner in owners if isinstance(owner, NativeRuntime)]
    adapters = [owner for owner in owners if isinstance(owner, FluxLoRA)]
    if len(runtimes) != 1 or len(adapters) != 1 or runtimes[0].identity != session.binding.identity["model"]:
        raise ValueError("expected the original Flux native factory owners")
    runtime, adapter = runtimes[0], adapters[0]
    if type(steps) is not int or not 1 <= steps <= 50:
        raise ValueError("Flux requires its supported native step count")
    if type(strength) not in (int, float) or not 0 <= strength <= 1:
        raise ValueError("Flux nominal strength must be in [0,1]")
    if gate_plan is None:
        values = resolve(strength, steps, step_gains)
    else:
        if step_gains is not None or not isinstance(gate_plan, LatentGatePlan):
            raise ValueError("Choose fixed Flux step gains or one calibrated LatentGatePlan")
        if strength == 0:
            raise ValueError("Gated Flux needs nonzero nominal strength; render OFF at strength zero")
        gate, direction = gate_plan._render_gate(runtime.transformer.config.in_channels, runtime.device)
        values = []
    trace = []
    position = 0
    sigmas = None
    scheduler = None
    scope = ExitStack()

    def before(_model, _args, kwargs):
        nonlocal position, sigmas, scheduler
        if position >= steps:
            raise RuntimeError("Flux made extra native denoiser calls")
        if sigmas is None:
            scheduler = runtime.pipe.scheduler
            sigmas = scheduler.sigmas.detach().cpu().tolist()
            if len(sigmas) != steps + 1 or any(
                not isfinite(start) or not isfinite(end) or start <= end for start, end in pairwise(sigmas)
            ):
                raise RuntimeError("Flux native sigma grid differs from the requested step count")
        if gate_plan is not None:
            expected_index = position if position else None
            if (runtime.pipe.scheduler is not scheduler or scheduler.begin_index != 0
                    or scheduler.step_index != expected_index):
                raise RuntimeError("Expected one pre-edit Flux forward per native Euler interval")
            score = gate_plan._score(kwargs.get("hidden_states"), direction, strength)
            decision = gate.step(score)
            value = strength if decision.fired else 0.0
            values.append(value)
            trace.append(dict(interval=position, projection=score, probability=decision.probability,
                              fired=decision.fired, effective_strength=value))
        scope.enter_context(adapter.scaled(values[position]))
        position += 1

    def after(_model, _args, _output):
        scope.close()

    with ExitStack() as handles:
        handles.callback(runtime.transformer.register_forward_pre_hook(before, with_kwargs=True).remove)
        handles.callback(runtime.transformer.register_forward_hook(after, always_call=True).remove)
        handles.callback(scope.close)
        image = session.render(strength=strength, steps=steps, **request)
    if position != steps:
        raise RuntimeError("Flux finished before applying all scheduled strengths")
    receipt = {"sigmas": sigmas, "effective_strengths": values}
    if gate_plan is not None:
        if scheduler.step_index != steps:
            raise RuntimeError("Flux scheduler did not complete all gated intervals")
        usage = gate.usage
        receipt.update(activation_gate=gate_plan._receipt(), decisions=trace, fired=usage.fired,
                       eligible=usage.eligible, gate_rate=usage.rate,
                       sigma_weighted_strength=sum(value * (start - end) for value, (start, end)
                                                   in zip(values, pairwise(sigmas), strict=True)))
    image.info["strength_schedule"] = receipt
    return image
