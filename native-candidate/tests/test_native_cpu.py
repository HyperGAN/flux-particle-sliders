"""Authored native CPU qualification, not an assertion that it has run.

Uses real tiny Diffusers Flux/VAE + Transformers CLIP/T5, local toy tokenizers,
and the exact external HNDL builder. No pretrained weights/network/download.
The tiny loader fixture is labeled synthetic and never claims a model lock.
"""
from copy import deepcopy
from hashlib import sha256
from itertools import pairwise
import json
from math import fsum
from pathlib import Path
from statistics import fmean, pstdev
from unittest.mock import patch

import pytest
import torch
from diffusers import (AutoencoderKL, FluxPipeline, FluxTransformer2DModel,
                       FlowMatchEulerDiscreteScheduler)
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer, T5Config, T5EncoderModel, T5TokenizerFast
from tokenizers import Tokenizer, models, pre_tokenizers
from particle_sliders.lifecycle import SessionConfig
from particle_sliders.backends import ResponseRow
from particle_sliders.engine import _hash

from ps_flux_native.adapter import FluxLoRA
from ps_flux_native.cache import CONDITION_POLICY, CachedResponses, write_row
from ps_flux_native.contracts import INITIALIZATION_POLICY, PROTOCOL, digest, file_hash, geometry
from ps_flux_native.native import NativeRuntime, PackedFluxResponse
from ps_flux_native.lifecycle import open_training, NATIVE_BACKEND
from ps_flux_native.render import native_pipeline_state, render_pair


@pytest.fixture(autouse=True)
def isolated_rng():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(95)
        yield


class FixtureRuntime(NativeRuntime):
    """Inspectable test-owned native loader, intentionally not pretrained."""
    def __init__(self, directory, **unused):
        self.device, self.dtype = torch.device("cpu"), torch.float32
        self.model_directory = str(directory)
        self.helper = FluxPipeline
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(111)
            self.transformer = FluxTransformer2DModel(in_channels=64, out_channels=64,
                num_layers=1, num_single_layers=1, attention_head_dim=8, num_attention_heads=2,
                joint_attention_dim=12, pooled_projection_dim=8, guidance_embeds=False,
                axes_dims_rope=(2, 2, 4)).requires_grad_(False).eval()
            vae = AutoencoderKL(in_channels=3, out_channels=3,
                down_block_types=("DownEncoderBlock2D",) * 4, up_block_types=("UpDecoderBlock2D",) * 4,
                block_out_channels=(8, 8, 8, 8), layers_per_block=1, latent_channels=16,
                norm_num_groups=4, sample_size=64, scaling_factor=.3611,
                shift_factor=.1159).requires_grad_(False).eval()
            text = CLIPTextModel(CLIPTextConfig(vocab_size=6, hidden_size=8, intermediate_size=16,
                num_hidden_layers=1, num_attention_heads=2, max_position_embeddings=77,
                bos_token_id=0, eos_token_id=1, pad_token_id=1)).requires_grad_(False).eval()
            text2 = T5EncoderModel(T5Config(vocab_size=6, d_model=12, d_ff=16, num_layers=1,
                num_heads=2, d_kv=6, decoder_start_token_id=0, pad_token_id=0, eos_token_id=1)).requires_grad_(False).eval()
        assets = Path(directory) / "fixture-tokenizers"
        assets.mkdir(exist_ok=True)
        vocabulary = {"<|startoftext|>": 0, "<|endoftext|>": 1, "a</w>": 2, "b</w>": 3, "<unk>": 4, "x</w>": 5}
        vocab = assets / "vocab.json"; vocab.write_text(json.dumps(vocabulary))
        merges = assets / "merges.txt"; merges.write_text("#version: 0.2\n")
        tokenizer = CLIPTokenizer(vocab_file=str(vocab), merges_file=str(merges), model_max_length=77)
        engine = Tokenizer(models.WordLevel({"<pad>": 0, "</s>": 1, "<unk>": 2, "a": 3, "b": 4, "x": 5}, unk_token="<unk>"))
        engine.pre_tokenizer = pre_tokenizers.Whitespace()
        tokenizer2 = T5TokenizerFast(tokenizer_object=engine, pad_token="<pad>", eos_token="</s>",
                                     unk_token="<unk>", extra_ids=0, model_max_length=3)
        self.pipe = FluxPipeline(scheduler=FlowMatchEulerDiscreteScheduler(), vae=vae,
            text_encoder=text, tokenizer=tokenizer, text_encoder_2=text2, tokenizer_2=tokenizer2,
            transformer=self.transformer)
        self.identity = {"qualification": "synthetic native CPU fixture only; no pretrained model lock",
                         "seed": 111, **self._execution_identity()}


def conditions(runtime, *, positive=False):
    value = .5 if positive else -.5
    ids = FluxPipeline._prepare_latent_image_ids(1, 4, 4, "cpu", torch.float32)
    return {"embeds": torch.full((1, 3, 12), value), "pooled": torch.full((1, 8), value),
            "text_ids": torch.zeros(3, 3), "image_ids": ids, "guidance": None}


def fixture_cache(root, runtime):
    cache = root / "cache"; cache.mkdir()
    adapter = FluxLoRA(runtime.transformer, image_tokens=16, text_tokens=3)
    records = []
    with adapter.borrowed():
        response = PackedFluxResponse(runtime, adapter)
        for index in range(2):
            latent = torch.randn((1, 16, 64), generator=torch.Generator().manual_seed(7 + index))
            neutral, positive = conditions(runtime), conditions(runtime, positive=True)
            prepared = response.prepare(ResponseRow(neutral, positive, neutral, latent,
                                                     timestep=torch.tensor(250. + index * 400.)))
            records.append(write_row(cache, index, prepared))
    value = {"protocol": PROTOCOL, "model": runtime.identity,
             "geometry": geometry({"height": 64, "width": 64, "text_tokens": 3}),
             "condition_policy": CONDITION_POLICY, "response_shape": [1, 16, 64], "rows": records}
    (cache / "index.json").write_text(json.dumps(value, indent=2) + "\n")
    return cache


def configuration(root, runtime):
    cache = fixture_cache(root, runtime)
    lock = root / "flux-model.lock.json"; lock.write_text('{"qualification":"fixture only"}\n')
    return SessionConfig("flux_native_packed", "tiny-native-packed-CPU-only",
        model_dir=str(root), model_lock_sha256=file_hash(lock), cache_dir=str(cache),
        cache_index_sha256=file_hash(cache / "index.json"), device="cpu", base_precision="float32",
        height=64, width=64, text_tokens=3, steps=2, batch=2, seed=19, critic_width=4,
        lr=.0004, d_lr_mult=.5000000000000001, prior_lr_mult=None, ema_decay=None,
        pg_version="0.11.1", pg_source_profile="release-0.11.1", penalty_input_domain="auto",
        initialization_policy=INITIALIZATION_POLICY)


def test_actual_native_ordinary_adapter_off_geometry_and_cleanup(tmp_path):
    runtime = FixtureRuntime(tmp_path)
    state = deepcopy(runtime.transformer.state_dict())
    modes = [module.training for module in runtime.transformer.modules()]
    before = torch.get_rng_state().clone()
    network = FluxLoRA(runtime.transformer, image_tokens=16, text_tokens=3)
    assert torch.equal(before, torch.get_rng_state())
    assert not hasattr(network, "particles")
    with network.borrowed():
        response = PackedFluxResponse(runtime, network)
        c = conditions(runtime)
        latent = torch.randn(1, 16, 64)
        time = torch.tensor(600.)
        off = response.velocity(c, latent, time, scale=0.)
        on = response.velocity(c, latent, time, scale=1.)
        assert off.shape == (1, 16, 64) and torch.equal(off, on)  # zero-up neutral
        with torch.no_grad():
            network.branches[0].graph["up"].weight.fill_(.01)
        on = response.velocity(c, latent, time, scale=.25)
        assert not torch.equal(off, on)
        assert torch.equal(off, response.velocity(c, latent, time, scale=0.))
        with pytest.raises(RuntimeError, match="injected"):
            with network.scaled(.125):
                raise RuntimeError("injected")
        assert network.strength == 0.
    assert not network.attached
    assert [module.training for module in runtime.transformer.modules()] == modes
    assert all(torch.equal(value, state[name]) for name, value in runtime.transformer.state_dict().items())


def test_full_exact_resume_raw_and_fresh_before_optimizer(tmp_path):
    import ps_flux_native.initialization as init
    runtime = FixtureRuntime(tmp_path)
    config = configuration(tmp_path, runtime)
    original_init = init.initialize_fresh_components
    calls = []
    def observed(*args, **kwargs):
        assert kwargs["optimizers"] == () and kwargs["completed_steps"] == 0 and kwargs["restored_from"] is None
        calls.append("fresh-before-optimizers")
        return original_init(*args, **kwargs)
    with patch.object(init, "initialize_fresh_components", side_effect=observed):
        with open_training(config, runtime=runtime) as uninterrupted:
            initial_g = deepcopy(uninterrupted.game.network.state_dict())
            initial_d = deepcopy(uninterrupted.game.critic.state_dict())
            assert uninterrupted.game.prior is None
            assert uninterrupted.game.recipe.lr.hex() == config.options["lr"].hex()
            assert uninterrupted.game.recipe.d_lr_mult.hex() == config.options["d_lr_mult"].hex()
            assert uninterrupted.game.response_domain == "continuous_packed"
            assert uninterrupted.game.signature["resolved_penalty_domain"] == "image"
            receipt = uninterrupted.game.backend.metadata()["fresh_initialization"]
            assert all(branch.graph["up"].weight.count_nonzero() == 0 for branch in uninterrupted.game.network.branches)
            uninterrupted.train()
            full = uninterrupted.checkpoint()
            raw = uninterrupted.export_inference(selection="raw")
            assert full["payload"]["game"]["completed_steps"] == 2
            assert raw["selection"] == "raw" and raw["origin"]["completed_steps"] == 2
            assert full["payload"]["ema"] == {"decay": None, "values": None}
            assert any(not torch.equal(initial_g[name], value) for name, value in uninterrupted.game.network.state_dict().items())
            assert any(not torch.equal(initial_d[name], value) for name, value in uninterrupted.game.critic.state_dict().items())
        with open_training(config, runtime=runtime) as split:
            split.train(stop_after=1)
            saved = split.checkpoint()
        assert len(calls) == 2
        with open_training(config, resume=saved, runtime=runtime) as resumed:
            assert len(calls) == 2  # restore invokes no fresh initializer
            assert resumed.game.backend.metadata()["fresh_initialization"] == receipt
            resumed.train()
            assert _hash(resumed.checkpoint()["payload"]) == _hash(full["payload"])
            assert resumed.export_inference(selection="raw")["sha256"] == raw["sha256"]
    assert not any(module._forward_hooks for module in runtime.transformer.modules())


def test_cache_tamper_refused_before_update_and_no_teacher(tmp_path):
    runtime = FixtureRuntime(tmp_path)
    config = configuration(tmp_path, runtime)
    with open_training(config, runtime=runtime) as session:
        with patch.object(PackedFluxResponse, "prepare", side_effect=AssertionError("no teacher replay")):
            session.step()
            row = Path(config.options["cache_dir"]) / "row-00000.pt"
            row.write_bytes(row.read_bytes() + b"tamper")
            with pytest.raises(ValueError, match="cache row bytes"):
                session.step()
            assert session.game.completed_steps == 1


def test_raw_actual_native_pipeline_pil_and_no_training_factory(tmp_path):
    import ps_flux_native.lifecycle as life
    import ps_flux_native.initialization as init
    import ps_flux_native.render as renderer
    from particle_sliders.inference import InferenceSession
    runtime = FixtureRuntime(tmp_path)
    config = configuration(tmp_path, runtime)
    raw = tmp_path / "RAW.pt"
    with open_training(config, runtime=runtime) as session:
        session.train()
        session.export_inference(raw, selection="raw")
    original_common_render = InferenceSession.render
    original_native_render = renderer.render_image
    bound_requests = []
    rejected_reference_request = []
    def common_render(self, **request):
        if not rejected_reference_request:
            # bool([]) is False, so only the actual product preflight can refuse
            # this supplied reference value before native render. No extra
            # successful native forward/render is introduced by the assertion.
            with pytest.raises(ValueError, match="reference images are unsupported"):
                original_common_render(self, images=[], **request)
            rejected_reference_request.append(True)
        return original_common_render(self, **request)
    def native_render(*args, **request):
        # The real core injects images=None on every render. Product validates
        # and consumes that common field instead of leaking it to the native
        # pipeline callback, whose signature deliberately has no image input.
        assert "images" not in request
        bound_requests.append(request["strength"])
        return original_native_render(*args, **request)
    with patch.object(life, "NativeRuntime", side_effect=lambda *args, **kwargs: FixtureRuntime(tmp_path)), \
            patch.object(life, "ConditionalParticleGAN", side_effect=AssertionError("no inference GAN/game")), \
            patch.object(life, "CachedResponses", side_effect=AssertionError("no inference teacher cache")), \
            patch.object(init, "initialize_fresh_components", side_effect=AssertionError("no inference initialization")), \
            patch.object(InferenceSession, "render", common_render), \
            patch.object(renderer, "render_image", native_render):
        receipt = render_pair(artifact=raw, model_dir=tmp_path,
            model_lock_sha256=config.options["model_lock_sha256"], output=tmp_path / "finished",
            prompt="a x", steps=1, seed=7, strength=.25, device="cpu", precision="float32")
    assert receipt["status"] == "complete" and len(receipt["images"]) == 2
    assert rejected_reference_request == [True] and bound_requests == [0., .25]
    assert receipt["training_updates"] == 0 and receipt["teacher_cache_used"] is False
    assert receipt["original_artifact_internal_sha256"]
    assert receipt["geometry"]["image_tokens"] == 16
    for image in receipt["images"]:
        assert (image["width"], image["height"]) == (64, 64)
        assert Path(image["output"]["path"]).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_scheduler_and_native_state_restore_on_render_failure(tmp_path):
    runtime = FixtureRuntime(tmp_path)
    pipe = runtime.pipe
    old_scheduler = pipe.scheduler
    before_times, before_sigmas = old_scheduler.timesteps.clone(), old_scheduler.sigmas.clone()
    before_config = deepcopy(dict(old_scheduler.config))
    assert "_current_timestep" not in vars(pipe)
    with pytest.raises(RuntimeError, match="injected"):
        with native_pipeline_state(pipe):
            pipe._current_timestep = 777
            pipe.scheduler.set_timesteps(2)
            raise RuntimeError("injected")
    assert pipe.scheduler is old_scheduler and "_current_timestep" not in vars(pipe)
    assert torch.equal(old_scheduler.timesteps, before_times) and torch.equal(old_scheduler.sigmas, before_sigmas)
    assert dict(old_scheduler.config) == before_config


def test_schedule_uses_original_native_pipeline_and_reports_its_sigma_grid(tmp_path):
    from particle_sliders.inference import open_inference
    import ps_flux_native.lifecycle as life
    from ps_flux_native.scheduled_render import render

    runtime = FixtureRuntime(tmp_path)
    config = configuration(tmp_path, runtime)
    with open_training(config, runtime=runtime) as training:
        raw = training.export_inference(selection="raw")
    reopened = FixtureRuntime(tmp_path)
    native = dict(prompt="a x", steps=2, strength=0.5, seed=7)
    original_scheduler = reopened.pipe.scheduler
    options = {key: config.options[key] for key in ("model_dir", "model_lock_sha256", "device", "base_precision")}
    with patch.object(life, "NativeRuntime", return_value=reopened):
        with open_inference(raw, backend=NATIVE_BACKEND, options=options,
                            native_request=dict(prompt="a x", steps=2)) as inference:
            default = inference.render(**native)
            constant = render(inference, **native, step_gains=[1.0, 1.0])
            assert constant.tobytes() == default.tobytes()
            timed = render(inference, **native, step_gains=[1.5, 0.5])
            assert timed.info["strength_schedule"]["effective_strengths"] == [0.75, 0.25]
            assert len(timed.info["strength_schedule"]["sigmas"]) == 3
            assert reopened.pipe.scheduler is original_scheduler
            assert not reopened.transformer._forward_pre_hooks and not reopened.transformer._forward_hooks


@pytest.fixture
def flux_steering_wheel():
    import os

    path = os.environ.get("FLUX_TEST_STEERING_WHEEL")
    if not path:
        pytest.skip("supply the verified core #269 wheel to exercise opt-in Flux steering")
    return Path(path)


def test_steering_wheel_rejects_changed_source(tmp_path):
    from zipfile import ZipFile

    from ps_flux_native.activation_gate import CORE_MODULE, LatentGatePlan

    forged = tmp_path / "unverified.whl"
    with ZipFile(forged, "w") as archive:
        archive.writestr(CORE_MODULE, "raise AssertionError('never execute changed wheel bytes')\n")
    with pytest.raises(ValueError, match="exact shared steering module"):
        LatentGatePlan(core_wheel=forged, method="random", parameter=0,
                       seed=0, calibration_sha256="a" * 64)


def test_actual_native_live_gate_replay_off_and_scope_restore(tmp_path, flux_steering_wheel):
    from particle_sliders.inference import open_inference
    import ps_flux_native.lifecycle as life
    from ps_flux_native.activation_gate import LatentGatePlan
    from ps_flux_native.scheduled_render import render

    runtime = FixtureRuntime(tmp_path)
    config = configuration(tmp_path, runtime)
    with open_training(config, runtime=runtime) as training:
        training.step()
        raw = training.export_inference(selection="raw")
    reopened = FixtureRuntime(tmp_path)
    original_scheduler = reopened.pipe.scheduler
    native = dict(prompt="a x", steps=4, strength=0.5, seed=7)
    options = {key: config.options[key] for key in ("model_dir", "model_lock_sha256", "device", "base_precision")}
    probe = torch.nn.functional.one_hot(torch.tensor(0), num_classes=64).float()
    settings = dict(core_wheel=flux_steering_wheel, calibration_sha256="1" * 64)
    off_gate = LatentGatePlan(**settings, method="random", parameter=0, seed=0)
    mixed_gate = LatentGatePlan(**settings, method="random", parameter=0.5, seed=0)

    with patch.object(life, "NativeRuntime", return_value=reopened):
        with open_inference(raw, backend=NATIVE_BACKEND, options=options,
                            native_request=dict(prompt="a x", steps=4)) as inference:
            observed = []

            def input_probe(_model, _args, kwargs):
                latent = kwargs["hidden_states"]
                observed.append(float(latent.mean(dim=(0, 1), dtype=torch.float32) @ probe))

            with reopened.transformer.register_forward_pre_hook(input_probe, with_kwargs=True):
                off = inference.render(**{**native, "strength": 0.0})
            off_scores = list(observed)
            assert len(set(off_scores)) > 1
            adaptive_off = render(inference, **native, gate_plan=off_gate)
            assert adaptive_off.tobytes() == off.tobytes()
            assert adaptive_off.info["strength_schedule"]["fired"] == 0

            mixed = render(inference, **native, gate_plan=mixed_gate)
            receipt = mixed.info["strength_schedule"]
            assert [step["interval"] for step in receipt["decisions"]] == list(range(4))
            assert [step["fired"] for step in receipt["decisions"]] == [False, False, True, True]
            assert receipt["effective_strengths"] == [0, 0, 0.5, 0.5]
            assert receipt["gate_rate"] == 0.5
            assert receipt["sigma_weighted_strength"] == pytest.approx(
                0.5 * (receipt["sigmas"][2] - receipt["sigmas"][-1]))
            assert render(inference, **native, gate_plan=mixed_gate).tobytes() == mixed.tobytes()
            replay = render(inference, **native, step_gains=[0, 0, 1, 1])
            assert replay.tobytes() == mixed.tobytes()

            observed.clear()
            threshold = (min(off_scores) + max(off_scores)) / 2
            cfp = LatentGatePlan(**settings, method="cfp", parameter=threshold, direction=probe)
            with reopened.transformer.register_forward_pre_hook(input_probe, with_kwargs=True):
                cfp_result = render(inference, **native, gate_plan=cfp)
            cfp_trace = cfp_result.info["strength_schedule"]["decisions"]
            assert [step["projection"] for step in cfp_trace] == observed
            assert [step["fired"] for step in cfp_trace] == [value <= threshold for value in observed]
            assert any(step["fired"] for step in cfp_trace)

            clabs = LatentGatePlan(**settings, method="clabs", parameter=0.5, seed=0, direction=probe,
                clabs=dict(mean=fmean(off_scores), std=pstdev(off_scores),
                           epsilon=torch.finfo(torch.float32).eps, setpoint=0.0,
                           proportional_gain=1.0, integral_gain=0.5, beta=0.5))
            clabs_result = render(inference, **native, gate_plan=clabs)
            clabs_trace = clabs_result.info["strength_schedule"]
            assert clabs_trace["eligible"] == 4
            assert all(0 <= step["probability"] <= 1 for step in clabs_trace["decisions"])
            replay_gains = [value / native["strength"] for value in clabs_trace["effective_strengths"]]
            assert render(inference, **native, step_gains=replay_gains).tobytes() == clabs_result.tobytes()

            def fail(_model, _args, _output):
                raise RuntimeError("synthetic native render failure")

            with reopened.transformer.register_forward_hook(fail):
                with pytest.raises(RuntimeError, match="synthetic native render failure"):
                    render(inference, **native, gate_plan=cfp)
            assert reopened.pipe.scheduler is original_scheduler
            assert inference.binding.network.strength == 0.0
            assert not reopened.transformer._forward_pre_hooks and not reopened.transformer._forward_hooks
            assert render(inference, **native, gate_plan=off_gate).tobytes() == off.tobytes()


@pytest.fixture
def flux_velocity_wheel():
    import os

    path = os.environ.get("FLUX_TEST_VELOCITY_WHEEL")
    if not path:
        pytest.skip("supply the published shared PromptSteering wheel to exercise Flux S-PSR")
    return Path(path)


def test_native_spsr_edits_euler_velocity_at_fixed_raw_and_matches_coefficient_dose(tmp_path, flux_velocity_wheel):
    from particle_sliders.inference import open_inference
    import ps_flux_native.lifecycle as life
    from ps_flux_native.scheduled_render import render
    from ps_flux_native.velocity_steering import VelocitySteeringPlan

    config = configuration(tmp_path, runtime := FixtureRuntime(tmp_path))
    with open_training(config, runtime=runtime) as training:
        training.step()
        raw = training.export_inference(selection="raw")
    reopened = FixtureRuntime(tmp_path)
    original_scheduler = reopened.pipe.scheduler
    native = dict(prompt="a x", steps=4, strength=0.5, seed=7)
    options = {key: config.options[key] for key in ("model_dir", "model_lock_sha256", "device", "base_precision")}
    direction = torch.linspace(-0.1, 0.1, 64)
    probe_weight = torch.zeros(64)
    probe_weight[0] = 1.0
    bias = torch.tensor([1.0])
    fit = tmp_path / "synthetic-velocity-fit.pt"
    fit_record = dict(weights="raw", ema_decay=None, native_raw_strength=native["strength"],
        geometry=raw["specification"]["configuration"]["geometry"], sigmas=[1., .75, .5, .25, 0.],
        raw_packet_sha256=raw["sha256"], model_identity_sha256=digest(reopened.identity),
        probe={"probe.weight": probe_weight[None], "probe.bias": bias, "direction.weight": direction[None]})
    torch.save(fit_record, fit)
    plan = VelocitySteeringPlan(core_wheel=flux_velocity_wheel, fit_path=fit)
    rng = torch.get_rng_state()
    plan._probe_for("cpu")
    assert torch.equal(torch.get_rng_state(), rng)

    original_step = FlowMatchEulerDiscreteScheduler.step
    unedited = []
    euler_inputs = []

    def native_output(_model, _args, output):
        unedited.append(output[0].clone())

    def euler_step(scheduler, model_output, *args, **kwargs):
        euler_inputs.append((inference.binding.network.strength, model_output.clone()))
        return original_step(scheduler, model_output, *args, **kwargs)

    def check_edit(image, constant=None):
        schedule = image.info["strength_schedule"]
        assert schedule["effective_strengths"] == [native["strength"]] * 4
        receipt = schedule["velocity_steering"]
        assert receipt["fit_sha256"] == sha256(fit.read_bytes()).hexdigest()
        assert receipt["ema_decay"] is None
        assert receipt["native_raw_strength"] == native["strength"]
        assert receipt["raw_packet_sha256"] == raw["sha256"]
        assert receipt["model_identity_sha256"] == digest(reopened.identity)
        assert receipt["geometry"] == fit_record["geometry"]
        widths = [start - end for start, end in pairwise(schedule["sigmas"])]
        coefficients, applied_coefficients, realized = [], [], []
        for index, (before, (scale, after), row) in enumerate(zip(unedited, euler_inputs, receipt["steps"], strict=True)):
            assert scale == row["raw_strength"] == native["strength"]
            assert row["interval"] == index
            pooled = before.mean(dim=(0, 1), dtype=torch.float32)
            coefficient = float((pooled @ probe_weight + bias).relu()) if constant is None else constant
            assert row["velocity_channel_mean"] == pooled.tolist()
            assert row["coefficient"] == pytest.approx(coefficient)
            assert row["applied_coefficient"] == torch.tensor(row["coefficient"], dtype=torch.float32).item()
            expected = (before + row["applied_coefficient"] * direction).to(before.dtype)
            assert torch.equal(after, expected)
            actual = float(torch.linalg.vector_norm(after - before, dim=-1).mean())
            assert row["realized_mean_edit_norm"] == pytest.approx(actual)
            coefficients.append(row["coefficient"])
            applied_coefficients.append(row["applied_coefficient"])
            realized.append(actual)
        assert len(coefficients) == 4
        assert receipt["coefficient_dose"] == fsum(w * c for w, c in zip(widths, coefficients, strict=True))
        assert receipt["fp32_coefficient_dose"] == fsum(w * c for w, c in zip(widths, applied_coefficients, strict=True))
        assert receipt["velocity_edit_dose"] == pytest.approx(receipt["coefficient_dose"] * float(direction.norm()))
        assert receipt["realized_velocity_edit_dose"] == fsum(w * a for w, a in zip(widths, realized, strict=True))
        unedited.clear()
        euler_inputs.clear()
        return receipt

    with patch.object(life, "NativeRuntime", return_value=reopened):
        with open_inference(raw, backend=NATIVE_BACKEND, options=options,
                            native_request=dict(prompt="a x", steps=4)) as inference:
            baseline = inference.render(**native)
            with (reopened.transformer.register_forward_hook(native_output),
                  patch.object(FlowMatchEulerDiscreteScheduler, "step", euler_step)):
                adaptive = render(inference, **native, velocity_plan=plan)
                adaptive_receipt = check_edit(adaptive)
                assert adaptive_receipt["method"] == "s-psr"
                assert len({row["coefficient"] for row in adaptive_receipt["steps"]}) > 1
                assert adaptive.tobytes() != baseline.tobytes()
                matched = adaptive_receipt["matched_constant_coefficient"]
                constant = render(inference, **native, velocity_plan=plan, constant_coefficient=matched)
                constant_receipt = check_edit(constant, matched)
                assert constant_receipt["method"] == "same-atom-constant"
                assert constant_receipt["coefficient_dose"] == adaptive_receipt["coefficient_dose"]
                zero = render(inference, **native, velocity_plan=plan, constant_coefficient=0)
                assert check_edit(zero, 0)["coefficient_dose"] == 0
                assert zero.tobytes() == baseline.tobytes()

            for changes, pattern in [({"strength": 0.4}, "fit's fixed native RAW strength"),
                                     ({"steps": 3}, "fit's native step count"),
                                     ({"step_gains": [1.] * 4}, "one Flux schedule"),
                                     ({"constant_coefficient": float("nan")}, "nonnegative same-atom")]:
                with pytest.raises(ValueError, match=pattern):
                    render(inference, **{**native, **changes}, velocity_plan=plan)
            with pytest.raises(ValueError, match="Supply the native RAW strength"):
                render(inference, prompt="a x", velocity_plan=plan)
            incompatible_fit = tmp_path / "incompatible.pt"
            for changes, pattern in [({"raw_packet_sha256": "a" * 64}, "exact RAW packet"),
                                     ({"model_identity_sha256": "b" * 64}, "exact RAW packet"),
                                     ({"geometry": geometry({"height": 80, "width": 64})}, "original native geometry")]:
                torch.save({**fit_record, **changes}, incompatible_fit)
                wrong_plan = VelocitySteeringPlan(core_wheel=flux_velocity_wheel, fit_path=incompatible_fit)
                with pytest.raises(ValueError, match=pattern):
                    render(inference, **native, velocity_plan=wrong_plan)
            torch.save(dict(fit_record, sigmas=[1., .875, .5, .25, 0.]), incompatible_fit)
            wrong_grid = VelocitySteeringPlan(core_wheel=flux_velocity_wheel, fit_path=incompatible_fit)
            with pytest.raises(RuntimeError, match="Euler sigmas differ"):
                render(inference, **native, velocity_plan=wrong_grid)

            with patch.object(FlowMatchEulerDiscreteScheduler, "step", side_effect=RuntimeError("Euler failed")):
                with pytest.raises(RuntimeError, match="Euler failed"):
                    render(inference, **native, velocity_plan=plan)
            assert reopened.pipe.scheduler is original_scheduler
            assert inference.binding.network.strength == 0.0
            assert not reopened.transformer._forward_pre_hooks and not reopened.transformer._forward_hooks
            assert render(inference, **native, velocity_plan=plan).tobytes() == adaptive.tobytes()


def test_spsr_requires_no_ema_and_verified_equations(tmp_path, flux_velocity_wheel):
    from zipfile import ZipFile

    from ps_flux_native.velocity_steering import CORE_MODULE, VelocitySteeringPlan

    fit_path = tmp_path / "probe.pt"
    probe = {"probe.weight": torch.zeros(1, 64), "probe.bias": torch.zeros(1), "direction.weight": torch.ones(1, 64)}
    fit = dict(weights="raw", ema_decay=0.99, native_raw_strength=0.5, probe=probe,
        geometry=geometry({"height": 64, "width": 64}), sigmas=[1., 0.],
        raw_packet_sha256="a" * 64, model_identity_sha256="b" * 64)
    torch.save(fit, fit_path)
    with pytest.raises(ValueError, match="EMA disabled"):
        VelocitySteeringPlan(core_wheel=flux_velocity_wheel, fit_path=fit_path)

    torch.save(dict(fit, ema_decay=None), fit_path)
    forged = tmp_path / "unverified.whl"
    with ZipFile(forged, "w") as archive:
        archive.writestr(CORE_MODULE, "raise AssertionError('never execute changed wheel bytes')\n")
    with pytest.raises(ValueError, match="exact shared prompt-steering module"):
        VelocitySteeringPlan(core_wheel=forged, fit_path=fit_path)


def test_native_velocity_hook_with_owner_callbacks_and_bf16_exposure(flux_velocity_wheel):
    from ps_flux_native.velocity_steering import _render_native_velocity, _shared_prompt_steering

    class PackedVelocity(torch.nn.Module):
        def forward(self, velocity):
            return (velocity,)

    transformer = PackedVelocity()
    probe = _shared_prompt_steering(flux_velocity_wheel)(2).requires_grad_(False)
    probe.direction.weight.copy_(torch.tensor([[0.125, 0.375]]))
    sigmas = [1.0, 0.5, 0.0]
    scheduler = FlowMatchEulerDiscreteScheduler()
    scheduler.set_timesteps(sigmas=sigmas[:-1])
    scheduler.set_begin_index(0)
    original = torch.full((1, 3, 2), 0.5, dtype=torch.bfloat16)
    strength = 2 / 3
    observed = []

    def owner_render():
        latent = torch.zeros_like(original)
        for timestep in scheduler.timesteps:
            edited, = transformer(original)
            observed.append(edited)
            latent, = scheduler.step(edited, timestep, latent, return_dict=False)
        return latent

    _, receipt = _render_native_velocity(owner_render, transformer=transformer,
        current_scheduler=lambda: scheduler, current_strength=lambda: strength, strength=strength,
        sigmas=sigmas, shape=original.shape, probe_for=lambda _device: probe, constant_coefficient=0.1)
    applied = torch.tensor(0.1, dtype=torch.float32)
    edit = (original.float() + applied * probe.direction.weight[0]).to(torch.bfloat16)
    assert len(observed) == 2 and all(torch.equal(value, edit) for value in observed)
    assert receipt["coefficient_dose"] == 0.1
    assert receipt["fp32_coefficient_dose"] == float(applied) != receipt["coefficient_dose"]
    assert receipt["realized_velocity_edit_dose"] == float(torch.linalg.vector_norm(edit.float() - original.float(), dim=-1).mean())
    assert receipt["realized_velocity_edit_dose"] != receipt["velocity_edit_dose"]
    assert not transformer._forward_hooks
