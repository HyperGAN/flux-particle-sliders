"""Authored native CPU qualification, not an assertion that it has run.

Uses real tiny Diffusers Flux/VAE + Transformers CLIP/T5, local toy tokenizers,
and the exact external HNDL builder. No pretrained weights/network/download.
The tiny loader fixture is labeled synthetic and never claims a model lock.
"""
from copy import deepcopy
import json
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from diffusers import (AutoencoderKL, FluxPipeline, FluxTransformer2DModel,
                       FlowMatchEulerDiscreteScheduler)
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer, T5Config, T5EncoderModel, T5TokenizerFast
from tokenizers import Tokenizer, models, pre_tokenizers
from particle_sliders.lifecycle import SessionConfig
from particle_sliders.pg09_backends import ResponseRow
from particle_sliders.pg09 import _hash

from ps_flux_native95.adapter import FluxLoRA
from ps_flux_native95.cache import CONDITION_POLICY, CachedResponses, write_row
from ps_flux_native95.contracts import INITIALIZATION_POLICY, PROTOCOL, file_hash, geometry
from ps_flux_native95.native import NativeRuntime, PackedFluxResponse
from ps_flux_native95.lifecycle import open_training, NATIVE_BACKEND
from ps_flux_native95.render import native_pipeline_state, render_pair


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
    return SessionConfig("flux_native_packed95", "tiny-native-packed-CPU-only",
        model_dir=str(root), model_lock_sha256=file_hash(lock), cache_dir=str(cache),
        cache_index_sha256=file_hash(cache / "index.json"), device="cpu", base_precision="float32",
        height=64, width=64, text_tokens=3, steps=2, batch=2, seed=19, critic_width=4,
        lr=.0004, d_lr_mult=.5000000000000001, prior_lr_mult=None, ema_decay=None,
        pg_version="0.9.5", pg_source_profile="release-0.9.5", penalty_input_domain="auto",
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
    import ps_flux_native95.initialization as init
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
    import ps_flux_native95.lifecycle as life
    import ps_flux_native95.initialization as init
    import ps_flux_native95.render as renderer
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
            patch.object(life, "ConditionalPG095", side_effect=AssertionError("no inference GAN/game")), \
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
