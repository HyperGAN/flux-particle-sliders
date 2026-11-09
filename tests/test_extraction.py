"""Standalone installation/registration and unsupported legacy-container checks."""
import importlib.util
import hashlib
import json
from importlib.metadata import distribution, entry_points
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from safetensors.torch import save_file

from particle_sliders.lifecycle import CheckpointError, SessionConfig
from particle_sliders.response import open_response
from ps_flux_packed_preview import COHORT
from ps_flux_packed_preview.native import FluxLoRA

from test_native import condition, model, response


def test_standalone_installation_and_no_training_inference_or_cli_registration():
    import ps_flux_packed_preview
    assert "site-packages" in ps_flux_packed_preview.__file__
    assert importlib.util.find_spec("conceptmod") is None
    assert not distribution("particle-sliders-flux-packed-preview").entry_points
    for group in ("particle_sliders.response_backends", "particle_sliders.inference_backends"):
        assert not any(ep.value.startswith("ps_flux_packed_preview") for ep in entry_points(group=group))


def test_installed_payload_relocated_locks_and_actual_core_origin_match_active_cohort():
    import ps_flux_packed_preview
    from ps_flux_packed_preview.provenance import checked_builder
    manifest = json.loads((Path(__file__).parents[1] / "EXTRACTION-PROVENANCE.json").read_text())
    installed = Path(ps_flux_packed_preview.__file__).parent
    runtime = [row for row in manifest["mappings"] if row["destination_path"].startswith("src/")]
    assert len(runtime) == 5
    for row in runtime:
        assert hashlib.sha256((installed / Path(row["destination_path"]).name).read_bytes()).hexdigest() == row["destination_sha256"]
        if row["destination_path"].endswith(("/native.py", "/lora.hndl")):
            assert row["action"] == "copied_byte_for_byte"
            assert row["destination_sha256"] == row["source_sha256"]
    _, identity = checked_builder()
    assert identity["flux_core_commit"] == manifest["active_core_pin"]
    assert identity["flux_cohort"] == COHORT == manifest["active_cohort"]
    assert distribution("particle-sliders-core").version == "0.12.0"
    origin = json.loads(distribution("particle-sliders-core").read_text("direct_url.json"))
    assert origin["vcs_info"]["commit_id"] == manifest["active_core_pin"]
    assert origin["url"] == "https://github.com/HyperGAN/particle-sliders.git"
    assert origin["subdirectory"] == "packages/particle-sliders-core"
    assert distribution("particle-sliders-flux-packed-preview").version == "0.2.0"


@pytest.mark.parametrize("kind", ["resume", "artifact"])
def test_synthetic_legacy_lora_container_refuses_before_backend_resolution_or_model_mutation(tmp_path, kind):
    # Historical-format fixture only, not an actual trained checkpoint. This
    # product has no session/EMA/resume/export API to qualify or migrate.
    native = model()
    before = {name: value.clone() for name, value in native.state_dict().items()}
    modes = [m.training for m in native.modules()]
    flags = [p.requires_grad for p in native.parameters()]
    rng = torch.get_rng_state().clone()
    legacy = tmp_path / "legacy-fixture.safetensors"
    save_file({"lora_unet_transformer_blocks_0_attn_to_q.lora_down.weight": torch.zeros(4, 16)}, legacy)
    config = SessionConfig("flux-packed-preview", COHORT)
    with patch("particle_sliders.response.response_backend", side_effect=AssertionError("backend resolution ran")):
        with pytest.raises(CheckpointError):
            open_response(config, **{kind: legacy}, model=native)
    assert native.state_dict().keys() == before.keys()
    assert all(torch.equal(value, before[name]) for name, value in native.state_dict().items())
    assert [m.training for m in native.modules()] == modes
    assert [p.requires_grad for p in native.parameters()] == flags
    assert torch.equal(torch.get_rng_state(), rng)
    assert all(not m._forward_hooks and not m._forward_pre_hooks for m in native.modules())


def test_off_and_raw_adapter_weights_roundtrip_without_an_umbrella_checkout():
    # Raw nn.Module state only, not a product checkpoint or shared export.
    native = model()
    bank = FluxLoRA(native, image_tokens=4, text_tokens=3)
    with torch.no_grad():
        bank.branches[0].graph["up"].weight.fill_(.01)
    raw = {name: value.clone() for name, value in bank.state_dict().items()}
    other = FluxLoRA(native, image_tokens=4, text_tokens=3)
    other.load_state_dict(raw)
    assert all(torch.equal(value, other.state_dict()[name]) for name, value in raw.items())
    c, latent, time = condition(), torch.randn(1, 4, 64), torch.tensor(500.)
    base = native(hidden_states=latent, timestep=time.reshape(1) / 1000,
        encoder_hidden_states=c["embeds"], pooled_projections=c["pooled"],
        txt_ids=c["text_ids"], img_ids=c["image_ids"], guidance=None, return_dict=False)[0]
    with bank.borrowed():
        adapter = response(native, bank)
        off = adapter.velocity(c, latent, time, scale=0.)
        expected_raw = adapter.velocity(c, latent, time, scale=1.)
    with other.borrowed():
        actual_raw = response(native, other).velocity(c, latent, time, scale=1.)
    torch.testing.assert_close(off, base, rtol=0, atol=0)
    torch.testing.assert_close(actual_raw, expected_raw, rtol=0, atol=0)
    assert not torch.equal(expected_raw, base)
    assert not bank.attached and not other.attached
