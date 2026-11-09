"""Tiny metadata fixtures only; no model weights or GPU work are executed."""
import importlib.util
import json
from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.fixture
def readiness():
    path = Path(__file__).parents[1] / "scripts/preflight_flux_readiness.py"
    spec = importlib.util.spec_from_file_location("flux_readiness_metadata", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def snapshot(tmp_path):
    root = tmp_path / ("a" * 40)
    documents = {
        "model_index.json": {"_class_name": "FluxPipeline"},
        "transformer/config.json": {"_class_name": "FluxTransformer2DModel", "in_channels": 64,
            "num_layers": 19, "num_single_layers": 38, "num_attention_heads": 24, "attention_head_dim": 128},
        "text_encoder/config.json": {}, "text_encoder_2/config.json": {},
        "vae/config.json": {}, "scheduler/scheduler_config.json": {},
    }
    for relative, value in documents.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
    for name in ["transformer", "text_encoder", "text_encoder_2", "vae"]:
        # Dummy bytes simulate stat metadata only; these are not tensor files.
        (root / name / "model.safetensors").write_bytes(b"metadata-fixture")
    for relative in ["tokenizer/tokenizer_config.json", "tokenizer/vocab.json", "tokenizer/merges.txt",
                     "tokenizer_2/tokenizer_config.json", "tokenizer_2/tokenizer.json"]:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("metadata-fixture")
    return root


def test_complete_stat_metadata_never_enables_readiness(readiness, snapshot):
    report = readiness.readiness_report(snapshot=snapshot, model_id="fixture/Flux", revision="a" * 40,
        image_tokens=1024, text_tokens=256, admission_probe=lambda count: {"status": "unexpected_admission"})
    assert report["cache"]["metadata_files_present"]
    assert report["ready"] is False and report["exit_code"] == 2
    assert len(report["blockers"]) == 4
    estimate = report["cache"]["adapter_resource_estimate"]
    assert estimate["estimate_only"] and estimate["adapter_parameters"] == 4669440


def test_weight_payloads_are_never_opened(readiness, snapshot):
    original = Path.open
    def guarded(path, *args, **kwargs):
        if path.suffix in (".safetensors", ".bin"):
            raise AssertionError("A weight payload was opened")
        return original(path, *args, **kwargs)
    with patch.object(Path, "open", guarded):
        result = readiness.inspect_snapshot(snapshot, model_id="fixture/Flux", revision="a" * 40)
    assert not result["missing"]
    assert result["weight_integrity"].startswith("Not verified")


def test_missing_transformer_payload_is_an_exact_blocker(readiness, snapshot):
    (snapshot / "transformer/model.safetensors").unlink()
    report = readiness.inspect_snapshot(snapshot, model_id="fixture/Flux", revision="a" * 40)
    assert "No weight file metadata is present for transformer." in report["missing"]


def test_missing_indexed_shard_and_traversal_are_refused(readiness, snapshot):
    index = snapshot / "transformer/model.safetensors.index.json"
    index.write_text(json.dumps({"weight_map": {"w": "missing.safetensors"}}))
    report = readiness.inspect_snapshot(snapshot, model_id="fixture/Flux", revision="a" * 40)
    assert any("Missing indexed shard" in value for value in report["missing"])
    index.write_text(json.dumps({"weight_map": {"w": "../outside.safetensors"}}))
    report = readiness.inspect_snapshot(snapshot, model_id="fixture/Flux", revision="a" * 40)
    assert any("local safetensors basename" in value for value in report["missing"])


def test_metadata_size_and_revision_checks(readiness, snapshot):
    path = snapshot / "model_index.json"
    path.write_bytes(b" " * (readiness.MAX_METADATA_BYTES + 1))
    report = readiness.inspect_snapshot(snapshot, model_id="fixture/Flux", revision="b" * 40)
    assert any("one MiB" in value for value in report["missing"])
    assert any("revision does not match" in value for value in report["missing"])


def test_absent_snapshot_and_nonpositive_shapes_are_not_ready(readiness, tmp_path):
    report = readiness.readiness_report(snapshot=tmp_path / "absent", image_tokens=1, text_tokens=1,
                                      admission_probe=lambda _: {"status": "blocked"})
    assert report["ready"] is False
    assert report["cache"]["metadata_files_present"] is False
    assert "snapshot directory is absent" in report["cache"]["missing"][0]
    with pytest.raises(ValueError, match="Positive"):
        readiness.readiness_report(image_tokens=0, text_tokens=256)


def test_metadata_access_failure_still_returns_a_blocking_report(readiness):
    with patch.object(readiness, "inspect_snapshot", side_effect=PermissionError("Fixture metadata is unavailable")):
        report = readiness.readiness_report(image_tokens=1024, text_tokens=256,
            admission_probe=lambda _: {"status": "blocked"})
    assert report["ready"] is False and report["exit_code"] == 2
    assert report["cache"]["metadata_files_present"] is False
    assert "PermissionError" in report["cache"]["missing"][0]


def test_cli_always_returns_two_without_a_training_path(readiness, snapshot, capsys):
    with patch.object(readiness, "probe_admission", return_value={"status": "blocked"}):
        code = readiness.main(["--snapshot", str(snapshot), "--model-id", "fixture/Flux",
                               "--revision", "a" * 40, "--image-tokens", "1024", "--text-tokens", "256"])
    report = json.loads(capsys.readouterr().out)
    assert code == 2 and report["ready"] is False


def test_actual_installed_admission_probe_constructs_no_model_or_cuda(readiness):
    import torch
    from diffusers import FluxTransformer2DModel
    from ps_flux_packed_preview.provenance import checked_builder
    checked_builder()
    rng = torch.get_rng_state().clone()
    with patch.object(FluxTransformer2DModel, "__init__", side_effect=AssertionError("Model construction ran")), \
         patch.object(torch.cuda, "_lazy_init", side_effect=AssertionError("CUDA initialization ran")):
        proof = readiness.probe_admission(1024)
    assert proof["status"] == "blocked"
    assert "continuous_field requires genuine native 4D/5D shape" in proof["cause"]
    assert proof["model_loaded"] is False
    assert torch.equal(rng, torch.get_rng_state())
