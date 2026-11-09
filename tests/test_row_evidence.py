import importlib.util
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from particle_sliders.pg09_backends import ResponseRow


path = Path(__file__).parents[1] / "src/ps_flux_packed_preview/row_evidence.py"
spec = importlib.util.spec_from_file_location("flux_row_evidence_test", path)
module = importlib.util.module_from_spec(spec)
import sys
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def make_row():
    def condition():
        return dict(embeds=torch.ones(1, 3, 12), pooled=torch.ones(1, 8),
                    text_ids=torch.zeros(3, 3), image_ids=torch.zeros(4, 3), guidance=None)
    return ResponseRow(condition(), condition(), condition(), torch.zeros(1, 4, 64), timestep=torch.tensor(500.))


def bind(row, **kwargs):
    options = dict(image_tokens=4, text_tokens=3, text_width=12, pooled_width=8,
                   guidance_embeds=False, asserted_model_sha256="a" * 64)
    return module.snapshot_row(row, **(options | kwargs))


def test_snapshot_preserves_native_axes_and_is_independent_of_input_mutation():
    row = make_row()
    first = bind(row)
    assert first.row.latent.shape == (1, 4, 64)
    assert first.identity["model_sha256_asserted_not_verified"] == "a" * 64
    assert bind(row).fingerprint == first.fingerprint
    row.neutral["embeds"].zero_()
    assert torch.equal(first.row.neutral["embeds"], torch.ones(1, 3, 12))
    assert bind(row).fingerprint != first.fingerprint
    assert bind(make_row(), asserted_model_sha256="b" * 64).fingerprint != first.fingerprint


def test_guidance_model_requires_native_guidance_tensor_for_every_branch():
    row = make_row()
    with pytest.raises(ValueError, match="Guidance presence"):
        bind(row, guidance_embeds=True)
    for condition in (row.neutral, row.positive, row.negative):
        condition["guidance"] = torch.tensor([3.5])
    evidence = bind(row, guidance_embeds=True)
    assert evidence.row.positive["guidance"].shape == (1,)
    assert evidence.identity["tensors"]["positive.guidance"]["shape"] == [1]


def test_verify_snapshot_detects_mutation_without_changing_rng_or_input():
    evidence = bind(make_row())
    rng = torch.random.get_rng_state().clone()
    original = evidence.row.positive["embeds"].clone()
    module.verify_snapshot(evidence)
    assert torch.equal(rng, torch.random.get_rng_state())
    assert torch.equal(original, evidence.row.positive["embeds"])
    evidence.row.positive["embeds"][0, 0, 0] = 2
    with pytest.raises(ValueError, match="modified row snapshot"):
        module.verify_snapshot(evidence)


@pytest.mark.parametrize("mutation", ["identity", "fingerprint", "shape", "guidance"])
def test_verify_snapshot_rejects_modified_identity_and_contract(mutation):
    evidence = bind(make_row())
    if mutation == "identity":
        evidence.identity["model_sha256_asserted_not_verified"] = "b" * 64
    elif mutation == "fingerprint":
        evidence = replace(evidence, fingerprint="0" * 64)
    elif mutation == "shape":
        evidence.row.neutral["image_ids"] = torch.zeros(1, 4, 3)
    else:
        evidence.row.negative["guidance"] = torch.ones(1)
    with pytest.raises(ValueError, match="modified row snapshot"):
        module.verify_snapshot(evidence)


@pytest.mark.parametrize("change", ["shape", "guidance", "dtype", "gradient", "timestep", "scale", "digest"])
def test_invalid_native_contract_is_rejected(change):
    row = make_row()
    options = {}
    if change == "shape": row.neutral["image_ids"] = torch.zeros(1, 4, 3)
    elif change == "guidance": row.positive["guidance"] = "not a tensor"
    elif change == "dtype": row = replace(row, latent=row.latent.double())
    elif change == "gradient": row.latent.requires_grad_()
    elif change == "timestep": row = replace(row, timestep=torch.tensor(1001.))
    elif change == "scale": row = replace(row, scale=0.5)
    else: options["asserted_model_sha256"] = "not-a-digest"
    with pytest.raises(ValueError):
        bind(row, **options)
