"""Actual HNDL ordinary leaves + original PG initializer; authored UNRUN.

No pretrained transformer/cache or model downloads; Root owns CPU execution.
"""
from types import SimpleNamespace
import math

import pytest
import torch
from torch import nn
from hndl.operators.linear import Linear as HNDLLinear
from particlegan.init import KEEP, Uniform
from ps_flux_native95.contracts import digest
from ps_flux_native95.initialization import (Construction, declared_hndl_weight,
                                           initialize_owned, policies)


def linear(ins, outs, **kwargs):
    return HNDLLinear(ins, outs, bias=kwargs.get("bias", False),
                      spectral_norm=kwargs.get("spectral_norm", False),
                      equalized=kwargs.get("equalized", False))


def owned():
    adapter=nn.Module();branch=nn.Module()
    branch.graph=nn.ModuleDict(dict(down=linear(8,4),up=linear(4,8)))
    adapter.branches=nn.ModuleList([branch])
    with torch.no_grad():
        branch.graph["up"].weight.zero_()
    return adapter,nn.Linear(8,1)


def test_actual_hndl_holder_preserves_parameters_law_and_fresh_rng():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(113);adapter,critic=owned()
    down,up=adapter.branches[0].graph["down"],adapter.branches[0].graph["up"]
    identities={id(p)for root in(adapter,critic)for p in root.parameters()}
    before=torch.get_rng_state().clone();up_before=up.weight.detach().clone()
    scopes=policies(adapter,critic,7)
    d,u=scopes["adapter.branch0.down"],scopes["adapter.branch0.up"]
    assert type(d.module)is nn.Module and type(u.module)is nn.Module
    assert d.module.weight is down.weight and u.module.weight is up.weight
    law=d.distributions["weight"]
    assert isinstance(law,Uniform)and(law.low,law.high)==(-math.sqrt(3/8),math.sqrt(3/8))
    assert u.distributions["weight"]is KEEP
    assert identities=={id(p)for scope in scopes.values()for p in scope.module.parameters()}
    assert torch.equal(before,torch.get_rng_state())
    options=dict(seed=7)
    receipt=initialize_owned(adapter,critic,Construction("fresh","actual-HNDL-CPU",digest(options)),options)
    assert identities=={id(p)for root in(adapter,critic)for p in root.parameters()}
    assert torch.equal(up.weight,up_before)and torch.equal(before,torch.get_rng_state())
    assert receipt.as_dict()["components"]["adapter.branch0.down"]["module_type"]=="Module"
    assert receipt.as_dict()["components"]["adapter.branch0.down"]["parameters"]["weight"]["host_declaration"]==dict(kind="Uniform",low=-math.sqrt(3/8),high=math.sqrt(3/8))
    x=torch.arange(16,dtype=down.weight.dtype).reshape(2,8)
    assert torch.equal(down(x),nn.functional.linear(x,down.weight,None))
    # A new temporary holder cannot hide the marker on the actual Parameter.
    with pytest.raises(ValueError,match="already initialized"):
        initialize_owned(adapter,critic,Construction("fresh","actual-HNDL-CPU",digest(options)),options)


@pytest.mark.parametrize("kind",["plain_torch","equalized","bias","child","spectral","subclass","buffer"])
def test_hndl_admission_refuses_other_forward_or_parameter_contracts(kind):
    if kind=="plain_torch":leaf=nn.Linear(8,4,bias=False)
    elif kind=="equalized":leaf=linear(8,4,equalized=True)
    elif kind=="bias":leaf=linear(8,4,bias=True)
    elif kind=="spectral":leaf=linear(8,4,spectral_norm=True)
    elif kind=="subclass":
        class Other(HNDLLinear):pass
        leaf=Other(8,4,bias=False,spectral_norm=False,equalized=False)
    else:
        leaf=linear(8,4)
        if kind=="child":leaf.add_module("unreviewed",nn.Identity())
        elif kind=="buffer":leaf.register_buffer("unreviewed",torch.ones(1))
    with pytest.raises(ValueError,match="exact ordinary"):
        declared_hndl_weight(leaf)


def test_original_receipt_restores_without_reinitializing_actual_hndl():
    from unittest.mock import patch
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(17);adapter,critic=owned()
    options=dict(seed=7);config=Construction("fresh","actual-HNDL-resume",digest(options))
    original=initialize_owned(adapter,critic,config,options)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(23);restored_adapter,restored_critic=owned()
    before=[p.detach().clone()for root in(restored_adapter,restored_critic)for p in root.parameters()]
    with patch("ps_flux_native95.initialization.initialize_fresh_components",side_effect=AssertionError("no initializer on restore")):
        resumed=initialize_owned(restored_adapter,restored_critic,
            Construction("resume",config.cohort,config.options_sha256,original.canonical_json),options)
    assert resumed.canonical_json==original.canonical_json
    assert all(torch.equal(p,value)for p,value in zip((p for root in(restored_adapter,restored_critic)for p in root.parameters()),before))
