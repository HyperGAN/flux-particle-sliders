"""Ordinary HNDL down/up LoRA, derived from the published packed prerequisite.

Original source: HyperGAN/flux-particle-sliders 11666361145d175e04ad571ae07100de84a98904
src/ps_flux_packed_preview/native.py (MIT). Graph, target selection, alpha/rank,
zero-up, native token axes and reversible hooks are retained. The new cohort
explicitly admits a colocated FP32/BF16 frozen base with FP32 adapter leaves.
No routed bank, projection of responses or gradient checkpointing is added.
"""
from contextlib import contextmanager
from copy import deepcopy
import json
import math
import torch
from torch import nn
from .provenance import ROOT, check_native
from .builder import checked_builder

class HNDLLoRA(nn.Module):
    """Exactly the ordinary bias-free down/up branch used by LoRAModule.

    The host retains its own forward; a removable public forward hook adds the
    branch. B and the fixed native token dimension remain separate graph axes.
    """
    def __init__(self, host, rank, alpha, builder, identity, *, tokens, _rng=None):
        super().__init__()
        self.tokens = tokens
        with torch.random.fork_rng(devices=[]):
            if _rng is not None:
                torch.set_rng_state(_rng.get_state())
            try:
                self.graph = builder((ROOT / "lora.hndl").read_text(),
                    input_shape={"value": ["B", tokens, host.in_features]},
                    output_shape=["B", tokens, host.out_features],
                    parameters={"rank": rank, "output_width": host.out_features})
                nn.init.kaiming_uniform_(self.graph["down"].weight, a=1)
                nn.init.zeros_(self.graph["up"].weight)
            finally:
                if _rng is not None:
                    _rng.set_state(torch.get_rng_state())
        self.scale = alpha / rank
        self._identity = deepcopy(identity)

    def extra_repr(self):
        return json.dumps(dict(self._identity, alpha_over_rank=self.scale, native_tokens=self.tokens), sort_keys=True)

    def forward(self, value):
        if value.ndim != 3 or value.shape[1] != self.tokens:
            raise ValueError("Flux attention projections require native [B,T,D]")
        return self.graph(value=value.to(self.graph["down"].weight)) * self.scale


class FluxLoRA(nn.Module):
    """Borrow an already staged frozen native model and restore every hook/mode.

    Only historical full/Attention linear targets are supported: add_* text
    projections are excluded exactly as in conceptmod's LoRANetwork selector.
    No model parameters are registered in the adapter network.
    """
    def __init__(self, model, *, image_tokens, text_tokens, rank=4, alpha=1.):
        super().__init__()
        check_native(model)
        if any(type(v) is not int or v < 1 for v in (image_tokens, text_tokens)):
            raise ValueError("fixed positive native image/text token counts required")
        self.image_tokens, self.text_tokens = image_tokens, text_tokens
        if type(rank) is not int or rank < 1 or not math.isfinite(alpha) or alpha <= 0:
            raise ValueError("positive rank/alpha required")
        if model.config.in_channels != 64 or model.config.out_channels not in (None, 64):
            raise ValueError("Flux packed64 input/output config required")
        if getattr(model, "gradient_checkpointing", False):
            raise ValueError("checkpointed recomputation needs reviewed scale capture; unsupported")
        parameters = tuple(model.parameters())
        if (not parameters or any(p.requires_grad or p.dtype not in (torch.float32, torch.bfloat16)
                                 or p.device != parameters[0].device or p.dtype != parameters[0].dtype
                                 for p in parameters)):
            raise ValueError("one frozen colocated FP32/BF16 native model required")
        builder, identity = checked_builder()
        targets, seen = [], set()
        for name, module in model.named_modules():
            if module.__class__.__name__ == "FluxAttention":
                for child_name, child in module.named_modules():
                    if isinstance(child, nn.Linear) and "add_" not in child_name and id(child) not in seen:
                        seen.add(id(child))
                        targets.append((name + "." + child_name, child))
        if not targets:
            raise ValueError("no historical full Attention linear targets")
        self.target_names = tuple(name for name, _ in targets)
        # Keep the caller's CPU RNG unchanged on successful construction and
        # graph/initialization failure. The private fork still advances between
        # branches, preserving distinct sequential native initializations.
        with torch.random.fork_rng(devices=[]):
            branch_rng = torch.Generator().set_state(torch.get_rng_state())
            self.branches = nn.ModuleList([HNDLLoRA(host, rank, alpha, builder, identity,
                tokens=image_tokens + text_tokens if name.startswith("single_transformer_blocks.") else image_tokens,
                _rng=branch_rng)
                for name, host in targets])
        self.names = tuple(name for name, _ in self.named_parameters())
        object.__setattr__(self, "_model", model)
        self._targets = targets
        self._model_config = deepcopy(dict(model.config))
        self._base_ids = tuple(id(p) for p in model.parameters())
        self._base_device, self._base_dtype = parameters[0].device, parameters[0].dtype
        self._handles = []
        self._modes = None
        self.strength = 0.

    @property
    def attached(self):
        return bool(self._handles)

    def attach(self):
        if self.attached:
            raise RuntimeError("Flux hooks already attached")
        check_native(self._model)
        checked_builder()
        if (dict(self._model.config) != self._model_config
                or tuple(id(p) for p in self._model.parameters()) != self._base_ids
                or any(p.requires_grad or p.dtype != self._base_dtype or p.device != self._base_device
                       for p in self._model.parameters())
                or getattr(self._model, "gradient_checkpointing", False)):
            raise ValueError("staged Flux model changed before attachment")
        if "forward" in self._model.__dict__ or any(host._forward_hooks or host._forward_pre_hooks or "forward" in host.__dict__
               for _, host in self._targets):
            raise ValueError("Flux target already carries hooks or a replaced forward")
        self._modes = [(m, m.training) for m in self._model.modules()]
        try:
            self._model.eval()
            for (_, host), branch in zip(self._targets, self.branches):
                def hook(module, args, output, branch=branch):
                    if self.strength == 0:
                        return output
                    return output + (branch(args[0]) * self.strength).to(output)
                self._handles.append(host.register_forward_hook(hook))
        except BaseException as original:
            try:
                self.close()
            except BaseException as cleanup:
                original.add_note("Flux cleanup also failed: " + repr(cleanup))
            raise
        return self

    def close(self):
        errors, remaining = [], []
        for handle in reversed(self._handles):
            try:
                handle.remove()
            except BaseException as exc:
                errors.append(exc)
                remaining.append(handle)
        self._handles = remaining
        if self._modes is not None:
            for module, training in self._modes:
                try:
                    module.training = training
                except BaseException as exc:
                    errors.append(exc)
            self._modes = None
        self.strength = 0.
        if errors:
            raise RuntimeError("Flux cleanup attempted every hook and mode; failures occurred") from errors[0]

    @contextmanager
    def borrowed(self):
        self.attach()
        try:
            yield self
        except BaseException as original:
            try:
                self.close()
            except BaseException as cleanup:
                original.add_note("Flux cleanup also failed: " + repr(cleanup))
            raise
        else:
            self.close()

    @contextmanager
    def scaled(self, scale):
        if not self.attached or not math.isfinite(scale):
            raise ValueError("attached ordinary adapter and finite strength required")
        previous = self.strength
        self.strength = float(scale)
        try:
            yield self
        finally:
            self.strength = previous
