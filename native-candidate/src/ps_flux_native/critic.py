"""Packed native critic: every response coordinate, original rank3 penalty."""
import torch
from torch import nn


class PackedCritic(nn.Module):
    """Position-aware per-token logits with a learned conditional readout.

    Only learned critic features/logits are aggregated. Neither the real/fake
    response nor the ParticleGAN penalty input is flattened, pooled or resized.
    The condition vector contains every original context coordinate, not a sketch.
    """
    score_bound = 0

    def __init__(self, image_ids, condition_width, width=16):
        super().__init__()
        if image_ids.ndim != 2 or image_ids.shape[1] != 3 or image_ids.requires_grad:
            raise ValueError("actual fixed native image position IDs required")
        self.register_buffer("image_ids", image_ids.detach().float().clone())
        self.value = nn.Linear(64, width)
        self.position = nn.Linear(3, width, bias=False)
        self.condition = nn.Linear(condition_width, 2 * width)
        self.hidden = nn.Linear(width, width)
        self.score = nn.Linear(width, 1)

    def forward(self, response, condition):
        if (response.ndim != 3 or tuple(response.shape[1:]) != (len(self.image_ids), 64)
                or condition.ndim != 2 or len(condition) != len(response)):
            raise ValueError("critic requires unchanged native [B,image_tokens,64] and full context")
        gain, bias = self.condition(condition).chunk(2, dim=-1)
        feature = self.value(response) + self.position(self.image_ids)[None]
        feature = torch.nn.functional.silu(feature * (1 + gain[:, None]) + bias[:, None])
        return self.score(torch.nn.functional.silu(self.hidden(feature))).mean(dim=1).squeeze(-1)
