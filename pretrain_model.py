#!/usr/bin/env python
"""Channel-as-token encoder for variable channel sets (PROJECT_PLAN §2, §7).

The cross-embodiment problem: no two logs carry the same channel set (mean 72.9
of 87 present, p10 55). A fixed-slot model has to learn "absent" as a value; a
channel-as-token model never sees the absent channel at all.

Tokenisation
------------
A window is (T=512 steps, C=87 channels). It is cut into P = T/16 = 32 time
patches, so a token is one **(channel, time-patch)** pair -- 87 x 32 = 2,784
tokens. Each token embeds as

    Linear(patch_values[16]) + channel_embedding[c] + time_position[p]

`channel_embedding` is the learned channel-identity embedding: it indexes the
STABLE ids in `channels.py`, which is why that list is append-only. Two logs
with different channel sets produce different *subsets* of the same token
vocabulary, so a quadrotor and a fixed-wing are the same kind of object to the
encoder -- that is the whole claim being tested.

Attention is **factorised** (axial), alternating inside each block:
  * temporal: over the 32 patches of one channel  -> 32x32 attention
  * channel : over the 87 channels at one patch   -> 87x87 attention
Full attention over 2,784 tokens would be 7.8M scores per head per layer and
buys nothing here; the factorised form is ~30x cheaper and lets absent channels
be dropped cleanly with a key-padding mask in the channel step.

Objective
---------
BERT-style in-place masked reconstruction: a fraction of *present* tokens are
replaced by a learned mask token (identity + position embeddings are kept, so
the model knows which channel at which time it is being asked to reconstruct),
and a linear head predicts the 16 raw normalised values. Loss is MSE over
**valid cells of masked tokens only**. MAE's asymmetric encoder/decoder is not
used: with axial attention the token grid has to stay rectangular, so dropping
tokens would not save the compute it saves in a plain ViT.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class AxialBlock(nn.Module):
    """One transformer block: temporal attention, then channel attention, then MLP."""

    def __init__(self, d, heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.n1 = nn.LayerNorm(d)
        self.time_attn = nn.MultiheadAttention(d, heads, dropout=dropout,
                                               batch_first=True)
        self.n2 = nn.LayerNorm(d)
        self.chan_attn = nn.MultiheadAttention(d, heads, dropout=dropout,
                                               batch_first=True)
        self.n3 = nn.LayerNorm(d)
        hidden = int(d * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(d, hidden), nn.GELU(),
                                 nn.Linear(hidden, d))

    def forward(self, h, chan_pad):
        """h: (B, C, P, d);  chan_pad: (B, C) True where the channel is ABSENT."""
        B, C, P, d = h.shape

        # --- temporal: each (batch, channel) is an independent sequence of P
        z = self.n1(h).reshape(B * C, P, d)
        a, _ = self.time_attn(z, z, z, need_weights=False)
        h = h + a.reshape(B, C, P, d)

        # --- channel: each (batch, patch) is an independent sequence of C.
        # Absent channels are masked out as keys so they contribute nothing.
        z = self.n2(h).permute(0, 2, 1, 3).reshape(B * P, C, d)
        pad = chan_pad.unsqueeze(1).expand(B, P, C).reshape(B * P, C)
        # A row that is entirely padding would make softmax produce NaN; such a
        # window is filtered upstream, but guard anyway by unmasking those rows.
        allpad = pad.all(dim=1, keepdim=True)
        pad = pad & ~allpad
        a, _ = self.chan_attn(z, z, z, key_padding_mask=pad, need_weights=False)
        h = h + a.reshape(B, P, C, d).permute(0, 2, 1, 3)

        return h + self.mlp(self.n3(h))


class ChannelTokenMAE(nn.Module):
    def __init__(self, n_channels=87, window=512, patch=16, d=512, depth=12,
                 heads=8, mlp_ratio=4.0, dropout=0.0, mask_ratio=0.5, mask_mode="random"):
        super().__init__()
        self.mask_mode = mask_mode
        assert window % patch == 0
        self.C, self.T, self.patch = n_channels, window, patch
        self.P = window // patch
        self.mask_ratio = mask_ratio

        self.patch_embed = nn.Linear(patch, d)
        self.chan_embed = nn.Embedding(n_channels, d)      # channel identity
        self.pos_embed = nn.Parameter(torch.zeros(1, 1, self.P, d))
        self.mask_token = nn.Parameter(torch.zeros(d))
        self.blocks = nn.ModuleList(
            [AxialBlock(d, heads, mlp_ratio, dropout) for _ in range(depth)])
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, patch)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.chan_embed.weight, std=0.02)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def to_patches(self, x, valid):
        """(B,T,C) -> (B,C,P,patch); same for the mask."""
        B, T, C = x.shape
        xp = x.permute(0, 2, 1).reshape(B, C, self.P, self.patch)
        vp = valid.permute(0, 2, 1).reshape(B, C, self.P, self.patch)
        return xp, vp

    def forward(self, x, valid, mask_ratio=None, generator=None, mask_mode=None):
        """x: (B,T,C) normalised, valid: (B,T,C) bool. Returns (loss, stats).

        mask_mode (2026-09-11, PROJECT_PLAN v6 item 8):
          "random"   in-place token masking, the original objective (default)
          "channel"  mask ALL patches of a random `mask_ratio` fraction of the present
                     channels -- reconstruct whole signals from the other channels, i.e.
                     the control-loop coupling (cannot be solved by interpolation)
          "span"     mask the LAST `mask_ratio` fraction of patches of every channel --
                     forecast the recent past from the earlier context (dynamics)
          "mixed"    per sample, one of the three at random
        """
        mask_ratio = self.mask_ratio if mask_ratio is None else mask_ratio
        mask_mode = mask_mode or getattr(self, "mask_mode", "random")
        xp, vp = self.to_patches(x, valid)                 # (B,C,P,patch)
        B, C, P, _ = xp.shape

        tok_has_data = vp.any(dim=-1)                      # (B,C,P)
        chan_present = tok_has_data.any(dim=-1)            # (B,C)
        chan_pad = ~chan_present

        # choose masked tokens among those that actually carry data
        r = torch.rand(B, C, P, device=x.device, generator=generator)
        if mask_mode != "random":
            rc = torch.rand(B, C, 1, device=x.device, generator=generator)
            rc = torch.where(chan_present[..., None], rc, torch.full_like(rc, 2.0))
            r_chan = rc.expand(B, C, P)                          # same draw per channel
            pos = torch.arange(P, device=x.device).float() / P   # 0..1 along time
            r_span = (1.0 - pos)[None, None, :].expand(B, C, P)  # last patches lowest
            if mask_mode == "channel":
                r = r_chan
            elif mask_mode == "span":
                r = r_span
            elif mask_mode == "mixed":
                pick = torch.randint(0, 3, (B, 1, 1), device=x.device, generator=generator)
                r = torch.where(pick == 0, r, torch.where(pick == 1, r_chan, r_span))
            else:
                raise ValueError(mask_mode)
        r = torch.where(tok_has_data, r, torch.full_like(r, 2.0))
        masked = (r < mask_ratio) & tok_has_data
        # guarantee at least one target per sample, else the loss is undefined
        empty = ~masked.any(dim=(1, 2))
        if empty.any():
            flat = tok_has_data.reshape(B, -1).float()
            pick = torch.multinomial(flat[empty] + 1e-9, 1)
            m2 = masked.reshape(B, -1).clone()
            m2[empty] = m2[empty].scatter(1, pick, True)
            masked = m2.reshape(B, C, P)

        h = self.patch_embed(xp)
        h = torch.where(masked.unsqueeze(-1), self.mask_token.to(h.dtype), h)
        h = h + self.chan_embed.weight.view(1, C, 1, -1) + self.pos_embed

        for blk in self.blocks:
            h = blk(h, chan_pad)
        pred = self.head(self.norm(h))                     # (B,C,P,patch)

        target_cells = vp & masked.unsqueeze(-1)
        n = target_cells.sum()
        if n == 0:
            return pred.sum() * 0.0, {"n_target": 0}
        se = (pred - xp) ** 2
        loss = se[target_cells].mean()
        with torch.no_grad():
            stats = {"n_target": int(n),
                     "masked_frac": float(masked.float().mean()),
                     "present_ch": float(chan_present.float().sum(1).mean())}
        return loss, stats

    def encode(self, x, valid):
        """Representation for downstream probing: (B, C, P, d) with no masking."""
        xp, vp = self.to_patches(x, valid)
        chan_pad = ~vp.any(dim=-1).any(dim=-1)
        h = self.patch_embed(xp) + self.chan_embed.weight.view(
            1, self.C, 1, -1) + self.pos_embed
        for blk in self.blocks:
            h = blk(h, chan_pad)
        return self.norm(h)


def build_model(cfg):
    return ChannelTokenMAE(
        n_channels=cfg["n_channels"], window=cfg["window"], patch=cfg["patch"],
        d=cfg["d"], depth=cfg["depth"], heads=cfg["heads"],
        mlp_ratio=cfg.get("mlp_ratio", 4.0), dropout=cfg.get("dropout", 0.0),
        mask_ratio=cfg.get("mask_ratio", 0.5), mask_mode=cfg.get("mask_mode", "random"))


def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
