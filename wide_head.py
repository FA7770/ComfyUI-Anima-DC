# -*- coding: utf-8 -*-
# Copyright 2026 FA770
# SPDX-License-Identifier: Apache-2.0
"""
final_layer の強化ヘッド (wide_head.py の推論部分だけ)。

学習コードの WideHead と state_dict のキー名が一致する。強化部は
`final_layer.head.<key>` として保存されている (main は final_layer.linear.weight)。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

HEAD_PREFIX = "final_layer.head."


class _Block(nn.Module):
    def __init__(self, w, heads, mlp_ratio=4.0):
        super().__init__()
        assert w % heads == 0, f"head_width {w} is not divisible by head_heads {heads}"
        self.h = heads
        self.n1 = nn.LayerNorm(w)
        self.qkv = nn.Linear(w, w * 3, bias=False)
        self.proj = nn.Linear(w, w, bias=False)
        self.n2 = nn.LayerNorm(w)
        hid = int(w * mlp_ratio)
        self.fc1 = nn.Linear(w, hid)
        self.fc2 = nn.Linear(hid, w)

    def forward(self, x):
        B, T, W = x.shape
        q, k, v = self.qkv(self.n1(x)).view(B, T, 3, self.h, W // self.h).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v)
        x = x + self.proj(a.transpose(1, 2).reshape(B, T, W))
        x = x + self.fc2(F.silu(self.fc1(self.n2(x))))
        return x


class WideHead(nn.Module):
    def __init__(self, dim, out_dim, mode="mlp", width=4096, depth=2, heads=8, ffn_ratio=4.0):
        super().__init__()
        self.mode, self.width, self.depth = mode, width, depth
        self.main = nn.Linear(dim, out_dim, bias=False)
        self.inp = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, width))
        if mode == "mlp":
            layers = []
            for _ in range(max(0, depth - 1)):
                layers += [nn.Linear(width, width), nn.SiLU()]
            self.branch = nn.Sequential(nn.SiLU(), *layers)
        elif mode == "ddt":
            self.branch = nn.ModuleList([_Block(width, heads, ffn_ratio) for _ in range(depth)])
            self.norm = nn.LayerNorm(width)
        else:
            raise ValueError(mode)
        self.out_proj = nn.Linear(width, out_dim, bias=False)

    @property
    def weight(self):
        return self.main.weight

    def forward(self, x):
        y = self.main(x)
        shp = x.shape
        if x.ndim == 2:
            h = x.unsqueeze(0)
        elif x.ndim == 3:
            h = x
        else:
            h = x.reshape(shp[0], -1, shp[-1])
        h = self.inp(h)
        if self.mode == "ddt":
            for b in self.branch:
                h = b(h)
            h = self.norm(h)
        else:
            h = self.branch(h)
        return y + self.out_proj(h).reshape(y.shape)


def infer_ffn_ratio(extra_sd, width):
    """ddt の fc1 形状から ffn_ratio を復元する (metadata に保存されていないため)。"""
    k = "branch.0.fc1.weight"
    if k in extra_sd:
        return extra_sd[k].shape[0] / width
    return 4.0


def build_wide_head(dim, out_dim, mode, width, depth, heads, extra_sd):
    """extra_sd は HEAD_PREFIX を除いたキー (main 以外)。"""
    h = WideHead(dim, out_dim, mode=mode, width=width, depth=depth, heads=heads,
                 ffn_ratio=infer_ffn_ratio(extra_sd, width))
    cur = h.state_dict()
    bad = [k for k, v in extra_sd.items() if k not in cur or cur[k].shape != v.shape]
    if bad:
        raise RuntimeError(f"Head weight shape mismatch: {bad[:4]}")
    missing = [k for k in cur if not k.startswith("main.") and k not in extra_sd]
    if missing:
        raise RuntimeError(f"Missing head weights: {missing[:4]}")
    h.load_state_dict(extra_sd, strict=False)
    return h
