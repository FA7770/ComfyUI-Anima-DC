# -*- coding: utf-8 -*-
# Copyright 2026 FA770
# SPDX-License-Identifier: Apache-2.0
"""
Anima-DC: ComfyUI nodes for Anima models adapted to a deeply compressed latent space (DC-Gen).

  Anima-DC Loader       loads an Anima-DC transformer (e.g. Battera-Anima-DC)
  Anima-DC Empty Latent empty latent at 1/new_f resolution with C_new channels
  Anima-DC Flow Shift   optional; sets the sampling shift
  Anima-DC VAE Loader   loads the matching VAE (encoder + decoder)
  Anima-DC VAE Decode   LATENT -> IMAGE
  Anima-DC VAE Encode   IMAGE -> LATENT (img2img)

Metadata keys in the model files keep the dcgen_* prefix.
"""

import inspect
import json
import logging
import math

import torch
import torch.nn as nn

import comfy.latent_formats
import comfy.model_detection
import comfy.model_management as mm
import comfy.model_sampling
import comfy.sd
import comfy.utils
import folder_paths

from .dcae import (CONFIG_META_KEY, ENCODER_META_KEY, Decoder, DecoderConfig,
                   Encoder, EncoderConfig)
from .wide_head import HEAD_PREFIX, build_wide_head

EMB_KEY = "x_embedder.proj.1.weight"
HEAD_KEY = "final_layer.linear.weight"
log = logging.getLogger("Anima-DC")


def calculate_shift(image_seq_len, base_seq_len=256, max_seq_len=4096,
                    base_shift=0.5, max_shift=1.15):
    """stage2_train.py / generate_gui.py と同じ式 (FLUX / DC-Gen の動的 shift)。"""
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    return math.exp(image_seq_len * m + b)


# ===========================================================================
# latent format
# ===========================================================================
class AnimaDCLatentFormat(comfy.latent_formats.LatentFormat):
    """process_in = (z - mean) / std,  process_out = z * std + mean。

    学習時の正規化 (latents_std / latents_mean) をサンプラの標準経路に載せる。
    これで KSampler の出力 LATENT は AE の生の潜在になり、デコーダにそのまま渡せる。
    """
    latent_dimensions = 3
    temporal_downscale_ratio = 1

    def __init__(self, std, mean, new_f, new_patch):
        self.scale_factor = 1.0
        self.latent_channels = std.numel()
        self.spacial_downscale_ratio = new_f
        self.new_f, self.new_patch = new_f, new_patch
        self.latents_std = std.float().view(1, -1, 1, 1, 1)
        self.latents_mean = (mean.float().view(1, -1, 1, 1, 1) if mean is not None
                             else torch.zeros_like(self.latents_std))

    def _stats(self, latent):
        s = self.latents_std.to(latent.device, latent.dtype)
        m = self.latents_mean.to(latent.device, latent.dtype)
        if latent.ndim == 4:
            s, m = s.squeeze(2), m.squeeze(2)
        return s, m

    def process_in(self, latent):
        s, m = self._stats(latent)
        return (latent - m) / s

    def process_out(self, latent):
        s, m = self._stats(latent)
        return latent * s + m


def _get_format(model):
    fmt = model.get_model_object("latent_format")
    if not isinstance(fmt, AnimaDCLatentFormat):
        raise RuntimeError("Connect a MODEL loaded with Anima-DC Loader")
    return fmt


# ===========================================================================
# loader
# ===========================================================================
class _LazyDeviceHead(nn.Module):
    """強化ヘッドは ComfyUI の重み管理の外にあるので、呼ばれた時に入力側へ寄せる。"""

    def __init__(self, head):
        super().__init__()
        self.head = head

    def forward(self, x):
        p = next(self.head.parameters())
        if p.device != x.device or p.dtype != x.dtype:
            self.head.to(device=x.device, dtype=x.dtype)
        return self.head(x)


def _load_with_override(sd, metadata, model_options, C, patch):
    """Anima の自動判定は in_channels を x_embedder の形から、patch=2 / out=16 を固定値で
    決める。判定の間だけ x_embedder を元の形に見せ、結果の in/out/patch を上書きする。
    RoPE の外挿比 (4.0) などは元の Anima と同じ判定を通る。"""
    orig = comfy.model_detection.detect_unet_config

    # ComfyUI の版によって metadata 引数の有無が違うので、渡された引数をそのまま流す
    def patched(state_dict, key_prefix, *args, **kwargs):
        k = key_prefix + EMB_KEY
        if k not in state_dict:
            return orig(state_dict, key_prefix, *args, **kwargs)
        w = state_dict[k]
        fake = dict(state_dict)
        fake[k] = torch.empty((w.shape[0], (16 + 1) * 4), device="meta")
        cfg = orig(fake, key_prefix, *args, **kwargs)
        if cfg is None or cfg.get("image_model") != "anima":
            raise RuntimeError(
                f"The model was not detected as Anima ({(cfg or {}).get('image_model')}). "
                f"This ComfyUI version may not support Anima; please update ComfyUI")
        cfg = dict(cfg)
        cfg["in_channels"] = C
        cfg["out_channels"] = C
        cfg["patch_spatial"] = patch
        return cfg

    comfy.model_detection.detect_unet_config = patched
    try:
        fn = comfy.sd.load_diffusion_model_state_dict
        kw = {"model_options": model_options}
        if "metadata" in inspect.signature(fn).parameters:
            kw["metadata"] = metadata
        return fn(sd, **kw)
    finally:
        comfy.model_detection.detect_unet_config = orig


def _apply_rope(dm, rope, new_f, new_patch, base_unit=16):
    for part in str(rope).split("+"):
        part = part.strip()
        if part in ("", "none"):
            continue
        if part == "ntk":
            pe = dm.pos_embedder
            dim_h = pe.dim_spatial_range.numel() * 2
            ratio = (new_f * new_patch) // base_unit
            base = getattr(dm, "rope_h_extrapolation_ratio", 4.0)
            f = (base / ratio) ** (dim_h / (dim_h - 2))
            pe.h_ntk_factor = pe.w_ntk_factor = f
            log.info(f"[Anima-DC] rope ntk factor={f:.4f}")
        else:
            raise RuntimeError(
                f"rope='{part}' is not supported. Use a model trained with rope=none")


class AnimaDCLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "unet_name": (folder_paths.get_filename_list("diffusion_models"),),
            "weight_dtype": (["default", "bf16", "fp16", "fp32", "fp8_e4m3fn"],),
        }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load"
    CATEGORY = "Anima-DC"

    def load(self, unet_name, weight_dtype):
        path = folder_paths.get_full_path_or_raise("diffusion_models", unet_name)
        sd, md = comfy.utils.load_torch_file(path, return_metadata=True)
        md = md or {}
        if "dcgen_new_f" not in md:
            raise RuntimeError("This file has no dcgen_* metadata. Select an Anima-DC model (e.g. Battera-Anima-DC)")

        new_f = int(md["dcgen_new_f"])
        new_patch = int(md["dcgen_new_patch"])
        rope = md.get("dcgen_rope", "none")
        head_type = md.get("dcgen_head_type", "linear")

        std = sd.pop("dcgen.latents_std")
        mean = sd.pop("dcgen.latents_mean", None)
        C = std.numel()
        extra = {k[len(HEAD_PREFIX):]: sd.pop(k) for k in list(sd) if k.startswith(HEAD_PREFIX)}
        head_main = sd[HEAD_KEY].clone() if head_type != "linear" else None
        D = sd[EMB_KEY].shape[0]

        opts = {}
        dt = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32,
              "fp8_e4m3fn": torch.float8_e4m3fn}.get(weight_dtype)
        if dt is not None:
            opts["dtype"] = dt

        patcher = _load_with_override(sd, md, opts, C, new_patch)
        if patcher is None:
            raise RuntimeError("Failed to load the model")
        model = patcher.model
        dm = model.diffusion_model
        if dm.x_embedder.proj[1].weight.shape[1] != (C + 1) * new_patch ** 2:
            raise RuntimeError("Unexpected x_embedder shape")

        fmt = AnimaDCLatentFormat(std, mean, new_f, new_patch)
        model.latent_format = fmt
        model.model_config.latent_format = fmt
        _apply_rope(dm, rope, new_f, new_patch)

        if head_type != "linear":
            head = build_wide_head(
                D, C * new_patch ** 2, head_type,
                int(md.get("dcgen_head_width", 4096)), int(md.get("dcgen_head_depth", 2)),
                int(md.get("dcgen_head_heads", 8)), extra)
            head.main.weight.data.copy_(head_main.float())
            head.eval().requires_grad_(False)
            patcher.add_object_patch("diffusion_model.final_layer.linear", _LazyDeviceHead(head))
            log.info(f"[Anima-DC] applied {head_type} head")

        log.info(f"[Anima-DC] {unet_name}: C={C} f{new_f}p{new_patch} rope={rope} "
                 f"head={head_type} ema={md.get('dcgen_ema')} lora={md.get('dcgen_lora_merged')}")
        return (patcher,)


# ===========================================================================
# latent / shift
# ===========================================================================
class AnimaDCEmptyLatent:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "width": ("INT", {"default": 1024, "min": 32, "max": 8192, "step": 32}),
            "height": ("INT", {"default": 1024, "min": 32, "max": 8192, "step": 32}),
            "batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),
        }}

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "generate"
    CATEGORY = "Anima-DC"

    def generate(self, model, width, height, batch_size):
        fmt = _get_format(model)
        unit = fmt.new_f * fmt.new_patch
        w = max(unit, width // unit * unit)
        h = max(unit, height // unit * unit)
        if (w, h) != (width, height):
            log.info(f"[Anima-DC] {width}x{height} -> {w}x{h} (rounded down to a multiple of {unit})")
        lat = torch.zeros([batch_size, fmt.latent_channels, 1, h // fmt.new_f, w // fmt.new_f],
                          device=mm.intermediate_device())
        return ({"samples": lat},)


class AnimaDCFlowShift:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "shift": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0, "step": 0.01,
                                "tooltip": "0 = compute from the token count of width / height. "
                                           "Any other value is used as is"}),
            "width": ("INT", {"default": 1024, "min": 32, "max": 8192, "step": 32,
                              "tooltip": "used only when shift = 0"}),
            "height": ("INT", {"default": 1024, "min": 32, "max": 8192, "step": 32,
                               "tooltip": "used only when shift = 0"}),
        }}

    RETURN_TYPES = ("MODEL", "FLOAT")
    RETURN_NAMES = ("model", "shift")
    FUNCTION = "patch"
    CATEGORY = "Anima-DC"

    def patch(self, model, shift, width, height):
        fmt = _get_format(model)
        unit = fmt.new_f * fmt.new_patch
        tokens = max(1, height // unit) * max(1, width // unit)
        auto = shift <= 0
        if auto:
            shift = calculate_shift(tokens)
        m = model.clone()

        class ModelSamplingAnimaDC(comfy.model_sampling.ModelSamplingDiscreteFlow,
                                 comfy.model_sampling.CONST):
            pass

        ms = ModelSamplingAnimaDC(model.model.model_config)
        ms.set_parameters(shift=float(shift), multiplier=1.0)
        m.add_object_patch("model_sampling", ms)
        log.info(f"[Anima-DC] shift={shift:.4f}" + (f" (auto, tokens={tokens})" if auto else ""))
        return (m, float(shift))


# ===========================================================================
# AE (encoder + decoder)
# ===========================================================================
class AnimaDCVAE:
    """生の潜在を扱う。正規化は MODEL 側の latent format が受け持つ。"""

    def __init__(self, encoder, decoder, dtype):
        self.encoder, self.decoder, self.dtype = encoder, decoder, dtype
        self.latent_channels = decoder.cfg.latent_channels
        self.f = decoder.spatial_compression_ratio

    @torch.no_grad()
    def _run(self, net, x, act_bytes, post):
        dev = mm.get_torch_device()
        nbytes = sum(p.numel() * p.element_size() for p in net.parameters())
        try:
            mm.free_memory(nbytes + act_bytes, dev)
        except Exception:
            pass
        net.to(dev)
        out = []
        pbar = comfy.utils.ProgressBar(x.shape[0])
        try:
            for i in range(x.shape[0]):
                mm.throw_exception_if_processing_interrupted()
                out.append(post(net(x[i:i + 1].to(dev, self.dtype)).float())
                           .to(mm.intermediate_device()))
                pbar.update(1)
        finally:
            net.to(mm.unet_offload_device())
            mm.soft_empty_cache()
        return torch.cat(out, 0)

    def decode(self, z):
        """(B,C,h,w) -> IMAGE (B,H,W,3) 0..1"""
        h, w = z.shape[-2:]
        act = (h * self.f) * (w * self.f) * 256 * 4 * 3      # 粗い見積もり
        return self._run(self.decoder, z, act,
                         lambda y: ((y.clamp(-1, 1) + 1) * 0.5).permute(0, 2, 3, 1))

    def encode(self, image):
        """IMAGE (B,H,W,3) 0..1 -> (B,C,h,w)。f の倍数に中央クロップする。"""
        if self.encoder is None:
            raise RuntimeError("This AE file has no encoder (decoder-only file). Use the full Battera-VAE file")
        H, W = image.shape[1:3]
        h, w = H // self.f * self.f, W // self.f * self.f
        if h == 0 or w == 0:
            raise RuntimeError(f"Image too small ({W}x{H}, minimum {self.f})")
        t, l = (H - h) // 2, (W - w) // 2
        x = image[:, t:t + h, l:l + w, :3].permute(0, 3, 1, 2) * 2.0 - 1.0
        return self._run(self.encoder, x, h * w * 256 * 4 * 3, lambda y: y)


class AnimaDCVAELoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "vae_name": (folder_paths.get_filename_list("vae"),),
            "dtype": (["fp32", "bf16", "fp16"],),
        }}

    RETURN_TYPES = ("ANIMA_DC_VAE",)
    RETURN_NAMES = ("vae",)
    FUNCTION = "load"
    CATEGORY = "Anima-DC"

    def load(self, vae_name, dtype):
        path = folder_paths.get_full_path_or_raise("vae", vae_name)
        sd, md = comfy.utils.load_torch_file(path, return_metadata=True)
        if not md or CONFIG_META_KEY not in md:
            raise RuntimeError("This file has no Anima-DC VAE metadata. Select an Anima-DC VAE file (e.g. Battera-VAE-f32c128)")
        dt = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[dtype]

        def build(cls_, cfg, prefix):
            m = cls_(cfg)
            m.load_state_dict({k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)},
                              strict=True)
            return m.to(dtype=dt).eval().requires_grad_(False)

        # decoder.* が無いのは旧形式 (デコーダだけ・prefix 無し)
        full = any(k.startswith("decoder.") for k in sd)
        dec = build(Decoder, DecoderConfig.from_dict(json.loads(md[CONFIG_META_KEY])),
                    "decoder." if full else "")
        enc = None
        if ENCODER_META_KEY in md:
            enc = build(Encoder, EncoderConfig.from_dict(json.loads(md[ENCODER_META_KEY])), "encoder.")
        log.info(f"[Anima-DC] VAE {vae_name}: latent {dec.cfg.latent_channels}ch "
                 f"f{dec.spatial_compression_ratio} {dtype} encoder={'yes' if enc else 'no'}")
        return (AnimaDCVAE(enc, dec, dt),)


class AnimaDCVAEDecode:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"samples": ("LATENT",), "vae": ("ANIMA_DC_VAE",)}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "decode"
    CATEGORY = "Anima-DC"

    def decode(self, samples, vae):
        z = samples["samples"]
        if z.ndim == 5:
            B, C, T, H, W = z.shape
            z = z.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        if z.shape[1] != vae.latent_channels:
            raise RuntimeError(f"Latent has {z.shape[1]} channels but the AE expects {vae.latent_channels}")
        return (vae.decode(z),)


class AnimaDCVAEEncode:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"pixels": ("IMAGE",), "vae": ("ANIMA_DC_VAE",)}}

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "encode"
    CATEGORY = "Anima-DC"

    def encode(self, pixels, vae):
        return ({"samples": vae.encode(pixels).unsqueeze(2)},)


NODE_CLASS_MAPPINGS = {
    "AnimaDCLoader": AnimaDCLoader,
    "AnimaDCEmptyLatent": AnimaDCEmptyLatent,
    "AnimaDCFlowShift": AnimaDCFlowShift,
    "AnimaDCVAELoader": AnimaDCVAELoader,
    "AnimaDCVAEDecode": AnimaDCVAEDecode,
    "AnimaDCVAEEncode": AnimaDCVAEEncode,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "AnimaDCLoader": "Anima-DC Loader",
    "AnimaDCEmptyLatent": "Anima-DC Empty Latent",
    "AnimaDCFlowShift": "Anima-DC Flow Shift",
    "AnimaDCVAELoader": "Anima-DC VAE Loader",
    "AnimaDCVAEDecode": "Anima-DC VAE Decode",
    "AnimaDCVAEEncode": "Anima-DC VAE Encode",
}
