# ComfyUI-Anima-DC

ComfyUI custom nodes for **Anima-DC** models: [Anima](https://huggingface.co/circlestone-labs/Anima)
moved onto a deeply compressed latent space with the [DC-Gen](https://arxiv.org/abs/2509.25180)
recipe, such as [Battera-Anima-DC](https://huggingface.co/FA770/Battera-Anima-DC).

These models use a 32× compressed autoencoder instead of the Qwen-Image VAE, so they cannot be
loaded with the stock Anima workflow. These nodes load the model, create latents of the right size,
and encode / decode with the matching VAE.

## Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/FA7770/ComfyUI-Anima-DC
```

Restart ComfyUI. No extra dependencies are needed. The nodes appear under the **Anima-DC** category.

Requires a ComfyUI version with Anima support.

## Models

| File | Download | Put it in |
|---|---|---|
| `Battera-Anima-DC-v1.0.safetensors` | [Battera-Anima-DC](https://huggingface.co/FA770/Battera-Anima-DC) | `models/diffusion_models` |
| `Battera-VAE-f32c128-v1.0.safetensors` | [Battera-VAE-f32c128](https://huggingface.co/FA770/Battera-VAE-f32c128) | `models/vae` |
| Qwen3 0.6B (same as stock Anima) | | `models/text_encoders` |

The model and the VAE are a matched pair. Use the same version number for both.

## Nodes

| Node | Inputs → Output | |
|---|---|---|
| **Anima-DC Loader** | model file → MODEL | Loads an Anima-DC model |
| **Anima-DC Empty Latent** | MODEL, width, height, batch → LATENT | Width / height are rounded down to a multiple of 32 |
| **Anima-DC VAE Loader** | VAE file → ANIMA_DC_VAE | Loads the matching VAE (encoder + decoder) |
| **Anima-DC VAE Decode** | LATENT, ANIMA_DC_VAE → IMAGE | |
| **Anima-DC VAE Encode** | IMAGE, ANIMA_DC_VAE → LATENT | For img2img; the image is center-cropped to a multiple of 32 |
| **Anima-DC Flow Shift** | MODEL, shift → MODEL | Optional. Changes the sampling shift (default 3.0, as in Anima). `shift = 0` computes it from width / height |

## Basic workflow

| Node | Input | Connect from |
|---|---|---|
| Anima-DC Empty Latent | model | Anima-DC Loader |
| KSampler | model | Anima-DC Loader |
| KSampler | latent_image | Anima-DC Empty Latent |
| KSampler | positive / negative | CLIP Text Encode (Load CLIP → Qwen3 0.6B) |
| Anima-DC VAE Decode | samples | KSampler |
| Anima-DC VAE Decode | vae | Anima-DC VAE Loader |
| Save Image | images | Anima-DC VAE Decode |

For img2img, replace Anima-DC Empty Latent with **Load Image → Anima-DC VAE Encode** and lower `denoise`
in KSampler.

Use the stock **CLIP Text Encode** and **KSampler**; prompting is the same as stock Anima. See the
model card for recommended settings.

## Notes

- Do not use the stock **VAE Decode** / **VAE Encode** nodes; use Anima-DC VAE Decode / Encode.
- The latent is already normalized inside the model, so Anima-DC VAE Decode receives raw latents from
  KSampler as is.
- Encoding and decoding are not tiled. Very large images need a lot of VRAM.

## How it works

ComfyUI's Anima support assumes a 16-channel latent and patch size 2. This node pack reuses
ComfyUI's own Anima implementation and only changes what differs:

- model detection reads the input / output channels and patch size from the weights
- the per-channel latent normalization stored in the model file is applied as the model's latent
  format
- the VAE is a standalone PyTorch module (`dcae.py`), loaded from the configuration stored in the
  file metadata

## License

Apache License 2.0. See `LICENSE` and `NOTICE`.

`dcae.py` is a modified port of the DC-AE building blocks from
[mit-han-lab/efficientvit](https://github.com/mit-han-lab/efficientvit) (Apache-2.0), obtained
through [lavinal712/AutoencoderKL](https://github.com/lavinal712/AutoencoderKL) (MIT). The
original license texts are in `licenses/`.

ComfyUI (GPL-3.0) is imported at runtime; no ComfyUI source code is included.

Model weights are not part of this repository and have their own licenses. Battera-Anima-DC is a
derivative of Anima under the CircleStone Labs Non-Commercial License.

## Credits

The code in this repository was written by Claude (Anthropic) under the direction of FA770.

- GitHub: [FA7770](https://github.com/FA7770) / Hugging Face: [FA770](https://huggingface.co/FA770)
