import importlib.util
import math
import urllib.request
from pathlib import Path

import safetensors.torch as stt
import torch
import torch.nn.functional as F
from diffusers import AutoencoderTiny
from diffusers.utils.accelerate_utils import apply_forward_hook
from transformers import AutoModel

TAEF2_WEIGHTS_URL = "https://huggingface.co/madebyollin/taef2/resolve/main/taef2.safetensors"
TAESD_URL = "https://raw.githubusercontent.com/madebyollin/taesd/refs/heads/main/taesd.py"
TAEF2_CACHE_DIR = Path.home() / ".cache" / "taef2"
DINO_MODEL_NAME = "facebook/dinov2-with-registers-small"
TAEF1_MODEL_NAME = "madebyollin/taef1"
TAESDXL_MODEL_NAME = "madebyollin/taesdxl"


class _DotDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__



def _ensure_taef2():
    TAEF2_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    weights = TAEF2_CACHE_DIR / "taef2.safetensors"
    taesd_py = TAEF2_CACHE_DIR / "taesd.py"
    if not weights.exists():
        print("Downloading taef2.safetensors...")
        urllib.request.urlretrieve(TAEF2_WEIGHTS_URL, weights)
    if not taesd_py.exists():
        print("Downloading taesd.py...")
        urllib.request.urlretrieve(TAESD_URL, taesd_py)
    return weights, taesd_py


def _convert_sd_to_taesd(sd):
    out = {}
    for k, v in sd.items():
        encdec, _layers, index, *suffix = k.split(".")
        offset = 1 if encdec == "decoder" else 0
        out[".".join([encdec, str(int(index) + offset), *suffix])] = v
    return out


class TAEF2Wrapper(torch.nn.Module):
    def __init__(self):
        super().__init__()
        weights, taesd_py = _ensure_taef2()
        spec = importlib.util.spec_from_file_location("taesd", taesd_py)
        taesd_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(taesd_mod)
        self.taesd = taesd_mod.TAESD(
            encoder_path=None, decoder_path=None, latent_channels=32, arch_variant="flux_2"
        ).to(torch.bfloat16)
        self.taesd.load_state_dict(_convert_sd_to_taesd(stt.load_file(weights)))
        self.config = _DotDict(batch_norm_eps=1e-5, scaling_factor=None, shift_factor=None)

    @apply_forward_hook
    def encode(self, x):
        return _DotDict(latent_dist=_DotDict(
            sample=lambda: self.taesd.encoder(x.to(torch.bfloat16).mul(0.5).add_(0.5)).to(x.dtype)
        ))

    @apply_forward_hook
    def decode(self, x, return_dict=True):
        out = self.taesd.decoder(x.to(torch.bfloat16)).mul(2).sub_(1).clamp_(-1, 1).to(x.dtype)
        return {'sample': out} if return_dict else (out,)


class VisionExtractor:
    """Diffusion latents (FLUX.2, FLUX.1, or SDXL) -> tiny VAE decoder -> DINOv2 -> L2-normalised features."""

    def __init__(self, model='flux2', device='cuda'):
        self.device = device

        if model == 'flux2':
            self.decoder = TAEF2Wrapper().to(device).eval()
        elif model == 'flux1':
            self.decoder = AutoencoderTiny.from_pretrained(
                TAEF1_MODEL_NAME, torch_dtype=torch.bfloat16
            ).to(device).eval()
        elif model == 'sdxl':
            self.decoder = AutoencoderTiny.from_pretrained(
                TAESDXL_MODEL_NAME, torch_dtype=torch.float16
            ).to(device).eval()
        else:
            raise ValueError(f"Unknown model: {model!r}")
        for p in self.decoder.parameters():
            p.requires_grad_(False)
        self.decoder_dtype = next(self.decoder.parameters()).dtype

        self.model = AutoModel.from_pretrained(DINO_MODEL_NAME).to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        self._mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self._std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        # Unpack FLUX latents (B, seq, C) -> (B, lat_ch, H, W); SDXL latents are already spatial.
        if latents.dim() == 3:
            B, seq_len, C = latents.shape
            h_p = int(math.sqrt(seq_len))
            lat_ch = C // 4  # patch_size=2, so C = lat_ch * 2 * 2
            latents = latents.view(B, h_p, h_p, lat_ch, 2, 2)
            latents = latents.permute(0, 3, 1, 4, 2, 5)
            latents = latents.reshape(B, lat_ch, h_p * 2, h_p * 2)

        cfg = getattr(self.decoder, 'config', None)
        if cfg:
            if getattr(cfg, 'scaling_factor', None):
                latents = latents / cfg.scaling_factor
            if getattr(cfg, 'shift_factor', None):
                latents = latents + cfg.shift_factor

        images = self.decoder.decode(latents.to(self.decoder_dtype))['sample'].float()
        images = images / 2 + 0.5  # [-1,1] -> [0,1]
        images = images.clamp(0, 1)

        images = F.interpolate(images.float(), size=(224, 224), mode='bilinear', align_corners=False)
        images = (images - self._mean) / self._std

        features = self.model(pixel_values=images).last_hidden_state[:, 0]
        return F.normalize(features, p=2, dim=-1)
