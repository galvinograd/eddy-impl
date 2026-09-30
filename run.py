import argparse
from dataclasses import dataclass
from typing import Any, Optional

import torch
from diffusers import (
    AutoencoderKL,
    Flux2KleinPipeline,
    FluxPipeline,
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
)

from cb_eddy import EDDYCallback, EDDYConfig
from embedding import VisionExtractor
from logistics import (
    outdir_commit,
    outdir_for,
    outdir_init,
    prompts_select_all,
)

DEVICE = "cuda"
SAVE_RESOLUTION = 512
_LANCZOS = 2

SEED = 42
VARIANTS_PER_PROMPT = 4


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    pipeline_cls: type
    dtype: torch.dtype
    parameterisation: str
    num_inference_steps: int
    guidance_scale: float
    resolution: int


MODELS: dict[str, ModelSpec] = {
    "flux2": ModelSpec(
        model_id="black-forest-labs/FLUX.2-klein-base-9B",
        pipeline_cls=Flux2KleinPipeline,
        dtype=torch.bfloat16,
        parameterisation="flux",
        num_inference_steps=28,
        guidance_scale=3.5,
        resolution=512,
    ),
    "flux1": ModelSpec(
        model_id="black-forest-labs/FLUX.1-dev",
        pipeline_cls=FluxPipeline,
        dtype=torch.bfloat16,
        parameterisation="flux",
        num_inference_steps=30,
        guidance_scale=3.5,
        resolution=512,
    ),
    "sdxl": ModelSpec(
        model_id="stabilityai/stable-diffusion-xl-base-1.0",
        pipeline_cls=StableDiffusionXLPipeline,
        dtype=torch.float16,
        parameterisation="sdxl",
        num_inference_steps=20,
        guidance_scale=7.5,
        resolution=1024,
    ),
}


EDDY_DEFAULTS: dict[str, dict[str, float]] = {
    "flux2": {"stop_step_ratio": 0.2, "fd_eps": 4.6e-4, "hutchinson_samples": 5, "gamma": 0.05},
    "flux1": {"stop_step_ratio": 0.2, "fd_eps": 1e-3, "hutchinson_samples": 5, "gamma": 0.025},
    "sdxl":  {"stop_step_ratio": 0.2, "fd_eps": 1e-3, "hutchinson_samples": 5, "gamma": 0.025},
}

EDDY_CONFIGS: dict[str, dict[str, dict[str, float]]] = {
    "flux1": {
        "low": {"wg": 40},
        "mid": {"wg": 80},
        "high": {"wg": 120},
    },
    "flux2": {
        "low": {"wg": 50},
        "mid": {"wg": 130},
        "high": {"wg": 160},
    },
    "sdxl": {
        "low": {"wg": 80},
        "mid": {"wg": 120},
        "high": {"wg": 160},
    },
}


DIVERSITY_LEVELS = ("low", "mid", "high")


def default_eddy_config(model: str, diversity: str = "mid", **overrides) -> EDDYConfig:
    if model not in EDDY_CONFIGS:
        raise ValueError(f"Unknown model: {model!r} (have {sorted(EDDY_CONFIGS)})")
    if diversity not in EDDY_CONFIGS[model]:
        raise ValueError(f"Unknown diversity: {diversity!r} (have {sorted(EDDY_CONFIGS[model])})")
    config = EDDY_CONFIGS[model][diversity]
    return EDDYConfig(**{**EDDY_DEFAULTS[model], **config, **overrides})


def coupled_init(X):
    n = X.shape[0]
    if n < 2:
        return X
    return (X - X.mean(dim=0, keepdim=True)) * math.sqrt(n / (n - 1))


def _install_coupled_init(pipe):
    inner = pipe.prepare_latents

    def prepare_latents(*args, **kwargs):
        out = inner(*args, **kwargs)
        if isinstance(out, tuple):
            latents, *rest = out
            return (coupled_init(latents), *rest)
        return coupled_init(out)

    pipe.prepare_latents = prepare_latents


def _build_pipeline(spec: ModelSpec, model_id: str):
    print(f"Loading model: {model_id}")

    if spec.parameterisation == "sdxl":
        unet = UNet2DConditionModel.from_pretrained(
            model_id,
            subfolder="unet",
            torch_dtype=torch.float16,
        )
        vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix",
            torch_dtype=spec.dtype,
        )
        pipe = spec.pipeline_cls.from_pretrained(
            model_id, 
            vae=vae, 
            unet=unet, 
            variant='fp16', 
            torch_dtype=spec.dtype
        )
        pipe.to(DEVICE)
        pipe.vae.enable_slicing()
        return pipe

    pipe = spec.pipeline_cls.from_pretrained(model_id, torch_dtype=spec.dtype)
    pipe.enable_model_cpu_offload()
    return pipe


def _build_callback(pipe, spec: ModelSpec, model: str, eddy_config: EDDYConfig,
                    num_inference_steps: int) -> EDDYCallback:
    """EDDY guidance callback, wired into the pipeline's tensor inputs."""
    cfg = eddy_config
    print(f"EDDY[{model}]: wg={cfg.wg}  sr={cfg.stop_step_ratio}"
          f"  hutchinson={cfg.hutchinson_samples}  fd_eps={cfg.fd_eps}  gamma={cfg.gamma}")
    if "noise_pred" not in pipe._callback_tensor_inputs:
        pipe._callback_tensor_inputs.append("noise_pred")
    extractor = VisionExtractor(model=model, device=DEVICE)
    return EDDYCallback(config=cfg, num_steps=num_inference_steps,
                        feature_extractor=extractor, model=spec.parameterisation)


def _to_save_size(image):
    if image.width == SAVE_RESOLUTION and image.height == SAVE_RESOLUTION:
        return image
    return image.resize((SAVE_RESOLUTION, SAVE_RESOLUTION), resample=_LANCZOS)


def generate_images(
    prompts,
    output_dir,
    eddy_config: Optional[EDDYConfig],
    model: str = "flux2",
    num_inference_steps: Optional[int] = None,
    guidance_scale: Optional[float] = None,
    seed=42,
    model_name: Optional[str] = None,
    variants_per_prompt=1,
):
    if model not in MODELS:
        raise ValueError(f"Unknown model: {model!r} (have {sorted(MODELS)})")
    spec = MODELS[model]
    model_id = model_name or spec.model_id
    if num_inference_steps is None:
        num_inference_steps = spec.num_inference_steps
    if guidance_scale is None:
        guidance_scale = spec.guidance_scale

    outdir_init(output_dir, seed)
    to_generate = len(prompts) * variants_per_prompt
    print(f"Generating {to_generate} images with {model}")

    pipe = _build_pipeline(spec, model_id)
    _install_coupled_init(pipe)
    callback = None
    if eddy_config is not None:
        callback = _build_callback(pipe, spec, model, eddy_config, num_inference_steps)

    images_generated = 0
    for pi, prompt in enumerate(prompts):
        print(f"Prompt {pi}: {prompt[:80]}...")
        gen = torch.Generator(device=DEVICE).manual_seed(seed + pi * 1000)

        call_kwargs: dict[str, Any] = dict(
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=gen,
            num_images_per_prompt=variants_per_prompt,
            width=spec.resolution,
            height=spec.resolution,
        )
        if callback is not None:
            call_kwargs["callback_on_step_end"] = callback
            call_kwargs["callback_on_step_end_tensor_inputs"] = ["latents", "noise_pred"]

        result = pipe(**call_kwargs)

        for vi in range(variants_per_prompt):
            outdir_commit(
                output_dir, seed, _to_save_size(result.images[vi]),
                prompt_index=pi,
                variant_index=vi,
                prompt=prompt,
                variant_seed=seed + pi * 1000 + vi,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                model_name=model_id,
            )
            images_generated += 1

    print(f"Done. {images_generated} images saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="T2I EDDY",)
    parser.add_argument("--model", required=True, choices=sorted(MODELS), help="backbone to generate with")
    parser.add_argument("--diversity", required=True, choices=[*DIVERSITY_LEVELS, "iid"], help="EDDY diversity tier")
    parser.add_argument("--num-prompts", required=True, type=int, help="how many COCO val2017 captions to draw")
    args = parser.parse_args()

    if args.num_prompts < 1:
        parser.error("--num-prompts must be at least 1")

    output_dir = outdir_for(args.model, args.diversity)
    eddy_config = None if args.diversity == "iid" else default_eddy_config(args.model, args.diversity)
    print(f"Output: {output_dir}")
    print(f"Seed {SEED}, {VARIANTS_PER_PROMPT} variants per prompt")

    prompts = prompts_select_all(args.num_prompts, SEED)
    generate_images(
        prompts,
        output_dir,
        eddy_config,
        model=args.model,
        seed=SEED,
        variants_per_prompt=VARIANTS_PER_PROMPT,
    )


if __name__ == "__main__":
    main()
