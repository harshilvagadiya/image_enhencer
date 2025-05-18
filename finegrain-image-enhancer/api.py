import io
import logging
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse
from PIL import Image
import torch
import pillow_heif
from src.enhancer import ESRGANUpscaler, ESRGANUpscalerCheckpoints
from huggingface_hub import hf_hub_download
from refiners.foundationals.latent_diffusion import Solver, solvers

pillow_heif.register_heif_opener()
pillow_heif.register_avif_opener()

app = FastAPI(title="Finegrain Image Enhancer API")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("enhancer_api")

# Load checkpoints once
CHECKPOINTS = ESRGANUpscalerCheckpoints(
    unet=Path(
        hf_hub_download(
            repo_id="refiners/juggernaut.reborn.sd1_5.unet",
            filename="model.safetensors",
            revision="347d14c3c782c4959cc4d1bb1e336d19f7dda4d2",
        )
    ),
    clip_text_encoder=Path(
        hf_hub_download(
            repo_id="refiners/juggernaut.reborn.sd1_5.text_encoder",
            filename="model.safetensors",
            revision="744ad6a5c0437ec02ad826df9f6ede102bb27481",
        )
    ),
    lda=Path(
        hf_hub_download(
            repo_id="refiners/juggernaut.reborn.sd1_5.autoencoder",
            filename="model.safetensors",
            revision="3c1aae3fc3e03e4a2b7e0fa42b62ebb64f1a4c19",
        )
    ),
    controlnet_tile=Path(
        hf_hub_download(
            repo_id="refiners/controlnet.sd1_5.tile",
            filename="model.safetensors",
            revision="48ced6ff8bfa873a8976fa467c3629a240643387",
        )
    ),
    esrgan=Path(
        hf_hub_download(
            repo_id="philz1337x/upscaler",
            filename="4x-UltraSharp.pth",
            revision="011deacac8270114eb7d2eeff4fe6fa9a837be70",
        )
    ),
    negative_embedding=Path(
        hf_hub_download(
            repo_id="philz1337x/embeddings",
            filename="JuggernautNegative-neg.pt",
            revision="203caa7e9cc2bc225031a4021f6ab1ded283454a",
        )
    ),
    negative_embedding_key="string_to_param.*",
    loras={
        "more_details": Path(
            hf_hub_download(
                repo_id="philz1337x/loras",
                filename="more_details.safetensors",
                revision="a3802c0280c0d00c2ab18d37454a8744c44e474e",
            )
        ),
        "sdxl_render": Path(
            hf_hub_download(
                repo_id="philz1337x/loras",
                filename="SDXLrender_v2.0.safetensors",
                revision="a3802c0280c0d00c2ab18d37454a8744c44e474e",
            )
        ),
    },
)

# Setup device and dtype
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32

# Initialize enhancer
enhancer = ESRGANUpscaler(checkpoints=CHECKPOINTS, device=DEVICE, dtype=DTYPE)
enhancer.to(device=DEVICE, dtype=DTYPE)

def enhance_image(
    image: Image.Image,
    prompt: str = "masterpiece, best quality, highres",
    negative_prompt: str = "worst quality, low quality, normal quality",
    seed: int = 42,
    upscale_factor: int = 2,
    controlnet_scale: float = 0.6,
    controlnet_decay: float = 1.0,
    condition_scale: int = 6,
    tile_width: int = 112,
    tile_height: int = 144,
    denoise_strength: float = 0.35,
    num_inference_steps: int = 18,
    solver: str = "DDIM",
) -> Image.Image:
    solver_type: type[Solver] = getattr(solvers, solver)

    generator = torch.Generator(device=DEVICE)
    generator.manual_seed(seed)

    side_size = min(image.size)
    if side_size > 768:
        scale = 768 / side_size
        new_size = (int(image.width * scale), int(image.height * scale))
        image = image.resize(new_size, resample=Image.Resampling.LANCZOS)

    enhanced = enhancer.upscale(
        image=image,
        prompt=prompt,
        negative_prompt=negative_prompt,
        upscale_factor=upscale_factor,
        controlnet_scale=controlnet_scale,
        controlnet_scale_decay=controlnet_decay,
        condition_scale=condition_scale,
        tile_size=(tile_height, tile_width),
        denoise_strength=denoise_strength,
        num_inference_steps=num_inference_steps,
        loras_scale={"more_details": 0.5, "sdxl_render": 1.0},
        solver_type=solver_type,
        generator=generator,
    )
    return enhanced

@app.post("/enhance/")
async def enhance(
    file: UploadFile = File(...),
    prompt: str = Form("masterpiece, best quality, highres"),
    negative_prompt: str = Form("worst quality, low quality, normal quality"),
    seed: int = Form(42),
    upscale_factor: int = Form(2),
    controlnet_scale: float = Form(0.6),
    controlnet_decay: float = Form(1.0),
    condition_scale: int = Form(6),
    tile_width: int = Form(112),
    tile_height: int = Form(144),
    denoise_strength: float = Form(0.35),
    num_inference_steps: int = Form(18),
    solver: str = Form("DDIM"),
):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        logger.info(f"Received image {file.filename} of size {image.size}")

        enhanced_image = enhance_image(
            image=image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=seed,
            upscale_factor=upscale_factor,
            controlnet_scale=controlnet_scale,
            controlnet_decay=controlnet_decay,
            condition_scale=condition_scale,
            tile_width=tile_width,
            tile_height=tile_height,
            denoise_strength=denoise_strength,
            num_inference_steps=num_inference_steps,
            solver=solver,
        )

        buf = io.BytesIO()
        enhanced_image.save(buf, format="PNG")
        buf.seek(0)

        logger.info("Image enhancement successful")
        return StreamingResponse(buf, media_type="image/png")

    except Exception as e:
        logger.error(f"Error processing image: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Image enhancement failed: {e}")
