import io
from pathlib import Path
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from PIL import Image
import torch
import asyncio
from concurrent.futures import ThreadPoolExecutor

from huggingface_hub import hf_hub_download
import pillow_heif
import pillow_avif  # just import, no register call

from src.enhancer import ESRGANUpscaler, ESRGANUpscalerCheckpoints
from refiners.foundationals.latent_diffusion import Solver, solvers

pillow_heif.register_heif_opener()

# Download and setup checkpoints (only happens once on startup)
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

# Setup device & dtype
DEVICE_CPU = torch.device("cpu")
DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
enhancer = ESRGANUpscaler(checkpoints=CHECKPOINTS, device=DEVICE_CPU, dtype=DTYPE)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
enhancer.to(device=DEVICE, dtype=DTYPE)

app = FastAPI()

# ThreadPoolExecutor for running CPU/GPU-heavy processing without blocking
executor = ThreadPoolExecutor(max_workers=1)  # adjust max_workers if you want concurrency

def process_api(input_image: Image.Image) -> Image.Image:
    prompt = "masterpiece, best quality, highres"
    negative_prompt = "worst quality, low quality, normal quality"
    seed = 42
    upscale_factor = 2
    controlnet_scale = 0.6
    controlnet_decay = 1.0
    condition_scale = 6
    tile_width = 112
    tile_height = 144
    denoise_strength = 0.35
    num_inference_steps = 18
    solver = "DDIM"

    solver_type = getattr(solvers, solver)
    generator = torch.Generator(device=DEVICE)
    generator.manual_seed(seed)

    side_size = min(input_image.size)
    if side_size > 768:
        scale = 768 / side_size
        new_size = (int(input_image.width * scale), int(input_image.height * scale))
        resized_image = input_image.resize(new_size, resample=Image.Resampling.LANCZOS)
    else:
        resized_image = input_image

    enhanced_image = enhancer.upscale(
        image=resized_image,
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
    return enhanced_image

@app.post("/enhance/")
async def enhance_image(file: UploadFile = File(...)):
    image_bytes = await file.read()
    input_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Run the heavy process_api in a thread to avoid blocking event loop
    loop = asyncio.get_event_loop()
    enhanced_image = await loop.run_in_executor(executor, process_api, input_image)

    buf = io.BytesIO()
    enhanced_image.save(buf, format="PNG")
    buf.seek(0)

    return StreamingResponse(buf, media_type="image/png")
