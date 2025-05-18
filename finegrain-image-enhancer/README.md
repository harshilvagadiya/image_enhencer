# Finegrain Image Enhancer

A high-quality image enhancement API powered by FastAPI and ESRGAN (Enhanced Super-Resolution Generative Adversarial Network). This service allows you to enhance, upscale, and improve the quality of your images with customizable parameters.

## Features

- **AI-powered image enhancement** using state-of-the-art ESRGAN models
- **Up to 4x upscaling** for higher resolution outputs
- **Customizable enhancement parameters** for fine-tuning results
- **Multiple file format support** including JPG, PNG, WEBP, HEIF, and AVIF
- **Fast and reliable API** built with FastAPI
- **Tiling support** for processing larger images efficiently
- **Detail restoration** to recover and enhance fine details in images
- **GPU acceleration** for faster processing when available

## Requirements

- Python 3.10+
- PyTorch
- FastAPI
- Pillow with HEIF and AVIF support
- CUDA-capable GPU recommended (but will run on CPU)
- 8GB+ VRAM

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/finegrain-image-enhancer.git
   cd finegrain-image-enhancer
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   
   # On Windows
   .\venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the API server:
   ```bash
   uvicorn api:app --host 0.0.0.0 --port 8000
   ```

On first run, the application automatically downloads the required model checkpoints from HuggingFace Hub, including:
- ESRGAN upscaling model
- Juggernaut.reborn SD1.5 UNet
- Juggernaut.reborn SD1.5 text encoder
- ControlNet Tile model
- Necessary embeddings and LoRAs

## API Usage

### POST `/enhance/`

Enhances and upscales an uploaded image using AI models.

**Request Format:**
- Content-Type: `multipart/form-data`
- Required Field: `file` (image file to enhance)
- Optional Fields: Various parameters detailed in the parameters section

**Response:**
- Content-Type: `image/png`
- Body: The enhanced image data

**Example using cURL:**

```bash
curl -X POST "http://localhost:8000/enhance/" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/your/image.jpg" \
  -F "prompt=masterpiece, best quality, highres" \
  -F "upscale_factor=2" \
  -F "denoise_strength=0.35" \
  --output enhanced_image.png
```

**Example using Python:**

```python
import requests

url = "http://localhost:8000/enhance/"
files = {"file": open("examples/clarity_bird.webp", "rb")}
data = {
    "prompt": "masterpiece, best quality, highres",
    "negative_prompt": "worst quality, low quality",
    "upscale_factor": 2,
    "denoise_strength": 0.35
}

response = requests.post(url, files=files, data=data)
with open("enhanced_image.png", "wb") as f:
    f.write(response.content)
```

### API Documentation

The API also provides automatic documentation via FastAPI:

- Interactive API documentation: `http://localhost:8000/docs`
- OpenAPI specification: `http://localhost:8000/openapi.json`

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | string | "masterpiece, best quality, highres" | Text prompt to guide the enhancement process |
| `negative_prompt` | string | "worst quality, low quality, normal quality" | Text prompt describing what to avoid |
| `seed` | integer | 42 | Random seed for reproducible results |
| `upscale_factor` | integer | 2 | The factor by which to upscale the image (1-4) |
| `controlnet_scale` | float | 0.6 | Scale for the ControlNet influence |
| `controlnet_decay` | float | 1.0 | Decay rate for the ControlNet influence |
| `condition_scale` | integer | 6 | Scale for conditioning strength |
| `tile_width` | integer | 112 | Width of tiles for processing large images |
| `tile_height` | integer | 144 | Height of tiles for processing large images |
| `denoise_strength` | float | 0.35 | Strength of the denoising process (0.0-1.0) |
| `num_inference_steps` | integer | 18 | Number of denoising steps |
| `solver` | string | "DDIM" | The solver algorithm to use (DDIM, Euler, etc.) |

## How It Works

The Finegrain Image Enhancer uses a combination of technologies:

1. **ESRGAN** (Enhanced Super-Resolution GAN) for initial 4x upscaling
2. **ControlNet Tile** for maintaining spatial coherence across the image
3. **Refiners** for enhancing image details based on text prompts
4. **Tiling mechanism** for processing larger images efficiently

The process involves:
- Loading models from HuggingFace Hub
- Processing images with an optional tiling mechanism to handle larger images
- Applying enhancement using prompt-guided diffusion models
- Adjusting details based on the provided parameters

## Examples

The `examples` directory contains sample images you can use to test the enhancement service:

- `clarity_bird.webp` - An example from Clarity AI
- Various unsplash images showing different subjects and scenes:
  - Portraits
  - Landscapes
  - Animals
  - Urban scenes

The enhancement process will improve details, color, and overall image quality, especially noticeable in:
- Fine textures (fur, hair, feathers)
- Text clarity
- Facial features
- Architectural details

## License

This project is licensed under the terms included in the [LICENSE](LICENSE) file.

## Credits & Acknowledgments

This project builds upon several open-source projects:

- ESRGAN: Original implementation from [xinntao/ESRGAN](https://github.com/xinntao/ESRGAN)
- Enhanced adaptation from:
  - [victorca25/iNNfer](https://github.com/victorca25/iNNfer)
  - [AUTOMATIC1111/stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui)
  - [philz1337x/clarity-upscaler](https://github.com/philz1337x/clarity-upscaler)
- FastAPI: [tiangolo/fastapi](https://github.com/tiangolo/fastapi)
- HuggingFace Hub: [huggingface/huggingface_hub](https://github.com/huggingface/huggingface_hub)
- Refiners: For the latent diffusion components and multi-upscaler implementation

### Example Image Credits

- `clarity_bird.webp` by [Clarity AI](https://clarityai.co/)
- Unsplash images by various photographers (included under the [Unsplash License](https://unsplash.com/license)):
  - Kara Eads
  - Melissa Walker Horn
  - Karina Vorozheeva
  - Tadeusz Lakota
  - KaroGraphix Photography
  - Ryoji Iwata
  - Edgar.infocus
  - Jeremy Wallace
