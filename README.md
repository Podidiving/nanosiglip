# nanosiglip

Minimal, pure-PyTorch SigLIP implementation focused on compatibility with Google SigLIP checkpoints from Hugging Face.

## Goals
- Keep runtime dependencies small (`torch`, `safetensors`, `sentencepiece`, `Pillow` for image loading).
- Load original SigLIP weights directly from Hugging Face.
- Match `transformers` outputs for preprocessing and inference.

## Repository Layout
- `src/nanosiglip/siglip/`: SigLIP implementation and preprocessors.
- `scripts/run_siglip.py`: End-to-end inference example for one image and multiple text prompts.
- `tests/`: parity tests against `transformers` (model, image preprocessing, text preprocessing, and e2e pipeline).

## Installation
```bash
uv sync --dev
```

## Usage
Run pure-torch SigLIP inference:

```bash
uv run python scripts/run_siglip.py assets/image.jpg "a cat" "a dog" "a landscape"
```

Optional flags:
- `--model` (default: `google/siglip-base-patch16-224`)
- `--revision` (default: `main`)
- `--device` (default: auto: `cuda` if available, else `cpu`)

## LoRA Fine-Tuning
Train LoRA adapters (default dataset: `nlphuji/flickr30k`):

```bash
uv run --group train python scripts/train_siglip_lora.py
```

The training script:
- prints evaluation metrics **before training** and **after training**
- writes TensorBoard logs to `runs/siglip_lora` (configurable via `--log-dir`)
- saves LoRA weights to `outputs/siglip_lora/siglip_lora.pt`

Start TensorBoard:

```bash
uv run --group train tensorboard --logdir runs/siglip_lora
```

Run inference with LoRA weights:

```bash
uv run python scripts/inference_siglip_lora.py assets/image.jpg \"a cat\" \"a dog\" --lora-path outputs/siglip_lora/siglip_lora.pt
```

## Linear Probe (Classification)
Run zero-shot classification first, then train a linear probe on top of frozen SigLIP image embeddings:

```bash
uv run --group train python scripts/linear_probe_siglip.py
```

Defaults:
- dataset: `Bingsu/Human_Action_Recognition`
- model: `google/siglip-base-patch16-224`
- evaluation: by default the script splits `train` into train/eval (`--split-train-for-eval`)

The script reports:
- zero-shot accuracy (`train_acc`, `eval_acc`)
- linear-probe accuracy after training (`train_acc`, `eval_acc`)

## Calibration Utility
Convert similarity score to probability:

```bash
uv run python scripts/siglip_calibration.py predict --similarity 0.42 --model google/siglip-base-patch16-224
```

Or provide custom calibration parameters:

```bash
uv run python scripts/siglip_calibration.py predict --similarity 0.42 --scale 10.0 --bias -5.0
```

Fit calibration (`scale`, `bias`) from labeled data:

```bash
uv run python scripts/siglip_calibration.py fit --data data/similarity_labels.csv --out calibration.json
```

CSV format:
- `similarity`: cosine similarity between image/text embeddings
- `label`: binary target (`0` or `1`)

Example Python usage:

```python
import torch
from PIL import Image
from nanosiglip.siglip import SigLIP, SigLIPImageProcessor, SigLIPTextProcessor

model, model_path = SigLIP.from_pretrained("google/siglip-base-patch16-224", return_model_path=True)
image_processor = SigLIPImageProcessor.from_pretrained(model_path)
text_processor = SigLIPTextProcessor.from_pretrained(model_path)

image = Image.open("assets/image.jpg")
texts = ["a cat", "a dog", "a landscape"]

image_inputs = image_processor(images=image, return_tensors="pt")
text_inputs = text_processor(texts, padding="max_length", truncation=True, max_length=64, return_tensors="pt")

with torch.no_grad():
    outputs = model(
        input_ids=text_inputs.input_ids,
        pixel_values=image_inputs.pixel_values,
    )
```

## Testing
Run all tests:

```bash
uv run pytest -q
```

The suite checks parity with `transformers` and ensures e2e inference matches on `assets/image.jpg`.
