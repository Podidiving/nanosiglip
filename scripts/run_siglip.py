from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image

from nanosiglip.siglip import SigLIP, SigLIPImageProcessor, SigLIPTextProcessor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SigLIP scoring for one image against one or more text prompts.",
    )
    parser.add_argument("image_path", type=Path, help="Path to input image")
    parser.add_argument("texts", nargs="+", help="One or more text prompts")
    parser.add_argument(
        "--model",
        default="google/siglip-base-patch16-224",
        help="Hugging Face model id or local directory",
    )
    parser.add_argument("--revision", default="main", help="Model revision")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device (e.g. cpu, cuda, cuda:0)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.image_path.exists():
        raise FileNotFoundError(f"Image file not found: {args.image_path}")

    model, model_path = SigLIP.from_pretrained(
        args.model,
        revision=args.revision,
        return_model_path=True,
    )
    model = model.to(args.device)
    model.eval()

    image_processor = SigLIPImageProcessor.from_pretrained(model_path)
    text_processor = SigLIPTextProcessor.from_pretrained(model_path)

    image = Image.open(args.image_path)

    image_batch = image_processor(images=image, return_tensors="pt")
    text_batch = text_processor(
        args.texts,
        padding="max_length",
        truncation=True,
        max_length=model.config.text_config.max_position_embeddings,
        return_tensors="pt",
    )

    pixel_values = image_batch.pixel_values.to(args.device)
    input_ids = text_batch.input_ids.to(args.device)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            pixel_values=pixel_values,
        )

    # SigLIP is trained with sigmoid multi-label scoring.
    probs = torch.sigmoid(outputs.logits_per_image[0]).cpu()
    logits = outputs.logits_per_image[0].cpu()

    print(f"Model path: {model_path}")
    print(f"Image: {args.image_path}")
    print("\nText scores:")
    for idx, (text, logit, prob) in enumerate(zip(args.texts, logits, probs), start=1):
        print(f"{idx:>2}. p={float(prob):.5f} logit={float(logit):.3f} text={text!r}")


def _entrypoint() -> None:
    main()


if __name__ == "__main__":
    _entrypoint()
