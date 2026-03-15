from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image

from nanosiglip.siglip import SigLIPImageProcessor, SigLIPLoRA, SigLIPTextProcessor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SigLIP inference with optional LoRA adapters"
    )
    parser.add_argument("image_path", type=Path)
    parser.add_argument("texts", nargs="+")
    parser.add_argument("--model", default="google/siglip-base-patch16-224")
    parser.add_argument(
        "--lora-path", type=Path, default=None, help="Path to saved LoRA weights (.pt)"
    )
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=16.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument(
        "--target-modules",
        nargs="+",
        default=["q_proj", "k_proj", "v_proj", "out_proj"],
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.image_path.exists():
        raise FileNotFoundError(f"Image not found: {args.image_path}")

    model = SigLIPLoRA.from_pretrained(
        args.model,
        rank=args.rank,
        alpha=args.alpha,
        dropout=args.dropout,
        target_modules=tuple(args.target_modules),
    )

    if args.lora_path is not None:
        state_dict = torch.load(args.lora_path, map_location="cpu")
        model.load_lora_state_dict(state_dict, strict=True)

    model.to(args.device)
    model.eval()

    model_path = model.model.model_path
    if model_path is None:
        raise RuntimeError("Base model path is unavailable")

    image_processor = SigLIPImageProcessor.from_pretrained(model_path)
    text_processor = SigLIPTextProcessor.from_pretrained(model_path)

    image = Image.open(args.image_path).convert("RGB")

    image_batch = image_processor(images=image, return_tensors="pt")
    text_batch = text_processor(
        args.texts,
        padding="max_length",
        truncation=True,
        max_length=model.model.config.text_config.max_position_embeddings,
        return_tensors="pt",
    )

    with torch.no_grad():
        outputs = model(
            input_ids=text_batch.input_ids.to(args.device),
            pixel_values=image_batch.pixel_values.to(args.device),
        )

    probs = torch.sigmoid(outputs.logits_per_image[0]).cpu()
    logits = outputs.logits_per_image[0].cpu()

    print(f"Model: {args.model}")
    print(f"Image: {args.image_path}")
    print(f"LoRA: {args.lora_path if args.lora_path is not None else 'none'}")
    print("\nText scores:")
    for idx, (text, logit, prob) in enumerate(zip(args.texts, logits, probs), start=1):
        print(f"{idx:>2}. p={float(prob):.5f} logit={float(logit):.3f} text={text!r}")


if __name__ == "__main__":
    main()
