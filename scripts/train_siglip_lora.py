from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset, DatasetDict, load_dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from nanosiglip.siglip import SigLIPImageProcessor, SigLIPLoRA, SigLIPTextProcessor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LoRA adapters for SigLIP")
    parser.add_argument("--model", default="google/siglip-base-patch16-224")
    parser.add_argument("--dataset", default="nlphuji/flickr30k")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-split", default="validation")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-train-samples", type=int, default=1024)
    parser.add_argument("--max-eval-samples", type=int, default=256)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=16.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument(
        "--target-modules",
        nargs="+",
        default=["q_proj", "k_proj", "v_proj", "out_proj"],
        help="Linear module names to LoRA-wrap",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-dir", type=Path, default=Path("runs/siglip_lora"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/siglip_lora"))
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_split(ds: DatasetDict, preferred: str) -> str:
    if preferred in ds:
        return preferred
    for fallback in ("validation", "test", "train"):
        if fallback in ds:
            return fallback
    raise ValueError(f"No usable split found. Available: {list(ds.keys())}")


def get_image(example: dict[str, Any]):
    for key in ("image", "img"):
        if key in example:
            image = example[key]
            if hasattr(image, "convert"):
                return image.convert("RGB")
            return image
    raise KeyError("Could not find image column (expected 'image' or 'img')")


def get_caption(example: dict[str, Any]) -> str:
    for key in ("caption", "captions", "sentence", "sentences", "text"):
        if key not in example:
            continue
        value = example[key]
        if isinstance(value, str):
            return value
        if isinstance(value, list) and value:
            for item in value:
                if isinstance(item, str):
                    return item
                if isinstance(item, dict):
                    for nested_key in ("raw", "caption", "text", "sentence"):
                        if nested_key in item and isinstance(item[nested_key], str):
                            return item[nested_key]
    raise KeyError("Could not find caption text in example")


def make_collate_fn(
    image_processor: SigLIPImageProcessor,
    text_processor: SigLIPTextProcessor,
    max_length: int,
):
    def collate(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        images = [get_image(example) for example in batch]
        texts = [get_caption(example) for example in batch]

        pixel_values = image_processor(images=images, return_tensors="pt").pixel_values
        text_batch = text_processor(
            texts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        return {
            "pixel_values": pixel_values,
            "input_ids": text_batch.input_ids,
        }

    return collate


@torch.no_grad()
def evaluate(
    model: SigLIPLoRA, dataloader: DataLoader, device: str
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_count = 0
    i2t_correct = 0
    t2i_correct = 0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        pixel_values = batch["pixel_values"].to(device)

        outputs = model(
            input_ids=input_ids, pixel_values=pixel_values, return_loss=True
        )
        bsz = input_ids.shape[0]

        total_loss += float(outputs.loss.item()) * bsz
        total_count += bsz

        targets = torch.arange(bsz, device=device)
        i2t_correct += int(
            (outputs.logits_per_image.argmax(dim=-1) == targets).sum().item()
        )
        t2i_correct += int(
            (outputs.logits_per_text.argmax(dim=-1) == targets).sum().item()
        )

    return {
        "loss": total_loss / max(total_count, 1),
        "image_to_text_top1": i2t_correct / max(total_count, 1),
        "text_to_image_top1": t2i_correct / max(total_count, 1),
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    model = SigLIPLoRA.from_pretrained(
        args.model,
        rank=args.rank,
        alpha=args.alpha,
        dropout=args.dropout,
        target_modules=tuple(args.target_modules),
    )
    model.to(args.device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = model.trainable_parameters()
    trainable_param_count = sum(p.numel() for p in trainable_params)
    trainable_pct = 100.0 * trainable_param_count / max(total_params, 1)
    print(
        "Parameter counts: "
        f"total={total_params:,} "
        f"trainable={trainable_param_count:,} "
        f"trainable_pct={trainable_pct:.4f}%"
    )

    model_path = model.model.model_path
    if model_path is None:
        raise RuntimeError("Base model path is not available after from_pretrained")

    image_processor = SigLIPImageProcessor.from_pretrained(model_path)
    text_processor = SigLIPTextProcessor.from_pretrained(model_path)

    ds = load_dataset(args.dataset)
    train_split = pick_split(ds, args.train_split)
    eval_split = pick_split(ds, args.eval_split)

    train_ds: Dataset = ds[train_split]
    eval_ds: Dataset = ds[eval_split]

    if args.max_train_samples > 0:
        train_ds = train_ds.select(range(min(args.max_train_samples, len(train_ds))))
    if args.max_eval_samples > 0:
        eval_ds = eval_ds.select(range(min(args.max_eval_samples, len(eval_ds))))

    collate_fn = make_collate_fn(
        image_processor,
        text_processor,
        model.model.config.text_config.max_position_embeddings,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
    )
    eval_loader = DataLoader(
        eval_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    optimizer = AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

    args.log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(args.log_dir))

    before_metrics = evaluate(model, eval_loader, args.device)
    print("Before training metrics:")
    print(json.dumps(before_metrics, indent=2))
    for key, value in before_metrics.items():
        writer.add_scalar(f"eval_before/{key}", value, 0)

    global_step = 0
    model.train()
    for epoch in range(args.epochs):
        progress = tqdm(train_loader, desc=f"epoch {epoch + 1}/{args.epochs}")
        for batch in progress:
            input_ids = batch["input_ids"].to(args.device)
            pixel_values = batch["pixel_values"].to(args.device)

            outputs = model(
                input_ids=input_ids, pixel_values=pixel_values, return_loss=True
            )
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            global_step += 1
            loss_value = float(loss.item())
            progress.set_postfix(loss=f"{loss_value:.4f}")
            writer.add_scalar("train/loss", loss_value, global_step)

        epoch_metrics = evaluate(model, eval_loader, args.device)
        print(f"Epoch {epoch + 1} eval metrics:")
        print(json.dumps(epoch_metrics, indent=2))
        for key, value in epoch_metrics.items():
            writer.add_scalar(f"eval_epoch/{key}", value, epoch + 1)

        model.train()

    after_metrics = evaluate(model, eval_loader, args.device)
    print("After training metrics:")
    print(json.dumps(after_metrics, indent=2))
    for key, value in after_metrics.items():
        writer.add_scalar(f"eval_after/{key}", value, global_step)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    lora_path = args.output_dir / "siglip_lora.pt"
    model.save_lora_weights(lora_path)

    metrics_path = args.output_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps({"before": before_metrics, "after": after_metrics}, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Saved LoRA weights: {lora_path}")
    print(f"Saved metrics: {metrics_path}")

    writer.close()


if __name__ == "__main__":
    main()
