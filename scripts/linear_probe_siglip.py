from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from datasets import Dataset, DatasetDict, load_dataset
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from nanosiglip.siglip import SigLIP, SigLIPImageProcessor, SigLIPTextProcessor


@dataclass
class DatasetSchema:
    image_col: str
    label_col: str
    class_names: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SigLIP zero-shot + linear-probe classification"
    )
    parser.add_argument(
        "--model",
        default="google/siglip-base-patch16-224",
        help="Base SigLIP checkpoint (HF id or local path)",
    )
    parser.add_argument(
        "--dataset",
        default="Bingsu/Human_Action_Recognition",
        help="Hugging Face dataset id for image classification",
    )
    parser.add_argument(
        "--train-split",
        default="train",
        help="Dataset split used as source training data",
    )
    parser.add_argument(
        "--eval-split",
        default="validation",
        help="Eval split used only when --no-split-train-for-eval is set",
    )
    parser.add_argument(
        "--split-train-for-eval",
        action="store_true",
        default=True,
        help="Split train split into train/eval (recommended for datasets with bad test labels)",
    )
    parser.add_argument(
        "--no-split-train-for-eval",
        action="store_false",
        dest="split_train_for_eval",
        help="Disable train/eval split and use --eval-split directly",
    )
    parser.add_argument(
        "--eval-size",
        type=float,
        default=0.2,
        help="Fraction of train split used for eval when --split-train-for-eval is enabled",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for feature extraction and probe train/eval",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of probe training epochs",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="Learning rate for AdamW probe optimizer",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="Weight decay for AdamW probe optimizer",
    )
    parser.add_argument(
        "--max-train-samples",
        help="Number of train samples. 0 for all",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--max-eval-samples",
        help="Number of eval samples. 0 for all",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for split/training reproducibility",
    )
    parser.add_argument(
        "--prompt-template",
        default="a photo of {}",
        help="Prompt template for zero-shot labels; include '{}' placeholder",
    )
    parser.add_argument(
        "--init-from-zeroshot",
        action="store_true",
        default=True,
        help="Initialize probe from zero-shot text embeddings (recommended)",
    )
    parser.add_argument(
        "--no-init-from-zeroshot",
        action="store_false",
        dest="init_from_zeroshot",
        help="Disable zero-shot initialization",
    )
    parser.add_argument(
        "--keep-best-checkpoint",
        action="store_true",
        default=True,
        help="Track/restore best eval-accuracy checkpoint during probe training",
    )
    parser.add_argument(
        "--no-keep-best-checkpoint",
        action="store_false",
        dest="keep_best_checkpoint",
        help="Use final epoch checkpoint instead of best eval checkpoint",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device (cpu, cuda, cuda:0, ...)",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_split(ds: DatasetDict, preferred: str, fallback: str) -> str:
    if preferred in ds:
        return preferred
    if fallback in ds:
        return fallback
    if ds:
        return next(iter(ds.keys()))
    raise ValueError("Dataset has no splits")


def infer_schema(dataset: Dataset) -> DatasetSchema:
    image_candidates = ["image", "img", "pixel_values"]
    label_candidates = ["label", "labels", "target", "category", "class", "action"]

    image_col = next((c for c in image_candidates if c in dataset.column_names), None)
    label_col = next((c for c in label_candidates if c in dataset.column_names), None)

    if image_col is None:
        raise ValueError(
            f"Could not infer image column. Found columns: {dataset.column_names}"
        )
    if label_col is None:
        raise ValueError(
            f"Could not infer label column. Found columns: {dataset.column_names}"
        )

    feature = dataset.features.get(label_col)
    class_names: list[str]
    if feature is not None and hasattr(feature, "names") and feature.names is not None:
        class_names = [str(x) for x in feature.names]
    else:
        labels = dataset[label_col]
        unique_labels = sorted(set(int(x) for x in labels))
        class_names = [str(x) for x in unique_labels]

    return DatasetSchema(
        image_col=image_col, label_col=label_col, class_names=class_names
    )


def _to_label_id(raw_label: Any, num_classes: int) -> int:
    idx = int(raw_label)
    if idx < 0 or idx >= num_classes:
        raise ValueError(f"Label index out of range: {idx}")
    return idx


def label_cardinality(dataset: Dataset, label_col: str) -> int:
    return len(set(int(x) for x in dataset[label_col]))


def _make_prompt(name: str, template: str) -> str:
    clean = name.replace("_", " ").replace("-", " ").strip()
    return template.format(clean)


@torch.no_grad()
def encode_text_prompts(
    model: SigLIP,
    text_processor: SigLIPTextProcessor,
    class_names: list[str],
    prompt_template: str,
    device: str,
) -> torch.Tensor:
    prompts = [_make_prompt(name, prompt_template) for name in class_names]
    batch = text_processor(
        prompts,
        padding="max_length",
        truncation=True,
        max_length=model.config.text_config.max_position_embeddings,
        return_tensors="pt",
    )
    text_out = model.get_text_features(input_ids=batch.input_ids.to(device))
    text_embeds = F.normalize(text_out.pooler_output, dim=-1)
    return text_embeds


def collate_images(
    batch: list[dict[str, Any]],
    image_col: str,
    label_col: str,
    processor: SigLIPImageProcessor,
    ncls: int,
):
    images = []
    labels = []
    for ex in batch:
        image = ex[image_col]
        if hasattr(image, "convert"):
            image = image.convert("RGB")
        images.append(image)
        labels.append(_to_label_id(ex[label_col], ncls))

    pixel_values = processor(images=images, return_tensors="pt").pixel_values
    return pixel_values, torch.tensor(labels, dtype=torch.long)


@torch.no_grad()
def zero_shot_accuracy(
    model: SigLIP,
    dataloader: DataLoader,
    text_embeds: torch.Tensor,
    device: str,
) -> float:
    model.eval()
    correct = 0
    total = 0

    for pixel_values, labels in dataloader:
        pixel_values = pixel_values.to(device)
        labels = labels.to(device)

        image_out = model.get_image_features(pixel_values=pixel_values)
        image_embeds = F.normalize(image_out.pooler_output, dim=-1)

        logits = image_embeds @ text_embeds.t()
        preds = logits.argmax(dim=-1)

        correct += int((preds == labels).sum().item())
        total += labels.size(0)

    return correct / max(total, 1)


@torch.no_grad()
def extract_image_features(
    model: SigLIP, dataloader: DataLoader, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    all_feats: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    for pixel_values, labels in tqdm(dataloader, desc="extract_features"):
        pixel_values = pixel_values.to(device)
        out = model.get_image_features(pixel_values=pixel_values)
        feats = F.normalize(out.pooler_output, dim=-1).cpu()

        all_feats.append(feats)
        all_labels.append(labels)

    return torch.cat(all_feats, dim=0), torch.cat(all_labels, dim=0)


@torch.no_grad()
def evaluate_linear_probe(
    head: nn.Module,
    feats: torch.Tensor,
    labels: torch.Tensor,
    device: str,
    batch_size: int,
) -> float:
    head.eval()
    ds = TensorDataset(feats, labels)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)

    correct = 0
    total = 0
    for x, y in dl:
        x = x.to(device)
        y = y.to(device)
        preds = head(x).argmax(dim=-1)
        correct += int((preds == y).sum().item())
        total += y.size(0)
    return correct / max(total, 1)


def train_linear_probe(
    train_feats: torch.Tensor,
    train_labels: torch.Tensor,
    eval_feats: torch.Tensor,
    eval_labels: torch.Tensor,
    num_classes: int,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    device: str,
    init_weights: torch.Tensor | None = None,
    keep_best_checkpoint: bool = True,
) -> tuple[nn.Module, dict[str, float]]:
    head = nn.Linear(train_feats.shape[1], num_classes).to(device)
    if init_weights is not None:
        expected = (num_classes, train_feats.shape[1])
        if tuple(init_weights.shape) != expected:
            raise ValueError(
                f"init_weights must have shape {expected}, got {tuple(init_weights.shape)}"
            )
        with torch.no_grad():
            head.weight.copy_(init_weights.to(device))
            head.bias.zero_()

    optimizer = AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)

    train_ds = TensorDataset(train_feats, train_labels)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    best_eval_acc = float("-inf")
    best_state: dict[str, torch.Tensor] | None = None

    for epoch in range(epochs):
        head.train()
        running_loss = 0.0
        count = 0
        for x, y in train_dl:
            x = x.to(device)
            y = y.to(device)

            logits = head(x)
            loss = F.cross_entropy(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item()) * y.size(0)
            count += y.size(0)

        avg_loss = running_loss / max(count, 1)
        eval_acc = evaluate_linear_probe(
            head, eval_feats, eval_labels, device=device, batch_size=batch_size
        )
        print(f"epoch={epoch + 1}/{epochs} loss={avg_loss:.4f} eval_acc={eval_acc:.4f}")
        if keep_best_checkpoint and eval_acc > best_eval_acc:
            best_eval_acc = eval_acc
            best_state = {
                k: v.detach().cpu().clone() for k, v in head.state_dict().items()
            }

    if keep_best_checkpoint and best_state is not None:
        head.load_state_dict(best_state)

    metrics = {
        "train_acc": evaluate_linear_probe(
            head, train_feats, train_labels, device=device, batch_size=batch_size
        ),
        "eval_acc": evaluate_linear_probe(
            head, eval_feats, eval_labels, device=device, batch_size=batch_size
        ),
    }
    return head, metrics


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    dataset = load_dataset(args.dataset)
    train_split = choose_split(dataset, args.train_split, "train")
    base_train_ds = dataset[train_split]
    schema = infer_schema(base_train_ds)

    if args.split_train_for_eval:
        try:
            split_ds = base_train_ds.train_test_split(
                test_size=args.eval_size,
                seed=args.seed,
                stratify_by_column=schema.label_col,
            )
        except Exception:
            split_ds = base_train_ds.train_test_split(
                test_size=args.eval_size,
                seed=args.seed,
            )
        train_ds = split_ds["train"]
        eval_ds = split_ds["test"]
        print(
            f"Using train holdout split: train={len(train_ds)} eval={len(eval_ds)} "
            f"(eval_size={args.eval_size})"
        )
    else:
        eval_split = choose_split(dataset, args.eval_split, "validation")
        train_ds = base_train_ds
        eval_ds = dataset[eval_split]
        if label_cardinality(eval_ds, schema.label_col) <= 1:
            print(
                "Warning: eval split has <=1 unique labels; falling back to train holdout split."
            )
            split_ds = base_train_ds.train_test_split(
                test_size=args.eval_size,
                seed=args.seed,
            )
            train_ds = split_ds["train"]
            eval_ds = split_ds["test"]

    if args.max_train_samples > 0:
        train_ds = train_ds.select(range(min(args.max_train_samples, len(train_ds))))
    if args.max_eval_samples > 0:
        eval_ds = eval_ds.select(range(min(args.max_eval_samples, len(eval_ds))))

    num_classes = len(schema.class_names)

    model, model_path = SigLIP.from_pretrained(args.model, return_model_path=True)
    model.to(args.device)
    model.eval()

    image_processor = SigLIPImageProcessor.from_pretrained(model_path)
    text_processor = SigLIPTextProcessor.from_pretrained(model_path)

    def collate(batch):
        return collate_images(
            batch, schema.image_col, schema.label_col, image_processor, num_classes
        )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate,
    )
    eval_loader = DataLoader(
        eval_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate,
    )

    text_embeds = encode_text_prompts(
        model=model,
        text_processor=text_processor,
        class_names=schema.class_names,
        prompt_template=args.prompt_template,
        device=args.device,
    )

    zs_train_acc = zero_shot_accuracy(
        model, train_loader, text_embeds, device=args.device
    )
    zs_eval_acc = zero_shot_accuracy(
        model, eval_loader, text_embeds, device=args.device
    )

    print("Zero-shot metrics:")
    print(json.dumps({"train_acc": zs_train_acc, "eval_acc": zs_eval_acc}, indent=2))

    train_feats, train_labels = extract_image_features(
        model, train_loader, device=args.device
    )
    eval_feats, eval_labels = extract_image_features(
        model, eval_loader, device=args.device
    )

    init_weights = text_embeds if args.init_from_zeroshot else None
    if init_weights is not None:
        baseline_head = nn.Linear(train_feats.shape[1], num_classes).to(args.device)
        with torch.no_grad():
            baseline_head.weight.copy_(init_weights.to(args.device))
            baseline_head.bias.zero_()
        probe_init_eval_acc = evaluate_linear_probe(
            baseline_head,
            eval_feats,
            eval_labels,
            device=args.device,
            batch_size=args.batch_size,
        )
        print(f"Probe init (from zero-shot) eval_acc={probe_init_eval_acc:.4f}")

    _, probe_metrics = train_linear_probe(
        train_feats=train_feats,
        train_labels=train_labels,
        eval_feats=eval_feats,
        eval_labels=eval_labels,
        num_classes=num_classes,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=args.device,
        init_weights=init_weights,
        keep_best_checkpoint=args.keep_best_checkpoint,
    )

    print("Linear probe metrics:")
    print(json.dumps(probe_metrics, indent=2))

    summary = {
        "dataset": args.dataset,
        "model": args.model,
        "num_classes": num_classes,
        "zero_shot": {"train_acc": zs_train_acc, "eval_acc": zs_eval_acc},
        "linear_probe": probe_metrics,
    }
    print("Summary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
