"""
SigLIP calibration utility.

What it does:
- Converts a similarity score between two embeddings into probability.
- Optionally uses SigLIP's pretrained calibration parameters from a checkpoint.
- Can also fit calibration parameters from labeled data.

How it works:
- Uses logistic calibration: probability = sigmoid(scale * similarity + bias).
- `predict` mode applies known (`scale`, `bias`) to new similarities.
- `fit` mode learns (`scale`, `bias`) with binary cross-entropy (Platt scaling).
"""

import argparse
import csv
import json
from pathlib import Path

import torch

from nanosiglip.siglip import SigLIP
from nanosiglip.siglip.calibration import fit_platt_scaling, similarity_to_probability


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SigLIP similarity calibration utility (similarity -> probability).",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    predict = subparsers.add_parser("predict", help="Convert similarity to probability")
    predict.add_argument(
        "--similarity",
        type=float,
        required=True,
        help="Cosine similarity between two vectors",
    )
    predict.add_argument("--scale", type=float, default=None, help="Calibration scale")
    predict.add_argument("--bias", type=float, default=None, help="Calibration bias")
    predict.add_argument(
        "--model",
        type=str,
        default=None,
        help="Load SigLIP logit_scale/logit_bias from this model id or local path",
    )

    fit = subparsers.add_parser(
        "fit", help="Fit calibration params from CSV with columns: similarity,label"
    )
    fit.add_argument("--data", type=Path, required=True, help="Path to CSV file")
    fit.add_argument("--steps", type=int, default=2000, help="Optimization steps")
    fit.add_argument("--lr", type=float, default=0.05, help="Learning rate")
    fit.add_argument(
        "--use-log-scale",
        action="store_true",
        help="Use SigLIP-style scale parameterization: scale = exp(logit_scale)",
    )
    fit.add_argument("--out", type=Path, default=None, help="Optional JSON output path")

    return parser.parse_args()


def run_predict(args: argparse.Namespace) -> None:
    if args.model is not None:
        model = SigLIP.from_pretrained(args.model)
        scale = float(torch.exp(model.logit_scale.detach().cpu()).item())
        bias = float(model.logit_bias.detach().cpu().item())
    else:
        scale = 1.0 if args.scale is None else float(args.scale)
        bias = 0.0 if args.bias is None else float(args.bias)

    prob = similarity_to_probability(args.similarity, scale=scale, bias=bias).item()
    logit = args.similarity * scale + bias

    print(f"similarity={args.similarity:.6f}")
    print(f"scale={scale:.6f}")
    print(f"bias={bias:.6f}")
    print(f"logit={logit:.6f}")
    print(f"probability={prob:.6f}")


def run_fit(args: argparse.Namespace) -> None:
    similarities: list[float] = []
    labels: list[float] = []

    with args.data.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "similarity" not in reader.fieldnames or "label" not in reader.fieldnames:
            raise ValueError("CSV must contain columns: similarity,label")
        for row in reader:
            similarities.append(float(row["similarity"]))
            labels.append(float(row["label"]))

    params = fit_platt_scaling(
        similarities=torch.tensor(similarities, dtype=torch.float32),
        labels=torch.tensor(labels, dtype=torch.float32),
        steps=args.steps,
        lr=args.lr,
        use_log_scale=args.use_log_scale,
    )

    result = {
        "scale": params.scale,
        "bias": params.bias,
        "use_log_scale": bool(args.use_log_scale),
    }
    print(json.dumps(result, indent=2))

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.command == "predict":
        run_predict(args)
    elif args.command == "fit":
        run_fit(args)
    else:
        raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
