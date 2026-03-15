from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class CalibrationParams:
    scale: float
    bias: float


def similarity_to_probability(
    similarity: torch.Tensor | float,
    *,
    scale: float,
    bias: float,
) -> torch.Tensor:
    similarity_tensor = torch.as_tensor(similarity, dtype=torch.float32)
    logits = similarity_tensor * float(scale) + float(bias)
    return torch.sigmoid(logits)


def siglip_similarity_to_probability(
    similarity: torch.Tensor | float,
    *,
    logit_scale: float,
    logit_bias: float,
) -> torch.Tensor:
    return similarity_to_probability(
        similarity,
        scale=float(torch.exp(torch.tensor(logit_scale)).item()),
        bias=logit_bias,
    )


def fit_platt_scaling(
    similarities: torch.Tensor,
    labels: torch.Tensor,
    *,
    steps: int = 2000,
    lr: float = 0.05,
    use_log_scale: bool = False,
) -> CalibrationParams:
    # Platt scaling fits a 1D logistic regression on top of raw similarity:
    #   p(y=1 | s) = sigmoid(scale * s + bias)
    # where `s` is similarity and (`scale`, `bias`) are learned on labeled data.
    # If `use_log_scale=True`, we optimize `logit_scale` and map to
    #   scale = exp(logit_scale)
    # which matches SigLIP's native parameterization.
    if similarities.ndim != 1 or labels.ndim != 1:
        raise ValueError("similarities and labels must be 1D tensors")
    if similarities.shape[0] != labels.shape[0]:
        raise ValueError("similarities and labels must have the same length")

    x = similarities.to(dtype=torch.float32)
    y = labels.to(dtype=torch.float32)

    if use_log_scale:
        scale_param = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
    else:
        scale_param = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
    bias = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

    # Optimize calibration parameters by minimizing binary cross-entropy
    # between predicted logits and ground-truth labels.
    optimizer = torch.optim.Adam([scale_param, bias], lr=lr)
    for _ in range(steps):
        if use_log_scale:
            scale = torch.exp(scale_param)
        else:
            scale = scale_param
        logits = scale * x + bias
        loss = F.binary_cross_entropy_with_logits(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if use_log_scale:
        final_scale = float(torch.exp(scale_param.detach()).item())
    else:
        final_scale = float(scale_param.detach().item())
    return CalibrationParams(scale=final_scale, bias=float(bias.detach().item()))
