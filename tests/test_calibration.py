from __future__ import annotations

import torch

from nanosiglip.siglip.calibration import fit_platt_scaling, similarity_to_probability


def test_similarity_to_probability_matches_sigmoid_formula() -> None:
    similarity = torch.tensor([-1.0, -0.2, 0.0, 0.3, 0.9], dtype=torch.float32)
    scale = 7.5
    bias = -1.2

    out = similarity_to_probability(similarity, scale=scale, bias=bias)
    expected = torch.sigmoid(similarity * scale + bias)

    assert torch.allclose(out, expected, atol=0.0, rtol=0.0)


def test_fit_platt_scaling_recovers_params_on_synthetic_data() -> None:
    torch.manual_seed(0)

    true_scale = 6.0
    true_bias = -1.3

    similarities = torch.empty(1200).uniform_(-1, 1)
    probs = torch.sigmoid(true_scale * similarities + true_bias)
    labels = torch.bernoulli(probs)

    params = fit_platt_scaling(similarities, labels, steps=2500, lr=0.05)

    # Platt scaling is identifiable up to sampling noise; enforce a practical tolerance.
    assert abs(params.scale - true_scale) < 0.8
    assert abs(params.bias - true_bias) < 0.5


def test_fit_platt_scaling_recovers_params_with_log_scale_parameterization() -> None:
    torch.manual_seed(0)

    true_scale = 6.0
    true_bias = -1.3

    similarities = torch.empty(1200).uniform_(-1, 1)
    probs = torch.sigmoid(true_scale * similarities + true_bias)
    labels = torch.bernoulli(probs)

    params = fit_platt_scaling(
        similarities, labels, steps=2500, lr=0.05, use_log_scale=True
    )

    assert abs(params.scale - true_scale) < 0.8
    assert abs(params.bias - true_bias) < 0.5
