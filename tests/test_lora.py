from __future__ import annotations

import torch

from nanosiglip.siglip import LoRAConfig, SigLIP, SigLIPLoRA


def _tiny_siglip_config() -> dict:
    return {
        "text_config": {
            "vocab_size": 100,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "max_position_embeddings": 8,
            "hidden_act": "gelu_pytorch_tanh",
            "layer_norm_eps": 1e-6,
            "attention_dropout": 0.0,
            "projection_size": 32,
        },
        "vision_config": {
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_channels": 3,
            "image_size": 32,
            "patch_size": 16,
            "hidden_act": "gelu_pytorch_tanh",
            "layer_norm_eps": 1e-6,
            "attention_dropout": 0.0,
            "vision_use_head": True,
        },
    }


def test_lora_wrap_preserves_initial_forward() -> None:
    torch.manual_seed(0)

    base_model = SigLIP.from_config(_tiny_siglip_config())

    input_ids = torch.randint(0, 100, (2, 8), dtype=torch.long)
    pixel_values = torch.randn(2, 3, 32, 32)

    with torch.no_grad():
        base_out = base_model(input_ids=input_ids, pixel_values=pixel_values)

    lora_model = SigLIPLoRA(
        base_model,
        config=LoRAConfig(
            rank=4,
            alpha=8.0,
            dropout=0.0,
            target_modules=("q_proj", "k_proj", "v_proj", "out_proj"),
        ),
    )

    with torch.no_grad():
        lora_out = lora_model(input_ids=input_ids, pixel_values=pixel_values)

    # LoRA B matrices are initialized to zero, so wrapped model starts equivalent to base model.
    assert torch.allclose(
        base_out.logits_per_image, lora_out.logits_per_image, atol=1e-6, rtol=1e-6
    )


def test_only_lora_params_are_trainable() -> None:
    model = SigLIP.from_config(_tiny_siglip_config())
    lora_model = SigLIPLoRA(model, config=LoRAConfig(rank=4, alpha=8.0, dropout=0.0))

    trainable = [
        (name, p) for name, p in lora_model.named_parameters() if p.requires_grad
    ]
    assert trainable
    assert all(("lora_A" in name or "lora_B" in name) for name, _ in trainable)
    assert lora_model.num_lora_layers > 0
