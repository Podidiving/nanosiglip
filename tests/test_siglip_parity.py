from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image
from transformers import SiglipImageProcessor, SiglipModel

from nanosiglip.siglip import SigLIP, SigLIPImageProcessor

MODEL_ID = "google/siglip-base-patch16-224"


@pytest.mark.parametrize("batch_size", [1, 2])
def test_model_forward_matches_transformers(batch_size: int) -> None:
    torch.manual_seed(0)

    our_model, model_path = SigLIP.from_pretrained(MODEL_ID, return_model_path=True)
    hf_model = SiglipModel.from_pretrained(model_path)

    seq_len = 64
    vocab_size = our_model.config.text_config.vocab_size
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long)
    attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long)
    pixel_values = torch.randn(
        batch_size,
        our_model.config.vision_config.num_channels,
        our_model.config.vision_config.image_size,
        our_model.config.vision_config.image_size,
    )

    with torch.no_grad():
        ours = our_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
        )
        hf = hf_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
        )

        ours_text = our_model.get_text_features(
            input_ids=input_ids, attention_mask=attention_mask
        )
        hf_text = hf_model.get_text_features(
            input_ids=input_ids, attention_mask=attention_mask
        )

        ours_image = our_model.get_image_features(pixel_values=pixel_values)
        hf_image = hf_model.get_image_features(pixel_values=pixel_values)

    assert torch.allclose(
        ours.logits_per_text, hf.logits_per_text, atol=1e-5, rtol=1e-4
    )
    assert torch.allclose(
        ours.logits_per_image, hf.logits_per_image, atol=1e-5, rtol=1e-4
    )
    assert torch.allclose(ours.text_embeds, hf.text_embeds, atol=1e-5, rtol=1e-4)
    assert torch.allclose(ours.image_embeds, hf.image_embeds, atol=1e-5, rtol=1e-4)

    assert torch.allclose(
        ours_text.pooler_output, hf_text.pooler_output, atol=1e-5, rtol=1e-4
    )
    assert torch.allclose(
        ours_image.pooler_output, hf_image.pooler_output, atol=1e-5, rtol=1e-4
    )


def test_image_processor_matches_transformers() -> None:
    rng = np.random.default_rng(0)
    image = Image.fromarray(
        rng.integers(0, 256, size=(333, 517, 3), dtype=np.uint8), mode="RGB"
    )

    our_processor, model_path = SigLIPImageProcessor.from_pretrained(
        MODEL_ID, return_model_path=True
    )
    hf_processor = SiglipImageProcessor.from_pretrained(model_path)

    ours = our_processor(images=[image, image], return_tensors="pt")
    hf = hf_processor(images=[image, image], return_tensors="pt")

    assert isinstance(ours.pixel_values, torch.Tensor)
    assert ours.pixel_values.shape == hf.pixel_values.shape
    assert torch.allclose(ours.pixel_values, hf.pixel_values, atol=1e-6, rtol=0.0)
