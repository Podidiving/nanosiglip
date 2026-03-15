from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, SiglipModel

from nanosiglip.siglip import SigLIP, SigLIPImageProcessor, SigLIPTextProcessor

MODEL_ID = "google/siglip-base-patch16-224"
ASSET_IMAGE = Path(__file__).resolve().parents[1] / "assets" / "image.jpg"


def test_e2e_inference_matches_transformers_pipeline() -> None:
    texts = ["a cat", "a dog", "a landscape"]

    our_model, model_path = SigLIP.from_pretrained(MODEL_ID, return_model_path=True)
    our_image_processor = SigLIPImageProcessor.from_pretrained(model_path)
    our_text_processor = SigLIPTextProcessor.from_pretrained(model_path)

    hf_model = SiglipModel.from_pretrained(model_path)
    hf_processor = AutoProcessor.from_pretrained(model_path)

    image = Image.open(ASSET_IMAGE)

    our_inputs = {
        "pixel_values": our_image_processor(
            images=image, return_tensors="pt"
        ).pixel_values,
        "input_ids": our_text_processor(
            texts,
            padding="max_length",
            truncation=True,
            max_length=our_model.config.text_config.max_position_embeddings,
            return_tensors="pt",
        ).input_ids,
    }

    hf_inputs = hf_processor(
        text=texts,
        images=image,
        padding="max_length",
        truncation=True,
        max_length=our_model.config.text_config.max_position_embeddings,
        return_tensors="pt",
    )

    with torch.no_grad():
        our_outputs = our_model(**our_inputs)
        hf_outputs = hf_model(**hf_inputs)

    assert torch.allclose(
        our_outputs.logits_per_image, hf_outputs.logits_per_image, atol=1e-5, rtol=1e-4
    )
    assert torch.allclose(
        our_outputs.logits_per_text, hf_outputs.logits_per_text, atol=1e-5, rtol=1e-4
    )

    our_probs = torch.sigmoid(our_outputs.logits_per_image)
    hf_probs = torch.sigmoid(hf_outputs.logits_per_image)
    assert torch.allclose(our_probs, hf_probs, atol=1e-5, rtol=1e-4)
