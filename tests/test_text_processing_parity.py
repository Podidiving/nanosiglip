from __future__ import annotations

import torch
from transformers import SiglipTokenizer

from nanosiglip.siglip import SigLIPTextProcessor

MODEL_ID = "google/siglip-base-patch16-224"


def test_text_processor_matches_transformers_max_length_padding() -> None:
    texts = [
        "A photo of a CAT!!",
        "two dogs, running across the field.",
        "   SPACES   and punctuation...???",
    ]

    our_processor, model_path = SigLIPTextProcessor.from_pretrained(
        MODEL_ID, return_model_path=True
    )
    hf_tokenizer = SiglipTokenizer.from_pretrained(model_path)

    ours = our_processor(
        texts,
        padding="max_length",
        truncation=True,
        max_length=64,
        return_tensors="pt",
    )
    hf = hf_tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=64,
        return_attention_mask=True,
        return_tensors="pt",
    )

    assert isinstance(ours.input_ids, torch.Tensor)
    assert isinstance(ours.attention_mask, torch.Tensor)
    assert torch.equal(ours.input_ids, hf.input_ids)
    assert torch.equal(ours.attention_mask, hf.attention_mask)


def test_text_processor_matches_transformers_longest_padding() -> None:
    texts = ["cat", "a very very long text about multiple cats and dogs"]

    our_processor = SigLIPTextProcessor.from_pretrained(MODEL_ID)
    hf_tokenizer = SiglipTokenizer.from_pretrained(MODEL_ID)

    ours = our_processor(
        texts, padding="longest", truncation=False, return_tensors="pt"
    )
    hf = hf_tokenizer(
        texts,
        padding="longest",
        truncation=False,
        return_attention_mask=True,
        return_tensors="pt",
    )

    assert torch.equal(ours.input_ids, hf.input_ids)
    assert torch.equal(ours.attention_mask, hf.attention_mask)
