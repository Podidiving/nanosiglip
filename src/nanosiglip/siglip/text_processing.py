from __future__ import annotations

import json
import re
import string
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import sentencepiece as spm
import torch

from .hub import resolve_pretrained_path


@dataclass
class SigLIPTextBatchFeature:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor


class SigLIPTextProcessor:
    def __init__(
        self,
        vocab_file: str | Path,
        *,
        model_max_length: int = 64,
        do_lower_case: bool = True,
        pad_token: str = "</s>",
        eos_token: str = "</s>",
        unk_token: str = "<unk>",
    ) -> None:
        self.vocab_file = str(vocab_file)
        self.model_max_length = model_max_length
        self.do_lower_case = do_lower_case
        self.pad_token = pad_token
        self.eos_token = eos_token
        self.unk_token = unk_token

        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.Load(self.vocab_file)

        self.pad_token_id = self.sp_model.piece_to_id(self.pad_token)
        self.eos_token_id = self.sp_model.piece_to_id(self.eos_token)
        self.unk_token_id = self.sp_model.piece_to_id(self.unk_token)
        self.unk_token_length = len(self.sp_model.encode(self.unk_token))

        self.model_path: Path | None = None

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path,
        *,
        revision: str = "main",
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        return_model_path: bool = False,
    ) -> "SigLIPTextProcessor" | tuple["SigLIPTextProcessor", Path]:
        model_path = resolve_pretrained_path(
            pretrained_model_name_or_path,
            revision=revision,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            include_tokenizer=True,
        )

        tokenizer_cfg = model_path / "tokenizer_config.json"
        config = {}
        if tokenizer_cfg.exists():
            with tokenizer_cfg.open("r", encoding="utf-8") as f:
                config = json.load(f)

        processor = cls(
            vocab_file=model_path / "spiece.model",
            model_max_length=config.get("model_max_length", 64),
            do_lower_case=config.get("do_lower_case", True),
            pad_token=config.get("pad_token", "</s>"),
            eos_token=config.get("eos_token", "</s>"),
            unk_token=config.get("unk_token", "<unk>"),
        )
        processor.model_path = model_path

        if return_model_path:
            return processor, model_path
        return processor

    def remove_punctuation(self, text: str) -> str:
        return text.translate(str.maketrans("", "", string.punctuation))

    def canonicalize_text(self, text: str) -> str:
        if self.do_lower_case:
            text = text.lower()
        text = self.remove_punctuation(text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _tokenize_to_ids(self, text: str) -> list[int]:
        text = self.canonicalize_text(text)
        text = "▁" + text.replace("▁", " ")
        ids = self.sp_model.encode(text, out_type=int)
        if not ids or ids[-1] != self.eos_token_id:
            ids = ids + [self.eos_token_id]
        return ids

    def _encode_one(
        self,
        text: str,
        *,
        padding: str | bool | None,
        truncation: bool,
        max_length: int | None,
    ) -> tuple[list[int], list[int]]:
        token_ids = self._tokenize_to_ids(text)

        target_max_length = (
            max_length if max_length is not None else self.model_max_length
        )

        if truncation and len(token_ids) > target_max_length:
            token_ids = token_ids[:target_max_length]

        attention_mask = [1] * len(token_ids)

        if padding in ("max_length", True):
            pad_len = max(0, target_max_length - len(token_ids))
            if pad_len:
                token_ids = token_ids + [self.pad_token_id] * pad_len
                attention_mask = attention_mask + [0] * pad_len

        return token_ids, attention_mask

    def __call__(
        self,
        text: str | Sequence[str],
        *,
        padding: str | bool | None = None,
        truncation: bool = False,
        max_length: int | None = None,
        return_tensors: str | None = None,
    ) -> SigLIPTextBatchFeature:
        if isinstance(text, str):
            texts = [text]
        else:
            texts = list(text)

        input_ids: list[list[int]] = []
        attention_mask: list[list[int]] = []

        for item in texts:
            ids, mask = self._encode_one(
                item,
                padding=padding,
                truncation=truncation,
                max_length=max_length,
            )
            input_ids.append(ids)
            attention_mask.append(mask)

        if padding in ("longest",):
            longest = max(len(row) for row in input_ids)
            for i in range(len(input_ids)):
                pad_len = longest - len(input_ids[i])
                if pad_len:
                    input_ids[i].extend([self.pad_token_id] * pad_len)
                    attention_mask[i].extend([0] * pad_len)

        if return_tensors == "pt":
            return SigLIPTextBatchFeature(
                input_ids=torch.tensor(input_ids, dtype=torch.long),
                attention_mask=torch.tensor(attention_mask, dtype=torch.long),
            )

        raise ValueError("Only return_tensors='pt' is currently supported")
