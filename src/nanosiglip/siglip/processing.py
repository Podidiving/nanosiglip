from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image

from .hub import resolve_pretrained_path


@dataclass
class SigLIPBatchFeature:
    pixel_values: torch.Tensor | np.ndarray | list[np.ndarray]


class SigLIPImageProcessor:
    def __init__(
        self,
        *,
        do_resize: bool = True,
        size: dict[str, int] | None = None,
        resample: int = int(Image.Resampling.BICUBIC),
        do_rescale: bool = True,
        rescale_factor: float = 1 / 255,
        do_normalize: bool = True,
        image_mean: float | list[float] | None = None,
        image_std: float | list[float] | None = None,
        do_convert_rgb: bool | None = None,
    ) -> None:
        self.do_resize = do_resize
        self.size = size if size is not None else {"height": 224, "width": 224}
        self.resample = resample
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = image_mean if image_mean is not None else [0.5, 0.5, 0.5]
        self.image_std = image_std if image_std is not None else [0.5, 0.5, 0.5]
        self.do_convert_rgb = do_convert_rgb
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
    ) -> "SigLIPImageProcessor" | tuple["SigLIPImageProcessor", Path]:
        model_path = resolve_pretrained_path(
            pretrained_model_name_or_path,
            revision=revision,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            include_preprocessor=True,
        )

        cfg = model_path / "preprocessor_config.json"
        if cfg.exists():
            with cfg.open("r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = {}

        processor = cls(
            do_resize=data.get("do_resize", True),
            size=data.get("size", {"height": 224, "width": 224}),
            resample=data.get("resample", int(Image.Resampling.BICUBIC)),
            do_rescale=data.get("do_rescale", True),
            rescale_factor=data.get("rescale_factor", 1 / 255),
            do_normalize=data.get("do_normalize", True),
            image_mean=data.get("image_mean", [0.5, 0.5, 0.5]),
            image_std=data.get("image_std", [0.5, 0.5, 0.5]),
            do_convert_rgb=data.get("do_convert_rgb", None),
        )
        processor.model_path = model_path

        if return_model_path:
            return processor, model_path
        return processor

    def _to_pil(self, image: Image.Image | np.ndarray | torch.Tensor) -> Image.Image:
        if isinstance(image, Image.Image):
            return image

        if isinstance(image, torch.Tensor):
            array = image.detach().cpu().numpy()
        else:
            array = np.asarray(image)

        if array.ndim == 3 and array.shape[0] in (1, 3):
            array = np.transpose(array, (1, 2, 0))

        if np.issubdtype(array.dtype, np.floating):
            if array.max() <= 1.0:
                array = array * 255.0
            array = np.clip(array, 0, 255).astype(np.uint8)
        elif array.dtype != np.uint8:
            array = array.astype(np.uint8)

        return Image.fromarray(array)

    def preprocess(
        self,
        images: Image.Image
        | np.ndarray
        | torch.Tensor
        | Sequence[Image.Image | np.ndarray | torch.Tensor],
        *,
        return_tensors: str | None = None,
    ) -> SigLIPBatchFeature:
        if isinstance(images, (Image.Image, np.ndarray, torch.Tensor)):
            image_list = [images]
        else:
            image_list = list(images)

        processed: list[np.ndarray] = []
        mean = np.asarray(self.image_mean, dtype=np.float32)
        std = np.asarray(self.image_std, dtype=np.float32)

        for image in image_list:
            pil = self._to_pil(image)
            if self.do_convert_rgb:
                pil = pil.convert("RGB")
            arr = np.asarray(pil)

            if self.do_resize:
                width = int(self.size["width"])
                height = int(self.size["height"])
                pil = Image.fromarray(arr).resize(
                    (width, height), resample=Image.Resampling(self.resample)
                )
                arr = np.asarray(pil)

            arr = arr.astype(np.float32)
            if self.do_rescale:
                arr = arr * float(self.rescale_factor)

            if self.do_normalize:
                arr = (arr - mean) / std

            arr = np.transpose(arr, (2, 0, 1))
            processed.append(arr)

        if return_tensors == "pt":
            batch = torch.from_numpy(np.stack(processed, axis=0))
            return SigLIPBatchFeature(pixel_values=batch)
        if return_tensors == "np":
            batch = np.stack(processed, axis=0)
            return SigLIPBatchFeature(pixel_values=batch)
        return SigLIPBatchFeature(pixel_values=processed)

    def __call__(self, images: Any, **kwargs: Any) -> SigLIPBatchFeature:
        return self.preprocess(images=images, **kwargs)
