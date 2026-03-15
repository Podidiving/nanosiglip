from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from .model import SigLIP


class LoRALinear(nn.Module):
    def __init__(
        self, base_layer: nn.Linear, rank: int, alpha: float, dropout: float = 0.0
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be > 0")

        self.base_layer = base_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout)

        for param in self.base_layer.parameters():
            param.requires_grad = False

        self.lora_A = nn.Parameter(torch.empty(rank, base_layer.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base_layer.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_layer(x)
        lora_out = (
            F.linear(F.linear(self.dropout(x), self.lora_A), self.lora_B) * self.scaling
        )
        return base_out + lora_out


@dataclass
class LoRAConfig:
    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.0
    target_modules: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "out_proj")


class SigLIPLoRA(nn.Module):
    def __init__(self, model: SigLIP, config: LoRAConfig | None = None) -> None:
        super().__init__()
        self.model = model
        self.config = config if config is not None else LoRAConfig()

        for param in self.model.parameters():
            param.requires_grad = False

        self.num_lora_layers = self._inject_lora(self.model)
        if self.num_lora_layers == 0:
            raise ValueError("No target modules were replaced by LoRA layers")

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path,
        *,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
        target_modules: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "out_proj"),
        revision: str = "main",
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        map_location: str | torch.device = "cpu",
    ) -> "SigLIPLoRA":
        base_model = SigLIP.from_pretrained(
            pretrained_model_name_or_path,
            revision=revision,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            map_location=map_location,
        )
        config = LoRAConfig(
            rank=rank, alpha=alpha, dropout=dropout, target_modules=target_modules
        )
        return cls(base_model, config=config)

    def _inject_lora(self, module: nn.Module) -> int:
        # torch.nn.MultiheadAttention directly reads out_proj.weight/out_proj.bias
        # inside its forward path, so replacing those internals with wrappers
        # breaks assumptions about parameter layout.
        if isinstance(module, nn.MultiheadAttention):
            return 0

        replaced = 0
        for name, child in list(module.named_children()):
            if isinstance(child, nn.Linear) and name in self.config.target_modules:
                setattr(
                    module,
                    name,
                    LoRALinear(
                        base_layer=child,
                        rank=self.config.rank,
                        alpha=self.config.alpha,
                        dropout=self.config.dropout,
                    ),
                )
                replaced += 1
                continue
            replaced += self._inject_lora(child)
        return replaced

    def lora_state_dict(self) -> dict[str, torch.Tensor]:
        state = self.state_dict()
        return {k: v for k, v in state.items() if ".lora_A" in k or ".lora_B" in k}

    def load_lora_state_dict(
        self, state_dict: dict[str, torch.Tensor], strict: bool = True
    ) -> None:
        result = self.load_state_dict(state_dict, strict=False)
        unexpected = result.unexpected_keys
        missing = [k for k in result.missing_keys if ".lora_" in k]
        if strict and (unexpected or missing):
            raise RuntimeError(
                f"LoRA state mismatch. Missing: {missing}, Unexpected: {unexpected}"
            )

    def save_lora_weights(self, path: str | Path) -> None:
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.lora_state_dict(), out_path)

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
