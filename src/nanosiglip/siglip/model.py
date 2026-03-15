from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from torch import nn

from .hub import resolve_pretrained_path


@dataclass
class BaseModelOutput:
    last_hidden_state: torch.Tensor


@dataclass
class BaseModelOutputWithPooling:
    last_hidden_state: torch.Tensor
    pooler_output: torch.Tensor


@dataclass
class SigLIPOutput:
    loss: torch.Tensor | None
    logits_per_image: torch.Tensor
    logits_per_text: torch.Tensor
    text_embeds: torch.Tensor
    image_embeds: torch.Tensor
    text_model_output: BaseModelOutputWithPooling
    vision_model_output: BaseModelOutputWithPooling


@dataclass
class SigLIPTextConfig:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    max_position_embeddings: int
    hidden_act: str
    layer_norm_eps: float
    attention_dropout: float
    projection_size: int

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SigLIPTextConfig":
        defaults = {
            "vocab_size": 32000,
            "hidden_size": 768,
            "intermediate_size": 3072,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            "max_position_embeddings": 64,
            "hidden_act": "gelu_pytorch_tanh",
            "layer_norm_eps": 1e-6,
            "attention_dropout": 0.0,
            "projection_size": 768,
        }
        cfg = {**defaults, **data}
        return cls(
            vocab_size=cfg["vocab_size"],
            hidden_size=cfg["hidden_size"],
            intermediate_size=cfg["intermediate_size"],
            num_hidden_layers=cfg["num_hidden_layers"],
            num_attention_heads=cfg["num_attention_heads"],
            max_position_embeddings=cfg["max_position_embeddings"],
            hidden_act=cfg["hidden_act"],
            layer_norm_eps=cfg["layer_norm_eps"],
            attention_dropout=cfg["attention_dropout"],
            projection_size=cfg["projection_size"],
        )


@dataclass
class SigLIPVisionConfig:
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_channels: int
    image_size: int
    patch_size: int
    hidden_act: str
    layer_norm_eps: float
    attention_dropout: float
    vision_use_head: bool

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SigLIPVisionConfig":
        defaults = {
            "hidden_size": 768,
            "intermediate_size": 3072,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            "num_channels": 3,
            "image_size": 224,
            "patch_size": 16,
            "hidden_act": "gelu_pytorch_tanh",
            "layer_norm_eps": 1e-6,
            "attention_dropout": 0.0,
            "vision_use_head": True,
        }
        cfg = {**defaults, **data}
        return cls(
            hidden_size=cfg["hidden_size"],
            intermediate_size=cfg["intermediate_size"],
            num_hidden_layers=cfg["num_hidden_layers"],
            num_attention_heads=cfg["num_attention_heads"],
            num_channels=cfg["num_channels"],
            image_size=cfg["image_size"],
            patch_size=cfg["patch_size"],
            hidden_act=cfg["hidden_act"],
            layer_norm_eps=cfg["layer_norm_eps"],
            attention_dropout=cfg["attention_dropout"],
            vision_use_head=cfg["vision_use_head"],
        )


@dataclass
class SigLIPConfig:
    text_config: SigLIPTextConfig
    vision_config: SigLIPVisionConfig

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SigLIPConfig":
        return cls(
            text_config=SigLIPTextConfig.from_dict(data["text_config"]),
            vision_config=SigLIPVisionConfig.from_dict(data["vision_config"]),
        )


def _activation_fn(name: str):
    if name == "gelu":
        return F.gelu
    if name == "gelu_pytorch_tanh":
        return lambda x: F.gelu(x, approximate="tanh")
    if name == "quick_gelu":
        return lambda x: x * torch.sigmoid(1.702 * x)
    if name == "relu":
        return F.relu
    if name == "silu":
        return F.silu
    raise ValueError(f"Unsupported hidden_act: {name}")


def _load_state_dict(
    model_path: Path, device: str | torch.device = "cpu"
) -> dict[str, torch.Tensor]:
    single_file = model_path / "model.safetensors"
    if single_file.exists():
        return load_file(str(single_file), device=str(device))

    index_file = model_path / "model.safetensors.index.json"
    if not index_file.exists():
        raise FileNotFoundError(
            f"Could not find model.safetensors or model.safetensors.index.json in {model_path}"
        )

    with index_file.open("r", encoding="utf-8") as f:
        index_data = json.load(f)

    weight_map: dict[str, str] = index_data["weight_map"]
    shards = sorted(set(weight_map.values()))
    state_dict: dict[str, torch.Tensor] = {}
    for shard in shards:
        shard_path = model_path / shard
        if not shard_path.exists():
            raise FileNotFoundError(f"Missing shard file '{shard_path}'")
        state_dict.update(load_file(str(shard_path), device=str(device)))
    return state_dict


def _create_attention_mask(
    attention_mask: torch.Tensor | None, hidden_states: torch.Tensor
) -> torch.Tensor | None:
    if attention_mask is None:
        return None
    if attention_mask.dim() == 4:
        return attention_mask.to(dtype=hidden_states.dtype, device=hidden_states.device)
    if attention_mask.dim() != 2:
        raise ValueError(
            f"attention_mask must have shape [batch, seq] or [batch, 1, q, k], got shape {tuple(attention_mask.shape)}"
        )

    mask = 1.0 - attention_mask[:, None, None, :].to(
        dtype=hidden_states.dtype, device=hidden_states.device
    )
    return mask * torch.finfo(hidden_states.dtype).min


class SigLIPAttention(nn.Module):
    def __init__(self, config: SigLIPTextConfig | SigLIPVisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError("hidden_size must be divisible by num_attention_heads")

        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_dropout

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, embed_dim = hidden_states.shape

        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        query = query.view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key = key.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(
            1, 2
        )
        value = value.view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)

        attn_weights = torch.matmul(query, key.transpose(-1, -2)) * self.scale
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
            query.dtype
        )
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.matmul(attn_weights, value)
        attn_output = (
            attn_output.transpose(1, 2)
            .reshape(batch_size, seq_len, embed_dim)
            .contiguous()
        )
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights


class SigLIPMLP(nn.Module):
    def __init__(self, config: SigLIPTextConfig | SigLIPVisionConfig):
        super().__init__()
        self.activation_fn = _activation_fn(config.hidden_act)
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class SigLIPEncoderLayer(nn.Module):
    def __init__(self, config: SigLIPTextConfig | SigLIPVisionConfig):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.self_attn = SigLIPAttention(config)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = SigLIPMLP(config)

    def forward(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states, attention_mask=attention_mask
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class SigLIPEncoder(nn.Module):
    def __init__(self, config: SigLIPTextConfig | SigLIPVisionConfig):
        super().__init__()
        self.layers = nn.ModuleList(
            [SigLIPEncoderLayer(config) for _ in range(config.num_hidden_layers)]
        )

    def forward(
        self, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> BaseModelOutput:
        hidden_states = inputs_embeds
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask=attention_mask)
        return BaseModelOutput(last_hidden_state=hidden_states)


class SigLIPTextEmbeddings(nn.Module):
    def __init__(self, config: SigLIPTextConfig):
        super().__init__()
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embedding = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )
        self.register_buffer(
            "position_ids",
            torch.arange(config.max_position_embeddings).expand((1, -1)),
            persistent=False,
        )

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        position_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        seq_length = (
            input_ids.shape[-1] if input_ids is not None else inputs_embeds.shape[-2]
        )
        max_position_embeddings = self.position_embedding.weight.shape[0]
        if seq_length > max_position_embeddings:
            raise ValueError(
                f"Sequence length {seq_length} exceeds max_position_embeddings={max_position_embeddings}"
            )

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("input_ids must be provided if inputs_embeds is None")
            inputs_embeds = self.token_embedding(input_ids)
        return inputs_embeds + self.position_embedding(position_ids)


class SigLIPTextTransformer(nn.Module):
    def __init__(self, config: SigLIPTextConfig):
        super().__init__()
        self.embeddings = SigLIPTextEmbeddings(config)
        self.encoder = SigLIPEncoder(config)
        self.final_layer_norm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )
        self.head = nn.Linear(config.hidden_size, config.projection_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> BaseModelOutputWithPooling:
        if input_ids is None:
            raise ValueError("input_ids must be provided")

        hidden_states = self.embeddings(input_ids=input_ids, position_ids=position_ids)
        attn_mask = _create_attention_mask(attention_mask, hidden_states)

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states, attention_mask=attn_mask
        )
        last_hidden_state = self.final_layer_norm(encoder_outputs.last_hidden_state)

        pooled_output = self.head(last_hidden_state[:, -1, :])

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
        )


class SigLIPVisionEmbeddings(nn.Module):
    def __init__(self, config: SigLIPVisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding=0,
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.position_embedding = nn.Embedding(self.num_patches, self.embed_dim)
        self.register_buffer(
            "position_ids",
            torch.arange(self.num_patches).expand((1, -1)),
            persistent=False,
        )

    def interpolate_pos_encoding(
        self, embeddings: torch.Tensor, height: int, width: int
    ) -> torch.Tensor:
        num_patches = embeddings.shape[1]
        num_positions = self.position_embedding.weight.shape[0]

        if num_patches == num_positions and height == width:
            return self.position_embedding(self.position_ids)

        patch_pos_embed = self.position_embedding.weight.unsqueeze(0)
        dim = embeddings.shape[-1]

        new_height = height // self.patch_size
        new_width = width // self.patch_size

        sqrt_num_positions = int(math.sqrt(num_positions))
        patch_pos_embed = patch_pos_embed.reshape(
            1, sqrt_num_positions, sqrt_num_positions, dim
        )
        patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)

        patch_pos_embed = F.interpolate(
            patch_pos_embed,
            size=(new_height, new_width),
            mode="bicubic",
            align_corners=False,
        )
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).reshape(1, -1, dim)
        return patch_pos_embed

    def forward(
        self, pixel_values: torch.Tensor, interpolate_pos_encoding: bool = False
    ) -> torch.Tensor:
        _, _, height, width = pixel_values.shape
        target_dtype = self.patch_embedding.weight.dtype
        patch_embeds = self.patch_embedding(pixel_values.to(dtype=target_dtype))
        embeddings = patch_embeds.flatten(2).transpose(1, 2)

        if interpolate_pos_encoding:
            embeddings = embeddings + self.interpolate_pos_encoding(
                embeddings, height, width
            )
        else:
            embeddings = embeddings + self.position_embedding(self.position_ids)
        return embeddings


class SigLIPMultiheadAttentionPoolingHead(nn.Module):
    def __init__(self, config: SigLIPVisionConfig):
        super().__init__()
        self.probe = nn.Parameter(torch.randn(1, 1, config.hidden_size))
        self.attention = nn.MultiheadAttention(
            config.hidden_size, config.num_attention_heads, batch_first=True
        )
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = SigLIPMLP(config)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        batch_size = hidden_state.shape[0]
        probe = self.probe.repeat(batch_size, 1, 1)
        hidden_state = self.attention(probe, hidden_state, hidden_state)[0]

        residual = hidden_state
        hidden_state = self.layernorm(hidden_state)
        hidden_state = residual + self.mlp(hidden_state)
        return hidden_state[:, 0]


class SigLIPVisionTransformer(nn.Module):
    def __init__(self, config: SigLIPVisionConfig):
        super().__init__()
        self.embeddings = SigLIPVisionEmbeddings(config)
        self.encoder = SigLIPEncoder(config)
        self.post_layernorm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )
        self.use_head = config.vision_use_head
        if self.use_head:
            self.head = SigLIPMultiheadAttentionPoolingHead(config)

    def forward(
        self,
        pixel_values: torch.Tensor,
        interpolate_pos_encoding: bool = False,
    ) -> BaseModelOutputWithPooling:
        hidden_states = self.embeddings(
            pixel_values, interpolate_pos_encoding=interpolate_pos_encoding
        )
        encoder_outputs = self.encoder(inputs_embeds=hidden_states)
        last_hidden_state = self.post_layernorm(encoder_outputs.last_hidden_state)
        pooler_output = (
            self.head(last_hidden_state) if self.use_head else last_hidden_state[:, 0]
        )

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooler_output,
        )


class SigLIP(nn.Module):
    def __init__(self, config: SigLIPConfig):
        super().__init__()
        self.config = config
        self.text_model = SigLIPTextTransformer(config.text_config)
        self.vision_model = SigLIPVisionTransformer(config.vision_config)

        self.logit_scale = nn.Parameter(torch.randn(1))
        self.logit_bias = nn.Parameter(torch.randn(1))

        self.model_path: Path | None = None

    @classmethod
    def from_config(cls, config_dict: dict[str, Any]) -> "SigLIP":
        config = SigLIPConfig.from_dict(config_dict)
        return cls(config)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path,
        *,
        revision: str = "main",
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        strict: bool = True,
        map_location: str | torch.device = "cpu",
        return_model_path: bool = False,
    ) -> "SigLIP" | tuple["SigLIP", Path]:
        model_path = resolve_pretrained_path(
            pretrained_model_name_or_path,
            revision=revision,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )

        with (model_path / "config.json").open("r", encoding="utf-8") as f:
            config_dict = json.load(f)

        model = cls.from_config(config_dict)
        state_dict = _load_state_dict(model_path, device=map_location)
        load_result = model.load_state_dict(state_dict, strict=strict)
        if strict and (load_result.missing_keys or load_result.unexpected_keys):
            raise RuntimeError(
                "Model state_dict mismatch. "
                f"Missing keys: {load_result.missing_keys}. Unexpected keys: {load_result.unexpected_keys}."
            )

        model.model_path = model_path
        model.to(map_location)
        model.eval()

        if return_model_path:
            return model, model_path
        return model

    def get_text_features(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> BaseModelOutputWithPooling:
        return self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )

    def get_image_features(
        self,
        pixel_values: torch.Tensor,
        interpolate_pos_encoding: bool = False,
    ) -> BaseModelOutputWithPooling:
        return self.vision_model(
            pixel_values=pixel_values,
            interpolate_pos_encoding=interpolate_pos_encoding,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        return_loss: bool | None = None,
        interpolate_pos_encoding: bool = False,
    ) -> SigLIPOutput:
        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            interpolate_pos_encoding=interpolate_pos_encoding,
        )
        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )

        image_embeds = vision_outputs.pooler_output
        text_embeds = text_outputs.pooler_output

        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

        logits_per_text = torch.matmul(
            text_embeds, image_embeds.t().to(text_embeds.device)
        )
        logits_per_text = logits_per_text * self.logit_scale.exp().to(
            text_embeds.device
        )
        logits_per_text = logits_per_text + self.logit_bias.to(text_embeds.device)

        logits_per_image = logits_per_text.t()

        loss = None
        if return_loss:
            eye = torch.eye(logits_per_text.size(0), device=logits_per_text.device)
            m1_diag1 = -torch.ones_like(logits_per_text) + 2 * eye
            loglik = F.logsigmoid(m1_diag1 * logits_per_text)
            nll = -torch.sum(loglik, dim=-1)
            loss = nll.mean()

        return SigLIPOutput(
            loss=loss,
            logits_per_image=logits_per_image,
            logits_per_text=logits_per_text,
            text_embeds=text_embeds,
            image_embeds=image_embeds,
            text_model_output=text_outputs,
            vision_model_output=vision_outputs,
        )
