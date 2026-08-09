from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .config import DiffusionConfig, MemoryConfig, ModelConfig
from .memory import NoiseAdaptiveGatedDeltaMixer


def resolve_ffn_widths(
    hidden_size: int,
    depth: int,
    mean_ratio: float,
    allocation: str = "uniform",
    strength: float = 0.0,
    multiple_of: int = 8,
) -> tuple[int, ...]:
    """Resolve per-block FFN widths while exactly conserving their total sum.

    Non-uniform profiles split depth into three equal consecutive stages. Strength
    is measured in MLP-ratio points. For example, mean_ratio=5 and a frontloaded
    strength of 1 resolves to stage ratios 6/5/4.
    """

    if hidden_size <= 0 or depth <= 0 or mean_ratio <= 0 or multiple_of <= 0:
        raise ValueError("hidden_size, depth, mean_ratio, and multiple_of must be positive")
    if allocation not in {"uniform", "frontloaded", "backloaded", "middle_heavy"}:
        raise ValueError(f"unsupported FFN allocation: {allocation}")
    target_per_block = int(round(hidden_size * mean_ratio / multiple_of)) * multiple_of
    if target_per_block <= 0:
        raise ValueError("resolved mean FFN width is not positive")
    if allocation == "uniform" or strength == 0:
        return (target_per_block,) * depth
    if depth % 3:
        raise ValueError("stage-wise FFN allocation requires depth divisible by three")

    if allocation == "frontloaded":
        stage_scores = (1.0, 0.0, -1.0)
    elif allocation == "backloaded":
        stage_scores = (-1.0, 0.0, 1.0)
    else:
        stage_scores = (-0.5, 1.0, -0.5)

    stage_size = depth // 3
    scores = [score for score in stage_scores for _ in range(stage_size)]
    raw = [target_per_block + hidden_size * strength * score for score in scores]
    if min(raw) <= 0:
        raise ValueError(f"allocation creates a non-positive FFN width: {raw}")

    widths = [int(math.floor(value / multiple_of)) * multiple_of for value in raw]
    target_total = target_per_block * depth
    missing_units, remainder = divmod(target_total - sum(widths), multiple_of)
    if remainder:
        raise AssertionError("FFN quantization residual is not divisible by multiple_of")
    fractions = sorted(
        range(depth),
        key=lambda index: raw[index] - widths[index],
        reverse=True,
    )
    if missing_units < 0 or missing_units > depth:
        raise AssertionError("unexpected FFN quantization correction")
    for index in fractions[:missing_units]:
        widths[index] += multiple_of
    if sum(widths) != target_total or min(widths) <= 0:
        raise AssertionError("failed to conserve FFN width budget")
    return tuple(widths)


def _modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def _timestep_embedding(timesteps: Tensor, dimension: int, max_period: int = 10_000) -> Tensor:
    half = dimension // 2
    frequencies = torch.exp(
        -math.log(max_period)
        * torch.arange(half, device=timesteps.device, dtype=torch.float32)
        / max(half, 1)
    )
    arguments = timesteps.float().unsqueeze(1) * frequencies.unsqueeze(0)
    embedding = torch.cat((torch.cos(arguments), torch.sin(arguments)), dim=-1)
    if dimension % 2:
        embedding = torch.cat((embedding, torch.zeros_like(embedding[:, :1])), dim=-1)
    return embedding


def _sincos_1d(positions: Tensor, dimension: int) -> Tensor:
    if dimension % 2:
        raise ValueError("1D sin-cos dimension must be even")
    frequencies = torch.arange(dimension // 2, dtype=torch.float32)
    frequencies = torch.pow(10_000.0, -frequencies / max(dimension // 2, 1))
    values = positions.reshape(-1, 1).float() * frequencies.reshape(1, -1)
    return torch.cat((torch.sin(values), torch.cos(values)), dim=1)


def make_2d_sincos_embedding(grid_size: int, dimension: int) -> Tensor:
    if dimension % 4:
        raise ValueError("2D sin-cos embedding dimension must be divisible by four")
    rows, columns = torch.meshgrid(
        torch.arange(grid_size, dtype=torch.float32),
        torch.arange(grid_size, dtype=torch.float32),
        indexing="ij",
    )
    row_embedding = _sincos_1d(rows.reshape(-1), dimension // 2)
    column_embedding = _sincos_1d(columns.reshape(-1), dimension // 2)
    return torch.cat((row_embedding, column_embedding), dim=1).unsqueeze(0)


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, frequency_size: int = 256) -> None:
        super().__init__()
        self.frequency_size = frequency_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, timesteps: Tensor) -> Tensor:
        frequency_embedding = _timestep_embedding(timesteps, self.frequency_size)
        return self.mlp(frequency_embedding)


class SelfAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int) -> None:
        super().__init__()
        if hidden_size % num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads
        self.qkv = nn.Linear(hidden_size, hidden_size * 3)
        self.projection = nn.Linear(hidden_size, hidden_size)

    def forward(self, x: Tensor, normalized_log_snr: Tensor | None = None) -> Tensor:
        del normalized_log_snr
        batch, tokens, hidden = x.shape
        qkv = self.qkv(x).reshape(batch, tokens, 3, self.num_heads, self.head_size)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(0)
        attended = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0)
        attended = attended.transpose(1, 2).reshape(batch, tokens, hidden)
        return self.projection(attended)


class FeedForward(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.intermediate_size = intermediate_size
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.activation = nn.GELU(approximate="tanh")
        self.fc2 = nn.Linear(intermediate_size, hidden_size)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(self.activation(self.fc1(x)))


class DiTBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_hidden_size: int,
        token_mixer: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        # Keep the historical attribute name for baseline checkpoint compatibility.
        self.attention = token_mixer or SelfAttention(hidden_size, num_heads)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp = FeedForward(hidden_size, mlp_hidden_size)
        self.ada_ln = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size))

    def forward(
        self,
        x: Tensor,
        conditioning: Tensor,
        normalized_log_snr: Tensor | None = None,
    ) -> Tensor:
        shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = self.ada_ln(
            conditioning
        ).chunk(6, dim=1)
        x = x + gate_attn.unsqueeze(1) * self.attention(
            _modulate(self.norm1(x), shift_attn, scale_attn), normalized_log_snr
        )
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            _modulate(self.norm2(x), shift_mlp, scale_mlp)
        )
        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.ada_ln = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size))
        self.projection = nn.Linear(hidden_size, patch_size * patch_size * out_channels)

    def forward(self, x: Tensor, conditioning: Tensor) -> Tensor:
        shift, scale = self.ada_ln(conditioning).chunk(2, dim=1)
        return self.projection(_modulate(self.norm(x), shift, scale))


class DiT(nn.Module):
    """Pixel-space, class-conditional DiT with per-block FFN widths."""

    def __init__(
        self,
        *,
        image_size: int,
        in_channels: int,
        patch_size: int,
        hidden_size: int,
        depth: int,
        num_heads: int,
        num_classes: int,
        mlp_hidden_sizes: Sequence[int],
        memory_config: MemoryConfig,
        log_snr_table: Tensor | None = None,
        normalized_log_snr_table: Tensor | None = None,
    ) -> None:
        super().__init__()
        if image_size % patch_size:
            raise ValueError("image_size must be divisible by patch_size")
        if len(mlp_hidden_sizes) != depth:
            raise ValueError("one mlp_hidden_size is required per block")
        self.image_size = image_size
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.depth = depth
        self.num_heads = num_heads
        self.num_classes = num_classes
        self.null_class_id = num_classes
        self.mlp_hidden_sizes = tuple(int(width) for width in mlp_hidden_sizes)
        self.memory_config = memory_config
        self.grid_size = image_size // patch_size
        self.num_tokens = self.grid_size * self.grid_size
        if memory_config.kind == "hybrid_gdn2" and normalized_log_snr_table is None:
            raise ValueError("hybrid GDN2 models require a diffusion log-SNR table")
        self.register_buffer("log_snr_table", log_snr_table, persistent=False)
        self.register_buffer(
            "normalized_log_snr_table", normalized_log_snr_table, persistent=False
        )
        shuffled_log_snr_table = None
        if normalized_log_snr_table is not None:
            shuffle_generator = torch.Generator().manual_seed(20260806)
            shuffled_indices = torch.randperm(
                normalized_log_snr_table.numel(), generator=shuffle_generator
            )
            shuffled_log_snr_table = normalized_log_snr_table[shuffled_indices]
        self.register_buffer(
            "shuffled_log_snr_table", shuffled_log_snr_table, persistent=False
        )

        self.patch_embed = nn.Conv2d(
            in_channels,
            hidden_size,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.register_buffer(
            "position_embedding",
            make_2d_sincos_embedding(self.grid_size, hidden_size),
            persistent=True,
        )
        self.timestep_embed = TimestepEmbedder(hidden_size)
        self.label_embed = nn.Embedding(num_classes + 1, hidden_size)
        memory_indices = set(memory_config.block_indices)
        directions = tuple(
            direction.strip() for direction in memory_config.scan_pattern.split(",")
        )
        blocks: list[DiTBlock] = []
        block_types: list[str] = []
        scan_directions: list[str | None] = []
        memory_number = 0
        for block_index, mlp_width in enumerate(self.mlp_hidden_sizes):
            if block_index in memory_indices:
                direction = directions[memory_number % len(directions)]
                mixer: nn.Module | None = NoiseAdaptiveGatedDeltaMixer(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    grid_size=self.grid_size,
                    direction=direction,
                    gate_mode=memory_config.gate_mode,
                    gate_rank=memory_config.gate_rank,
                    lambda_hidden_size=memory_config.lambda_hidden_size,
                    backend=memory_config.backend,
                )
                block_types.append("gdn2_memory")
                scan_directions.append(direction)
                memory_number += 1
            else:
                mixer = None
                block_types.append("softmax_attention")
                scan_directions.append(None)
            blocks.append(DiTBlock(hidden_size, num_heads, mlp_width, mixer))
        self.blocks = nn.ModuleList(blocks)
        self.block_types = tuple(block_types)
        self.scan_directions = tuple(scan_directions)
        self.memory_block_indices = tuple(memory_config.block_indices)
        self.log_snr_intervention = "normal"
        self.final = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self) -> None:
        def initialize(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        self.apply(initialize)
        patch_weight = self.patch_embed.weight.reshape(self.patch_embed.weight.shape[0], -1)
        nn.init.xavier_uniform_(patch_weight)
        nn.init.zeros_(self.patch_embed.bias)
        nn.init.normal_(self.label_embed.weight, std=0.02)
        nn.init.normal_(self.timestep_embed.mlp[0].weight, std=0.02)
        nn.init.normal_(self.timestep_embed.mlp[2].weight, std=0.02)

        for block in self.blocks:
            nn.init.zeros_(block.ada_ln[-1].weight)
            nn.init.zeros_(block.ada_ln[-1].bias)
            if isinstance(block.attention, NoiseAdaptiveGatedDeltaMixer):
                nn.init.zeros_(block.attention.lambda_mlp[-1].weight)
                nn.init.zeros_(block.attention.lambda_mlp[-1].bias)
        nn.init.zeros_(self.final.ada_ln[-1].weight)
        nn.init.zeros_(self.final.ada_ln[-1].bias)
        nn.init.zeros_(self.final.projection.weight)
        nn.init.zeros_(self.final.projection.bias)

    def _unpatchify(self, patches: Tensor) -> Tensor:
        batch = patches.shape[0]
        patches = patches.reshape(
            batch,
            self.grid_size,
            self.grid_size,
            self.patch_size,
            self.patch_size,
            self.out_channels,
        )
        return patches.permute(0, 5, 1, 3, 2, 4).reshape(
            batch,
            self.out_channels,
            self.image_size,
            self.image_size,
        )

    def forward(self, x: Tensor, timesteps: Tensor, labels: Tensor) -> Tensor:
        if x.ndim != 4 or x.shape[1:] != (
            self.in_channels,
            self.image_size,
            self.image_size,
        ):
            raise ValueError(
                f"expected x shape [B,{self.in_channels},{self.image_size},{self.image_size}], "
                f"received {tuple(x.shape)}"
            )
        batch = x.shape[0]
        if timesteps.shape != (batch,) or labels.shape != (batch,):
            raise ValueError("timesteps and labels must both have shape [B]")
        if labels.dtype != torch.long:
            raise TypeError("labels must have dtype torch.long")
        # Value-range checks are useful for CPU tests, but converting a CUDA
        # reduction to a Python bool synchronizes every denoiser forward.  The
        # training/sampling pipeline constructs bounded CUDA indices; embedding
        # and table lookup still reject out-of-range positive indices.
        if labels.device.type == "cpu" and (
            torch.any(labels < 0) or torch.any(labels > self.null_class_id)
        ):
            raise ValueError(f"labels must be in [0, {self.null_class_id}]")

        tokens = self.patch_embed(x).flatten(2).transpose(1, 2)
        tokens = tokens + self.position_embedding.to(dtype=tokens.dtype)
        conditioning = self.timestep_embed(timesteps) + self.label_embed(labels)
        normalized_log_snr = None
        if self.normalized_log_snr_table is not None:
            if timesteps.dtype != torch.long:
                raise TypeError("memory models require integer diffusion timesteps")
            if timesteps.device.type == "cpu" and (
                torch.any(timesteps < 0)
                or torch.any(timesteps >= self.normalized_log_snr_table.numel())
            ):
                raise ValueError("timestep is outside the configured diffusion schedule")
            lookup_timesteps = timesteps
            if self.log_snr_intervention == "reversed":
                lookup_timesteps = self.normalized_log_snr_table.numel() - 1 - timesteps
            if self.log_snr_intervention == "shuffled":
                if self.shuffled_log_snr_table is None:
                    raise AssertionError("missing shuffled log-SNR lookup table")
                normalized_log_snr = self.shuffled_log_snr_table[timesteps]
            else:
                normalized_log_snr = self.normalized_log_snr_table[lookup_timesteps]
            if self.log_snr_intervention == "zero":
                normalized_log_snr = torch.zeros_like(normalized_log_snr)
        for block in self.blocks:
            tokens = block(tokens, conditioning, normalized_log_snr)
        return self._unpatchify(self.final(tokens, conditioning))

    def set_memory_diagnostics(self, enabled: bool) -> None:
        for block in self.blocks:
            if isinstance(block.attention, NoiseAdaptiveGatedDeltaMixer):
                block.attention.set_diagnostics(enabled)

    def set_memory_intervention(
        self,
        *,
        lambda_override: float | None = None,
        blockwise_mean_lambda: bool = False,
        log_snr_mode: str = "normal",
    ) -> dict[str, float]:
        if log_snr_mode not in {"normal", "reversed", "shuffled", "zero"}:
            raise ValueError(
                "log_snr_mode must be normal, reversed, shuffled, or zero"
            )
        if lambda_override is not None and blockwise_mean_lambda:
            raise ValueError(
                "lambda_override and blockwise_mean_lambda are mutually exclusive"
            )
        if (lambda_override is not None or blockwise_mean_lambda) and (
            log_snr_mode != "normal"
        ):
            raise ValueError(
                "lambda interventions cannot be combined with a log-SNR intervention"
            )
        if not self.memory_block_indices and (
            lambda_override is not None
            or blockwise_mean_lambda
            or log_snr_mode != "normal"
        ):
            raise ValueError("memory interventions require a hybrid GDN2 model")
        self.log_snr_intervention = log_snr_mode
        memory_mixers = [
            (block_index, block.attention)
            for block_index, block in enumerate(self.blocks)
            if isinstance(block.attention, NoiseAdaptiveGatedDeltaMixer)
        ]
        for _, mixer in memory_mixers:
            mixer.set_lambda_override(None)

        resolved_overrides: dict[str, float] = {}
        if blockwise_mean_lambda:
            if self.normalized_log_snr_table is None:
                raise AssertionError("missing normalized log-SNR table")
            with torch.no_grad():
                for block_index, mixer in memory_mixers:
                    mean_lambda = float(
                        mixer.decoupling_strength(self.normalized_log_snr_table)
                        .float()
                        .mean()
                    )
                    mixer.set_lambda_override(mean_lambda)
                    resolved_overrides[f"b{block_index + 1:02d}"] = mean_lambda
        else:
            for block_index, mixer in memory_mixers:
                mixer.set_lambda_override(lambda_override)
                if lambda_override is not None:
                    resolved_overrides[f"b{block_index + 1:02d}"] = lambda_override
        return resolved_overrides

    def memory_diagnostics(self) -> dict[str, Tensor]:
        diagnostics: dict[str, Tensor] = {}
        for block_index, block in enumerate(self.blocks):
            if not isinstance(block.attention, NoiseAdaptiveGatedDeltaMixer):
                continue
            for name, value in block.attention.latest_diagnostics().items():
                diagnostics[f"memory_b{block_index + 1:02d}_{name}"] = value
        return diagnostics


def build_model(
    config: ModelConfig,
    diffusion_config: DiffusionConfig | None = None,
) -> DiT:
    mlp_widths = resolve_ffn_widths(
        hidden_size=config.hidden_size,
        depth=config.depth,
        mean_ratio=config.mlp_ratio,
        allocation=config.allocation.kind,
        strength=config.allocation.strength,
        multiple_of=config.allocation.multiple_of,
    )
    log_snr_table = None
    normalized_log_snr_table = None
    if config.memory.kind == "hybrid_gdn2":
        if diffusion_config is None:
            raise ValueError("a DiffusionConfig is required to build a hybrid GDN2 model")
        # Local import avoids coupling the diffusion objective to model internals.
        from .diffusion import make_beta_schedule

        betas = make_beta_schedule(diffusion_config)
        alpha_bars = torch.cumprod(1.0 - betas, dim=0)
        log_snr_table = torch.log(alpha_bars) - torch.log1p(-alpha_bars)
        clipped = log_snr_table.clamp(
            -config.memory.log_snr_clip, config.memory.log_snr_clip
        )
        normalized_log_snr_table = (clipped - clipped.mean()) / clipped.std(
            unbiased=False
        ).clamp_min(1e-6)
    return DiT(
        image_size=config.image_size,
        in_channels=config.in_channels,
        patch_size=config.patch_size,
        hidden_size=config.hidden_size,
        depth=config.depth,
        num_heads=config.num_heads,
        num_classes=config.num_classes,
        mlp_hidden_sizes=mlp_widths,
        memory_config=config.memory,
        log_snr_table=log_snr_table,
        normalized_log_snr_table=normalized_log_snr_table,
    )
