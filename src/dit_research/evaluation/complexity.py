from __future__ import annotations

from typing import Any

from torch import nn

from ..config import ModelConfig
from ..memory import NoiseAdaptiveGatedDeltaMixer
from ..model import DiT


COUNTING_CONVENTION = (
    "analytic dense MACs per image; one multiply-accumulate=1 MAC; "
    "includes patch/timestep projections, mixer projections, softmax QK+AV or the "
    "operator-equivalent GDN2 state recurrence, FFN, adaLN and output projection; "
    "excludes bias, normalization, nonlinear activation, trigonometric embedding and additions"
)


def _parameter_count(module: nn.Module, trainable_only: bool) -> int:
    return sum(
        parameter.numel()
        for parameter in module.parameters()
        if parameter.requires_grad or not trainable_only
    )


def analytic_complexity(model: DiT, config: ModelConfig) -> dict[str, Any]:
    tokens = model.num_tokens
    hidden = config.hidden_size
    patch = config.patch_size
    channels = config.in_channels

    patch_macs = tokens * (patch * patch * channels) * hidden
    timestep_macs = 256 * hidden + hidden * hidden
    block_macs: list[int] = []
    mixer_macs: list[int] = []
    block_types: list[str] = []
    for block, mlp_hidden in zip(model.blocks, model.mlp_hidden_sizes, strict=True):
        if isinstance(block.attention, NoiseAdaptiveGatedDeltaMixer):
            head_size = block.attention.head_size
            heads = block.attention.num_heads
            rank = block.attention.gate_rank
            controller_hidden = block.attention.lambda_hidden_size
            # qkv + output projection, low-rank gate controller, lambda MLP.
            mixer_projection_macs = 4 * tokens * hidden * hidden
            gate_projection_macs = 5 * tokens * hidden * rank
            lambda_controller_macs = 2 * controller_hidden
            # Per token/head: decay, erase read, rank-1 update, and query read.
            # Three channel products cover b*k, w*v, and output scaling.
            recurrence_macs = (
                4 * tokens * heads * head_size * head_size
                + 3 * tokens * heads * head_size
            )
            current_mixer_macs = (
                mixer_projection_macs
                + gate_projection_macs
                + lambda_controller_macs
                + recurrence_macs
            )
            block_types.append("gdn2_memory")
        else:
            attention_projections = 4 * tokens * hidden * hidden
            attention_products = 2 * tokens * tokens * hidden
            current_mixer_macs = attention_projections + attention_products
            block_types.append("softmax_attention")
        ffn = 2 * tokens * hidden * mlp_hidden
        ada_ln = 6 * hidden * hidden
        mixer_macs.append(current_mixer_macs)
        block_macs.append(current_mixer_macs + ffn + ada_ln)
    final_macs = 2 * hidden * hidden + tokens * hidden * (patch * patch * channels)
    total = patch_macs + timestep_macs + sum(block_macs) + final_macs
    trainable = _parameter_count(model, trainable_only=True)
    all_parameters = _parameter_count(model, trainable_only=False)
    mlp_parameters = sum(_parameter_count(block.mlp, True) for block in model.blocks)

    return {
        "image_size": config.image_size,
        "patch_size": patch,
        "tokens": tokens,
        "hidden_size": hidden,
        "depth": config.depth,
        "num_heads": config.num_heads,
        "allocation": config.allocation.kind,
        "allocation_strength": config.allocation.strength,
        "block_types": block_types,
        "memory_block_indices": list(model.memory_block_indices),
        "scan_directions": list(model.scan_directions),
        "gate_mode": (
            config.memory.gate_mode if config.memory.kind == "hybrid_gdn2" else None
        ),
        "memory_backend": (
            config.memory.backend if config.memory.kind == "hybrid_gdn2" else None
        ),
        "recurrent_state_shape": (
            [
                config.num_heads,
                config.hidden_size // config.num_heads,
                config.hidden_size // config.num_heads,
            ]
            if config.memory.kind == "hybrid_gdn2"
            else None
        ),
        "mlp_hidden_sizes": list(model.mlp_hidden_sizes),
        "mlp_hidden_sum": sum(model.mlp_hidden_sizes),
        "parameters_trainable": trainable,
        "parameters_total": all_parameters,
        "parameters_mlp": mlp_parameters,
        "macs_per_image": total,
        "gmacs_per_image": total / 1e9,
        "gflops_fma2": 2 * total / 1e9,
        "mixer_macs": mixer_macs,
        "block_macs": block_macs,
        "counting_convention": COUNTING_CONVENTION,
    }


def assert_exact_match(stats: list[dict[str, Any]]) -> None:
    if len(stats) < 2:
        raise ValueError("at least two models are required for matching")
    expected_parameters = stats[0]["parameters_trainable"]
    expected_macs = stats[0]["macs_per_image"]
    mismatches = []
    for index, item in enumerate(stats[1:], start=1):
        if item["parameters_trainable"] != expected_parameters:
            mismatches.append(
                f"model {index} params={item['parameters_trainable']} != {expected_parameters}"
            )
        if item["macs_per_image"] != expected_macs:
            mismatches.append(f"model {index} MACs={item['macs_per_image']} != {expected_macs}")
    if mismatches:
        raise AssertionError("; ".join(mismatches))
