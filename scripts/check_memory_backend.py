#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import re

import torch
import torch.nn.functional as F

from _bootstrap import ROOT  # noqa: F401
from dit_research.memory import NoiseAdaptiveGatedDeltaMixer, recurrent_gdn2_reference


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare the optional FLA GDN2 kernel with the FP32 reference"
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--heads", type=int, default=6)
    parser.add_argument("--head-size", type=int, default=64)
    parser.add_argument("--atol", type=float, default=0.03)
    parser.add_argument("--rtol", type=float, default=0.03)
    parser.add_argument("--max-gradient-relative-l2", type=float, default=0.1)
    parser.add_argument(
        "--require-sm120",
        action="store_true",
        help="fail unless the active GPU is compute capability 12.0 and torch includes sm_120",
    )
    return parser.parse_args()


def error_stats(expected: torch.Tensor, actual: torch.Tensor) -> dict[str, float]:
    difference = (expected.float() - actual.float()).abs()
    return {
        "max_abs": float(difference.max()),
        "mean_abs": float(difference.mean()),
        "relative_l2": float(
            torch.linalg.vector_norm(difference)
            / torch.linalg.vector_norm(expected.float()).clamp_min(1e-12)
        ),
    }


def major_minor(version: str) -> tuple[int, int]:
    match = re.match(r"^(\d+)\.(\d+)", version)
    if match is None:
        raise ValueError(f"cannot parse package version: {version}")
    return int(match.group(1)), int(match.group(2))


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the FLA backend parity check")
    if min(args.batch_size, args.tokens, args.heads, args.head_size) <= 0:
        raise ValueError("all shape arguments must be positive")
    if args.head_size > 256:
        raise ValueError("FLA GDN2 supports head-size <= 256")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("the active CUDA device does not support BF16")
    if major_minor(torch.__version__) < (2, 7):
        raise RuntimeError("fla-core 0.5.1 CUDA kernels require torch>=2.7")
    triton_version = importlib.metadata.version("triton")
    if major_minor(triton_version) < (3, 3):
        raise RuntimeError("fla-core 0.5.1 CUDA kernels require triton>=3.3")
    fla_version = importlib.metadata.version("fla-core")
    if fla_version != "0.5.1":
        raise RuntimeError(
            f"this experiment pins fla-core==0.5.1; installed version is {fla_version}"
        )
    capability = torch.cuda.get_device_capability()
    arch_list = torch.cuda.get_arch_list()
    if args.require_sm120 and (capability != (12, 0) or "sm_120" not in arch_list):
        raise RuntimeError(
            f"expected active SM120 Blackwell support; capability={capability}, "
            f"arch_list={arch_list}"
        )

    from fla.ops.gdn2 import chunk_gdn2

    device = torch.device("cuda")
    dtype = torch.bfloat16
    generator = torch.Generator(device=device).manual_seed(20260806)
    shape = (args.batch_size, args.tokens, args.heads, args.head_size)

    def random_tensor() -> torch.Tensor:
        return torch.randn(shape, device=device, dtype=dtype, generator=generator)

    base = [random_tensor() for _ in range(3)]
    # Exercise the epsilon-dominated normalization regime as well as normal inputs.
    base[0][:, 0].zero_()
    base[1][:, 0].zero_()
    if args.tokens > 1:
        base[0][:, 1].mul_(1e-4)
        base[1][:, 1].mul_(1e-4)
    base.extend(
        [
            -F.softplus(random_tensor().float()).to(dtype),
            torch.sigmoid(random_tensor()),
            torch.sigmoid(random_tensor()),
        ]
    )
    reference_inputs = [item.detach().clone().requires_grad_(True) for item in base]
    fused_inputs = [item.detach().clone().requires_grad_(True) for item in base]

    reference_output, reference_state = recurrent_gdn2_reference(*reference_inputs)
    fused_output, fused_state = chunk_gdn2(
        *(item.contiguous() for item in fused_inputs),
        scale=args.head_size**-0.5,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=False,
    )
    if fused_state is None:
        raise AssertionError("FLA did not return the requested final state")

    upstream = torch.randn(
        reference_output.shape,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    reference_gradients = torch.autograd.grad(
        reference_output, reference_inputs, upstream, retain_graph=False
    )
    fused_gradients = torch.autograd.grad(
        fused_output, fused_inputs, upstream, retain_graph=False
    )

    torch.testing.assert_close(
        fused_output.float(),
        reference_output.float(),
        atol=args.atol,
        rtol=args.rtol,
    )
    torch.testing.assert_close(
        fused_state.float(),
        reference_state.float(),
        atol=args.atol,
        rtol=args.rtol,
    )
    gradient_errors = {}
    for name, expected, actual in zip(
        ("q", "k", "v", "g", "b", "w"),
        reference_gradients,
        fused_gradients,
        strict=True,
    ):
        if not torch.isfinite(actual).all():
            raise FloatingPointError(f"non-finite fused gradient for {name}")
        torch.testing.assert_close(
            actual.float(), expected.float(), atol=args.atol, rtol=args.rtol
        )
        gradient_errors[name] = error_stats(expected, actual)

    grid_size = math.isqrt(args.tokens)
    if grid_size * grid_size != args.tokens:
        raise ValueError("full-mixer parity requires tokens to be a square 2D grid")
    hidden_size = args.heads * args.head_size
    torch.manual_seed(20260806)
    reference_mixer = NoiseAdaptiveGatedDeltaMixer(
        hidden_size=hidden_size,
        num_heads=args.heads,
        grid_size=grid_size,
        direction="tb",
        gate_mode="adaptive",
        gate_rank=67,
        lambda_hidden_size=16,
        backend="reference",
    ).to(device)
    fused_mixer = NoiseAdaptiveGatedDeltaMixer(
        hidden_size=hidden_size,
        num_heads=args.heads,
        grid_size=grid_size,
        direction="tb",
        gate_mode="adaptive",
        gate_rank=67,
        lambda_hidden_size=16,
        backend="fla",
    ).to(device)
    fused_mixer.load_state_dict(reference_mixer.state_dict())
    mixer_input = torch.randn(
        args.batch_size,
        args.tokens,
        hidden_size,
        device=device,
        generator=generator,
    )
    reference_x = mixer_input.detach().clone().requires_grad_(True)
    fused_x = mixer_input.detach().clone().requires_grad_(True)
    normalized_log_snr = torch.linspace(
        -2.0, 2.0, args.batch_size, device=device, dtype=torch.float32
    )
    with torch.autocast(device_type="cuda", dtype=dtype):
        reference_mixed = reference_mixer(reference_x, normalized_log_snr)
        fused_mixed = fused_mixer(fused_x, normalized_log_snr)
    mixer_upstream = torch.randn(
        reference_mixed.shape,
        device=device,
        dtype=reference_mixed.dtype,
        generator=generator,
    )
    reference_parameters = tuple(reference_mixer.parameters())
    fused_parameters = tuple(fused_mixer.parameters())
    reference_mixer_gradients = torch.autograd.grad(
        reference_mixed,
        (reference_x, *reference_parameters),
        mixer_upstream,
    )
    fused_mixer_gradients = torch.autograd.grad(
        fused_mixed,
        (fused_x, *fused_parameters),
        mixer_upstream,
    )
    torch.testing.assert_close(
        fused_mixed.float(),
        reference_mixed.float(),
        atol=args.atol,
        rtol=args.rtol,
    )
    full_gradient_errors = {}
    gradient_names = ("input", *dict(reference_mixer.named_parameters()).keys())
    for name, expected, actual in zip(
        gradient_names,
        reference_mixer_gradients,
        fused_mixer_gradients,
        strict=True,
    ):
        stats = error_stats(expected, actual)
        if not torch.isfinite(actual).all():
            raise FloatingPointError(f"non-finite full-mixer fused gradient for {name}")
        expected_norm = float(torch.linalg.vector_norm(expected.float()))
        if expected_norm >= 1e-8 and stats["relative_l2"] > args.max_gradient_relative_l2:
            raise AssertionError(
                f"full-mixer gradient {name} relative_l2={stats['relative_l2']:.6f} "
                f"> {args.max_gradient_relative_l2}"
            )
        if expected_norm < 1e-8 and stats["max_abs"] > args.atol:
            raise AssertionError(
                f"full-mixer near-zero gradient {name} max_abs={stats['max_abs']:.6f} "
                f"> {args.atol}"
            )
        full_gradient_errors[name] = stats

    payload = {
        "status": "passed",
        "shape": list(shape),
        "dtype": str(dtype),
        "atol": args.atol,
        "rtol": args.rtol,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "triton": triton_version,
        "fla_core": fla_version,
        "gpu": torch.cuda.get_device_name(device),
        "capability": list(capability),
        "arch_list": arch_list,
        "output_error": error_stats(reference_output, fused_output),
        "state_error": error_stats(reference_state, fused_state),
        "gradient_errors": gradient_errors,
        "full_mixer_output_error": error_stats(reference_mixed, fused_mixed),
        "full_mixer_gradient_errors": full_gradient_errors,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
