from __future__ import annotations

import math
from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


SCAN_DIRECTIONS = ("lr", "rl", "tb", "bt")


def make_scan_order(grid_size: int, direction: str) -> tuple[Tensor, Tensor]:
    """Return a 2D scan permutation and its exact inverse.

    ``lr``/``rl`` scan rows and ``tb``/``bt`` scan columns.  The returned
    tensors index a canonical row-major token sequence.
    """

    if grid_size <= 0:
        raise ValueError("grid_size must be positive")
    if direction not in SCAN_DIRECTIONS:
        raise ValueError(f"unsupported scan direction: {direction}")
    grid = torch.arange(grid_size * grid_size).reshape(grid_size, grid_size)
    if direction == "lr":
        order = grid.reshape(-1)
    elif direction == "rl":
        order = grid.flip(1).reshape(-1)
    elif direction == "tb":
        order = grid.transpose(0, 1).reshape(-1)
    else:
        order = grid.transpose(0, 1).flip(1).reshape(-1)
    inverse = torch.empty_like(order)
    inverse[order] = torch.arange(order.numel())
    return order, inverse


def recurrent_gdn2_reference(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    g: Tensor,
    b: Tensor,
    w: Tensor,
) -> tuple[Tensor, Tensor]:
    """Differentiable FP32 reference for the GDN2 matrix-state recurrence.

    Shapes are ``q,k,g,b=[B,T,H,K]`` and ``v,w=[B,T,H,V]``.  The state is
    intentionally local to this call; it is never cached across diffusion
    denoiser evaluations.
    """

    if q.shape != k.shape or q.shape != g.shape or q.shape != b.shape:
        raise ValueError("q, k, g, and b must have identical shapes")
    if v.shape != w.shape or v.shape[:3] != q.shape[:3]:
        raise ValueError("v/w must match q on batch, sequence, and heads")
    if q.ndim != 4:
        raise ValueError("GDN2 inputs must have shape [B,T,H,D]")

    output_dtype = q.dtype
    batch, length, heads, key_dim = q.shape
    value_dim = v.shape[-1]
    q_float = q.float()
    k_float = k.float()
    q_float = q_float * torch.rsqrt(
        q_float.square().sum(dim=-1, keepdim=True) + 1e-6
    )
    k_float = k_float * torch.rsqrt(
        k_float.square().sum(dim=-1, keepdim=True) + 1e-6
    )
    q_float = q_float * (key_dim**-0.5)
    v_float = v.float()
    g_float = g.float()
    b_float = b.float()
    w_float = w.float()
    state = torch.zeros(
        batch,
        heads,
        key_dim,
        value_dim,
        device=q.device,
        dtype=torch.float32,
    )
    outputs: list[Tensor] = []
    for token_index in range(length):
        state = state * torch.exp(g_float[:, token_index]).unsqueeze(-1)
        erased_value = torch.einsum(
            "bhk,bhkv->bhv",
            b_float[:, token_index] * k_float[:, token_index],
            state,
        )
        value_update = w_float[:, token_index] * v_float[:, token_index] - erased_value
        state = state + torch.einsum(
            "bhk,bhv->bhkv",
            k_float[:, token_index],
            value_update,
        )
        outputs.append(
            torch.einsum("bhk,bhkv->bhv", q_float[:, token_index], state)
        )
    return torch.stack(outputs, dim=1).to(output_dtype), state


def _load_fla_kernel() -> Callable[..., tuple[Tensor, Tensor | None]]:
    try:
        from fla.ops.gdn2 import chunk_gdn2
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "memory backend 'fla' requires fla-core with GDN2 support; "
            "install the pinned optional dependency and run the parity test"
        ) from exc
    return chunk_gdn2


class NoiseAdaptiveGatedDeltaMixer(nn.Module):
    """Four-direction spatial GDN2 mixer with controllable gate separation.

    The same projections and lambda controller are evaluated in all four
    modes, so ``coupled``, ``separated``, ``static``, and ``adaptive`` have
    identical nominal trainable parameter counts and analytic dense-MAC paths.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        num_heads: int,
        grid_size: int,
        direction: str,
        gate_mode: str,
        gate_rank: int,
        lambda_hidden_size: int,
        backend: str,
    ) -> None:
        super().__init__()
        if hidden_size % num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        if gate_mode not in {"coupled", "separated", "static", "adaptive"}:
            raise ValueError(f"unsupported gate mode: {gate_mode}")
        if backend not in {"reference", "fla"}:
            raise ValueError(f"unsupported memory backend: {backend}")
        if min(gate_rank, lambda_hidden_size) <= 0:
            raise ValueError("gate/controller hidden sizes must be positive")

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads
        self.grid_size = grid_size
        self.direction = direction
        self.gate_mode = gate_mode
        self.gate_rank = gate_rank
        self.lambda_hidden_size = lambda_hidden_size
        self.backend = backend

        order, inverse = make_scan_order(grid_size, direction)
        self.register_buffer("scan_order", order, persistent=False)
        self.register_buffer("inverse_scan_order", inverse, persistent=False)

        self.qkv = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        self.gate_in = nn.Linear(hidden_size, gate_rank, bias=False)
        # decay, independent erase, independent write, and output gate
        self.gate_out = nn.Linear(gate_rank, 4 * hidden_size, bias=True)
        self.output_projection = nn.Linear(hidden_size, hidden_size, bias=False)
        self.lambda_mlp = nn.Sequential(
            nn.Linear(1, lambda_hidden_size),
            nn.SiLU(),
            nn.Linear(lambda_hidden_size, 1),
        )

        self.A_log = nn.Parameter(
            torch.log(torch.empty(num_heads, dtype=torch.float32).uniform_(1.0, 16.0))
        )
        dt = torch.exp(
            torch.rand(hidden_size, dtype=torch.float32)
            * (math.log(0.1) - math.log(0.001))
            + math.log(0.001)
        ).clamp_min(1e-4)
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
        self.A_log._no_weight_decay = True  # type: ignore[attr-defined]
        self.dt_bias._no_weight_decay = True  # type: ignore[attr-defined]

        nn.init.zeros_(self.lambda_mlp[-1].weight)
        nn.init.zeros_(self.lambda_mlp[-1].bias)
        self.collect_diagnostics = False
        self.lambda_override: float | None = None
        self._latest_diagnostics: dict[str, Tensor] = {}

    def decoupling_strength(self, normalized_log_snr: Tensor) -> Tensor:
        """Map normalized log-SNR ``[B]`` to lambda ``[B,1,1,1]``."""

        if normalized_log_snr.ndim != 1:
            raise ValueError("normalized_log_snr must have shape [B]")
        controller_input = normalized_log_snr.float()
        if self.gate_mode == "static":
            # A constant one keeps the controller noise-independent while
            # allowing both linear layers to receive gradients.  A zero input
            # would leave the first layer's weight mathematically inactive.
            controller_input = torch.ones_like(controller_input)
        learned = torch.sigmoid(self.lambda_mlp(controller_input.unsqueeze(-1)))
        if self.gate_mode == "coupled":
            learned = learned * 0.0
        elif self.gate_mode == "separated":
            learned = learned * 0.0 + 1.0
        if self.lambda_override is not None:
            learned = learned * 0.0 + self.lambda_override
        return learned.reshape(-1, 1, 1, 1)

    def set_lambda_override(self, value: float | None) -> None:
        if value is not None and not 0.0 <= value <= 1.0:
            raise ValueError("lambda override must be in [0, 1]")
        self.lambda_override = value

    def set_diagnostics(self, enabled: bool) -> None:
        self.collect_diagnostics = bool(enabled)
        if not enabled:
            self._latest_diagnostics = {}

    def latest_diagnostics(self) -> dict[str, Tensor]:
        return dict(self._latest_diagnostics)

    def _fla_mix(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        g: Tensor,
        b: Tensor,
        w: Tensor,
    ) -> tuple[Tensor, Tensor | None]:
        if not q.is_cuda:
            raise RuntimeError("the FLA GDN2 backend requires a CUDA tensor")
        if q.dtype not in {torch.float16, torch.bfloat16}:
            raise RuntimeError(
                "the FLA GDN2 backend must run under FP16 or BF16 autocast"
            )
        kernel = _load_fla_kernel()
        return kernel(
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            g.to(dtype=q.dtype).contiguous(),
            b.to(dtype=q.dtype).contiguous(),
            w.to(dtype=v.dtype).contiguous(),
            scale=self.head_size**-0.5,
            output_final_state=self.collect_diagnostics,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=False,
        )

    def forward(self, x: Tensor, normalized_log_snr: Tensor | None = None) -> Tensor:
        if x.ndim != 3 or x.shape[1:] != (
            self.grid_size * self.grid_size,
            self.hidden_size,
        ):
            raise ValueError(
                f"expected memory input [B,{self.grid_size ** 2},{self.hidden_size}], "
                f"received {tuple(x.shape)}"
            )
        if normalized_log_snr is None or normalized_log_snr.shape != (x.shape[0],):
            raise ValueError("memory mixer requires normalized log-SNR with shape [B]")

        scanned = x.index_select(1, self.scan_order)
        batch, tokens, _ = scanned.shape
        qkv = self.qkv(scanned).reshape(
            batch, tokens, 3, self.num_heads, self.head_size
        )
        q, k, v = qkv.unbind(dim=2)

        gate_features = F.silu(self.gate_in(scanned))
        gate_values = self.gate_out(gate_features).reshape(
            batch, tokens, 4, self.num_heads, self.head_size
        )
        raw_decay, erase_independent, write_independent, output_gate = gate_values.unbind(
            dim=2
        )

        separation = self.decoupling_strength(normalized_log_snr)
        shared = 0.5 * (erase_independent + write_independent)
        delta = 0.5 * (erase_independent - write_independent)
        erase = torch.sigmoid(shared + separation * delta)
        write = torch.sigmoid(shared - separation * delta)

        decay_rate = torch.exp(self.A_log.float()).reshape(1, 1, self.num_heads, 1)
        dt_bias = self.dt_bias.float().reshape(1, 1, self.num_heads, self.head_size)
        log_decay = -decay_rate * F.softplus(raw_decay.float() + dt_bias)
        log_decay = log_decay.clamp_min(-20.0)

        if self.backend == "reference":
            mixed, final_state = recurrent_gdn2_reference(
                q, k, v, log_decay, erase, write
            )
        else:
            mixed, final_state = self._fla_mix(q, k, v, log_decay, erase, write)

        mixed_float = mixed.float()
        mixed_float = mixed_float * torch.rsqrt(
            mixed_float.square().mean(dim=-1, keepdim=True) + 1e-6
        )
        mixed_float = mixed_float * torch.sigmoid(output_gate.float())
        projected = self.output_projection(
            mixed_float.reshape(batch, tokens, self.hidden_size).to(dtype=x.dtype)
        )
        restored = projected.index_select(1, self.inverse_scan_order)

        if self.collect_diagnostics:
            diagnostics = {
                "lambda": separation.detach().reshape(batch),
                "erase": erase.detach().mean(dim=(1, 2, 3)),
                "write": write.detach().mean(dim=(1, 2, 3)),
                "erase_write_abs_gap": (erase.detach() - write.detach())
                .abs()
                .mean(dim=(1, 2, 3)),
                "decay": torch.exp(log_decay.detach()).mean(dim=(1, 2, 3)),
                "gate_saturation_fraction": (
                    ((erase.detach() < 0.01) | (erase.detach() > 0.99))
                    .float()
                    .mean(dim=(1, 2, 3))
                    + ((write.detach() < 0.01) | (write.detach() > 0.99))
                    .float()
                    .mean(dim=(1, 2, 3))
                )
                * 0.5,
                "output_rms": mixed_float.detach()
                .square()
                .mean(dim=(1, 2, 3))
                .sqrt(),
            }
            if final_state is not None:
                diagnostics["state_rms"] = (
                    final_state.detach().square().mean(dim=(1, 2, 3)).sqrt()
                )
            self._latest_diagnostics = diagnostics
        return restored

    def extra_repr(self) -> str:
        return (
            f"hidden_size={self.hidden_size}, heads={self.num_heads}, "
            f"direction={self.direction}, gate_mode={self.gate_mode}, "
            f"backend={self.backend}"
        )
