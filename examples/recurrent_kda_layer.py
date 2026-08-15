# Copyright (c) 2026 Tianjin University, Ltd.

"""Complete recurrent Kimi Delta Attention layer example on Ascend NPU."""

from __future__ import annotations

import argparse
import math
import os

# Keep --help and argument validation independent from NPU initialization.
os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

import torch
import torch.nn as nn
import torch.nn.functional as F


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a complete recurrent KDA decode/MTP layer on Ascend NPU."
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument(
        "--mtp",
        type=int,
        default=1,
        help="Tokens processed per sequence in this decode step; supported range is [1, 8].",
    )
    parser.add_argument("--hidden-size", type=int, default=1024)
    parser.add_argument("--query-heads", type=int, default=4)
    parser.add_argument("--value-heads", type=int, default=8)
    parser.add_argument("--key-dim", type=int, default=128)
    parser.add_argument("--value-dim", type=int, choices=(128, 256), default=128)
    parser.add_argument(
        "--use-short-conv",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--conv-kernel", type=int, choices=(2, 3, 4), default=4)
    parser.add_argument("--conv-bias", action="store_true")
    parser.add_argument(
        "--conv-state-capacity",
        type=int,
        default=None,
        help="Number of slots in the fused QKV convolution cache; inferred by default.",
    )
    parser.add_argument(
        "--cache-indices",
        type=int,
        nargs="+",
        default=None,
        metavar="SLOT",
        help="Convolution cache slot for each sequence; defaults to 0..batch-1.",
    )
    parser.add_argument("--state-dtype", choices=("bf16", "fp32"), default="fp32")
    parser.add_argument(
        "--state-capacity",
        type=int,
        default=None,
        help="Number of recurrent state slots; inferred from state indices by default.",
    )
    parser.add_argument(
        "--ssm-state-indices",
        type=int,
        nargs="+",
        default=None,
        metavar="SLOT",
        help="Packed recurrent state slot for each token; defaults to sequence-owned state.",
    )
    parser.add_argument(
        "--num-accepted-tokens",
        type=int,
        nargs="+",
        default=None,
        metavar="COUNT",
        help="Accepted tokens per sequence; defaults to accepting all input tokens.",
    )
    parser.add_argument("--safe-gate", action="store_true")
    parser.add_argument("--lower-bound", type=float, default=-5.0)
    parser.add_argument("--allow-neg-eigval", action="store_true")
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def _sequence_slot_sets(indices: list[int], batch: int, mtp: int) -> list[set[int]]:
    return [set(indices[index * mtp : (index + 1) * mtp]) for index in range(batch)]


def _validate_args(args: argparse.Namespace) -> None:
    if args.batch <= 0:
        raise ValueError(f"batch must be positive, got {args.batch}")
    if not 1 <= args.mtp <= 8:
        raise ValueError(f"mtp must be in [1, 8], got {args.mtp}")
    if args.hidden_size <= 0:
        raise ValueError(f"hidden_size must be positive, got {args.hidden_size}")
    if not 1 <= args.query_heads <= 256 or not 1 <= args.value_heads <= 256:
        raise ValueError("query_heads and value_heads must be in [1, 256]")
    if args.value_heads % args.query_heads != 0:
        raise ValueError(
            "value_heads must be a multiple of query_heads, "
            f"got {args.value_heads} and {args.query_heads}"
        )
    if args.key_dim != 128:
        raise ValueError(f"recurrent_kda currently requires key_dim=128, got {args.key_dim}")
    if args.steps <= 0:
        raise ValueError(f"steps must be positive, got {args.steps}")
    if args.safe_gate and not -5.0 <= args.lower_bound < 0.0:
        raise ValueError(
            f"safe_gate requires lower_bound in [-5, 0), got {args.lower_bound}"
        )
    if args.use_short_conv and args.mtp > 1 and args.conv_kernel != 4:
        raise ValueError(
            "causal_conv1d speculative decode currently requires conv_kernel=4"
        )

    if not args.use_short_conv and (
        args.conv_state_capacity is not None
        or args.cache_indices is not None
        or args.conv_bias
    ):
        raise ValueError(
            "conv_state_capacity, cache_indices and conv_bias require --use-short-conv"
        )
    if args.use_short_conv:
        cache_indices = args.cache_indices or list(range(args.batch))
        if len(cache_indices) != args.batch:
            raise ValueError(
                "cache_indices must contain one slot per sequence, "
                f"expected {args.batch}, got {len(cache_indices)}"
            )
        if any(index < 0 for index in cache_indices):
            raise ValueError(f"cache_indices must be non-negative, got {cache_indices}")
        if len(set(cache_indices)) != len(cache_indices):
            raise ValueError("active sequences must use distinct cache_indices")
        if (
            args.conv_state_capacity is not None
            and max(cache_indices) >= args.conv_state_capacity
        ):
            raise ValueError(
                "cache_indices entries must be smaller than conv_state_capacity, "
                f"got capacity {args.conv_state_capacity} and indices {cache_indices}"
            )

    total_tokens = args.batch * args.mtp
    state_indices = args.ssm_state_indices
    if state_indices is None:
        if args.state_capacity is not None and args.state_capacity != args.batch:
            raise ValueError(
                "without ssm_state_indices, state_capacity must equal batch "
                f"({args.batch}), got {args.state_capacity}"
            )
    else:
        if len(state_indices) != total_tokens:
            raise ValueError(
                "ssm_state_indices must contain one slot per packed token, "
                f"expected {total_tokens}, got {len(state_indices)}"
            )
        if any(index < 0 for index in state_indices):
            raise ValueError(
                f"ssm_state_indices must be non-negative, got {state_indices}"
            )
        if (
            args.state_capacity is not None
            and max(state_indices) >= args.state_capacity
        ):
            raise ValueError(
                "ssm_state_indices entries must be smaller than state_capacity, "
                f"got capacity {args.state_capacity} and indices {state_indices}"
            )
        slot_sets = _sequence_slot_sets(state_indices, args.batch, args.mtp)
        for left in range(args.batch):
            for right in range(left + 1, args.batch):
                overlap = slot_sets[left] & slot_sets[right]
                if overlap:
                    raise ValueError(
                        "active sequences must not share recurrent state slots, "
                        f"but sequences {left} and {right} share {sorted(overlap)}"
                    )

    accepted = args.num_accepted_tokens
    if accepted is not None:
        if len(accepted) != args.batch:
            raise ValueError(
                "num_accepted_tokens must contain one value per sequence, "
                f"expected {args.batch}, got {len(accepted)}"
            )
        if any(value < 1 or value > args.mtp for value in accepted):
            raise ValueError(
                f"num_accepted_tokens values must be in [1, {args.mtp}], got {accepted}"
            )
        if state_indices is None and any(value != args.mtp for value in accepted):
            raise ValueError(
                "partial MTP acceptance requires ssm_state_indices so recurrent_kda "
                "can select the committed state slot"
            )


def _tensor_changed(before: torch.Tensor, after: torch.Tensor) -> bool:
    return not bool(torch.equal(before, after))


class SigmoidGatedRMSNorm(nn.Module):
    def __init__(self, head_dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(head_dim))
        self.eps = eps

    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        normalized = x.float()
        normalized = normalized * torch.rsqrt(
            normalized.square().mean(dim=-1, keepdim=True) + self.eps
        )
        normalized = normalized * self.weight.float()
        return (normalized * torch.sigmoid(gate.float())).to(input_dtype)


class RecurrentKdaLayer(nn.Module):
    """A complete stateful KDA mixer for decode and short MTP inputs."""

    def __init__(
        self,
        hidden_size: int,
        *,
        query_heads: int,
        value_heads: int,
        key_dim: int,
        value_dim: int,
        use_short_conv: bool,
        conv_kernel: int,
        conv_bias: bool,
        safe_gate: bool,
        lower_bound: float,
        allow_neg_eigval: bool,
    ):
        super().__init__()
        self.query_heads = query_heads
        self.value_heads = value_heads
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.use_short_conv = use_short_conv
        self.safe_gate = safe_gate
        self.lower_bound = lower_bound
        self.allow_neg_eigval = allow_neg_eigval

        query_size = query_heads * key_dim
        value_size = value_heads * value_dim
        gate_size = value_heads * key_dim
        conv_size = 2 * query_size + value_size
        self.query_size = query_size
        self.value_size = value_size

        self.in_proj_qkv = nn.Linear(hidden_size, conv_size, bias=False)
        if use_short_conv:
            self.conv_weight = nn.Parameter(torch.empty(conv_kernel, conv_size))
            nn.init.uniform_(
                self.conv_weight,
                -1.0 / math.sqrt(conv_kernel),
                1.0 / math.sqrt(conv_kernel),
            )
            self.conv_bias = (
                nn.Parameter(torch.zeros(conv_size)) if conv_bias else None
            )

        self.in_proj_a = nn.Sequential(
            nn.Linear(hidden_size, value_dim, bias=False),
            nn.Linear(value_dim, gate_size, bias=False),
        )
        self.in_proj_b = nn.Linear(hidden_size, value_heads, bias=False)
        self.in_proj_z = nn.Sequential(
            nn.Linear(hidden_size, value_dim, bias=False),
            nn.Linear(value_dim, value_size, bias=True),
        )

        if safe_gate:
            self.A_log = nn.Parameter(torch.zeros(value_heads, dtype=torch.float32))
        else:
            initial_a = torch.empty(value_heads, dtype=torch.float32).uniform_(1, 16)
            self.A_log = nn.Parameter(torch.log(initial_a))
        dt = torch.exp(
            torch.rand(gate_size, dtype=torch.float32)
            * (math.log(0.1) - math.log(0.001))
            + math.log(0.001)
        ).clamp(min=1e-4)
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))

        self.norm = SigmoidGatedRMSNorm(value_dim)
        self.out_proj = nn.Linear(value_size, hidden_size, bias=False)

    def _short_conv(
        self,
        x: torch.Tensor,
        state: torch.Tensor,
        *,
        cache_indices: list[int],
        accepted_tokens_host: list[int] | None,
    ) -> torch.Tensor:
        from fla_npu.ops.ascendc import causal_conv1d

        return causal_conv1d(
            x,
            self.conv_weight,
            bias=self.conv_bias,
            conv_states=state,
            cache_indices=cache_indices,
            num_accepted_tokens=accepted_tokens_host,
            activation_mode=1,
            pad_slot_id=-1,
            run_mode=1,
            head_num=0,
        )

    @torch.no_grad()
    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        conv_state: torch.Tensor | None,
        recurrent_state: torch.Tensor,
        cache_indices: list[int],
        accepted_tokens_host: list[int] | None,
        cu_seqlens: torch.Tensor,
        ssm_state_indices: torch.Tensor | None,
        num_accepted_tokens: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from fla_npu.ops.ascendc import recurrent_kda

        batch, mtp, _ = hidden_states.shape
        mixed_qkv = self.in_proj_qkv(hidden_states)
        z = self.in_proj_z(hidden_states).reshape(
            batch, mtp, self.value_heads, self.value_dim
        )
        b = self.in_proj_b(hidden_states)
        a = self.in_proj_a(hidden_states)

        if self.use_short_conv:
            if conv_state is None:
                raise ValueError(
                    "the fused QKV convolution state is required when short conv is enabled"
                )
            mixed_qkv = self._short_conv(
                mixed_qkv,
                conv_state,
                cache_indices=cache_indices,
                accepted_tokens_host=accepted_tokens_host,
            )
        else:
            mixed_qkv = F.silu(mixed_qkv)

        query, key, value = torch.split(
            mixed_qkv,
            (self.query_size, self.query_size, self.value_size),
            dim=-1,
        )

        query = query.reshape(batch, mtp, self.query_heads, self.key_dim).contiguous()
        key = key.reshape(batch, mtp, self.query_heads, self.key_dim).contiguous()
        value = value.reshape(batch, mtp, self.value_heads, self.value_dim).contiguous()
        raw_gate = a.reshape(
            batch, mtp, self.value_heads, self.key_dim
        )
        raw_beta = b

        recurrent_out, final_state = recurrent_kda(
            query,
            key,
            value,
            raw_gate,
            raw_beta,
            recurrent_state,
            cu_seqlens=cu_seqlens,
            ssm_state_indices=ssm_state_indices,
            A_log=self.A_log,
            dt_bias=self.dt_bias.reshape(self.value_heads, self.key_dim),
            num_accepted_tokens=num_accepted_tokens,
            layout="BSND",
            scale=self.key_dim**-0.5,
            output_final_state=True,
            inplace_final_state=True,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            use_beta_sigmoid_in_kernel=True,
            allow_neg_eigval=self.allow_neg_eigval,
            safe_gate=self.safe_gate,
            lower_bound=self.lower_bound,
            state_v_first=True,
        )
        if final_state is None:
            raise RuntimeError("recurrent_kda did not return the requested final state")

        recurrent_out = self.norm(recurrent_out, z)
        output = self.out_proj(recurrent_out.reshape(batch, mtp, -1))
        return output, final_state


def main() -> None:
    args = _parse_args()
    _validate_args(args)

    # Load the packaged custom OPP before torch_npu exposes same-named CANN ops.
    import fla_npu.ops.ascendc  # noqa: F401
    import torch_npu

    device = torch.device(f"npu:{args.device}")
    torch_npu.npu.set_device(device)
    torch.npu.set_compile_mode(jit_compile=False)
    torch.manual_seed(args.seed)
    torch.npu.manual_seed_all(args.seed)

    batch, mtp = args.batch, args.mtp
    total_tokens = batch * mtp
    cache_indices = args.cache_indices or list(range(batch))
    accepted_values = args.num_accepted_tokens or [mtp] * batch
    conv_state_capacity = args.conv_state_capacity or (max(cache_indices) + 1)
    state_indices = args.ssm_state_indices
    state_capacity = (
        args.state_capacity
        if args.state_capacity is not None
        else (max(state_indices) + 1 if state_indices is not None else batch)
    )
    state_dtype = torch.bfloat16 if args.state_dtype == "bf16" else torch.float32

    layer = RecurrentKdaLayer(
        args.hidden_size,
        query_heads=args.query_heads,
        value_heads=args.value_heads,
        key_dim=args.key_dim,
        value_dim=args.value_dim,
        use_short_conv=args.use_short_conv,
        conv_kernel=args.conv_kernel,
        conv_bias=args.conv_bias,
        safe_gate=args.safe_gate,
        lower_bound=args.lower_bound,
        allow_neg_eigval=args.allow_neg_eigval,
    ).to(device=device, dtype=torch.bfloat16)
    # The model runs in BF16, while recurrent_kda requires gate parameters in FP32.
    layer.A_log.data = layer.A_log.data.float()
    layer.dt_bias.data = layer.dt_bias.data.float()
    layer.eval()
    if layer.A_log.dtype != torch.float32 or layer.dt_bias.dtype != torch.float32:
        raise RuntimeError("A_log and dt_bias must remain FP32")

    query_size = args.query_heads * args.key_dim
    value_size = args.value_heads * args.value_dim
    conv_size = 2 * query_size + value_size
    if args.use_short_conv:
        conv_state_len = args.conv_kernel + mtp - 2
        conv_state = torch.zeros(
            conv_state_capacity,
            conv_state_len,
            conv_size,
            dtype=torch.bfloat16,
            device=device,
        )
    else:
        conv_state = None

    recurrent_state = torch.zeros(
        state_capacity,
        args.value_heads,
        args.value_dim,
        args.key_dim,
        dtype=state_dtype,
        device=device,
    )
    cu_seqlens = torch.tensor(
        [index * mtp for index in range(batch + 1)],
        dtype=torch.int32,
        device=device,
    )
    ssm_state_indices = (
        torch.tensor(state_indices, dtype=torch.int32, device=device)
        if state_indices is not None
        else None
    )
    num_accepted_tokens = (
        torch.tensor(accepted_values, dtype=torch.int32, device=device)
        if state_indices is not None
        else None
    )
    accepted_tokens_host = (
        accepted_values if args.use_short_conv and mtp > 1 else None
    )

    print(
        "config:",
        f"B={batch}",
        f"MTP={mtp}",
        f"hidden={args.hidden_size}",
        f"Hk={args.query_heads}",
        f"Hv={args.value_heads}",
        f"K={args.key_dim}",
        f"V={args.value_dim}",
        f"short_conv={args.use_short_conv}",
        f"conv_state_capacity={conv_state_capacity if args.use_short_conv else 0}",
        f"state_capacity={state_capacity}",
        f"cache_indices={cache_indices}",
        f"ssm_state_indices={state_indices}",
        f"num_accepted_tokens={accepted_values}",
        f"safe_gate={args.safe_gate}",
    )

    with torch.inference_mode():
        for step in range(args.steps):
            hidden_states = torch.randn(
                batch,
                mtp,
                args.hidden_size,
                dtype=torch.bfloat16,
                device=device,
            )
            conv_before = conv_state.clone() if conv_state is not None else None
            recurrent_before = recurrent_state.clone()

            output, final_state = layer(
                hidden_states,
                conv_state=conv_state,
                recurrent_state=recurrent_state,
                cache_indices=cache_indices,
                accepted_tokens_host=accepted_tokens_host,
                cu_seqlens=cu_seqlens,
                ssm_state_indices=ssm_state_indices,
                num_accepted_tokens=num_accepted_tokens,
            )
            torch_npu.npu.synchronize()

            conv_changed = (
                "disabled"
                if conv_state is None
                else str(_tensor_changed(conv_before, conv_state))
            )
            print(
                f"step {step}:",
                f"output={tuple(output.shape)} {output.dtype}",
                f"finite={bool(torch.isfinite(output.float()).all().item())}",
                f"qkv_conv_state_changed={conv_changed}",
                f"recurrent_state_changed={_tensor_changed(recurrent_before, recurrent_state)}",
                f"final_state_aliases_input={final_state.data_ptr() == recurrent_state.data_ptr()}",
            )


if __name__ == "__main__":
    main()
