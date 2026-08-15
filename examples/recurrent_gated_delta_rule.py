# Copyright (c) 2026 Tianjin University, Ltd.

"""Complete recurrent Gated DeltaNet layer example on Ascend NPU."""

from __future__ import annotations

import argparse
import math
import os

# Import the NPU backend explicitly in main so --help does not initialize a device.
os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

import torch
import torch.nn as nn
import torch.nn.functional as F


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a complete recurrent GDN decode/MTP layer on Ascend NPU."
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
    parser.add_argument("--value-dim", type=int, default=128)
    parser.add_argument("--conv-kernel", type=int, choices=(2, 3, 4), default=4)
    parser.add_argument("--state-dtype", choices=("bf16", "fp32"), default="fp32")
    parser.add_argument(
        "--conv-state-capacity",
        type=int,
        default=None,
        help="Number of causal-convolution cache slots; inferred from cache indices by default.",
    )
    parser.add_argument(
        "--delta-state-capacity",
        "--state-capacity",
        dest="delta_state_capacity",
        type=int,
        default=None,
        help="Number of recurrent Delta state slots; inferred from state indices by default.",
    )
    parser.add_argument(
        "--cache-indices",
        type=int,
        nargs="+",
        default=None,
        metavar="SLOT",
        help="Causal-convolution cache slot for each sequence; defaults to 0..batch-1.",
    )
    parser.add_argument(
        "--ssm-state-indices",
        type=int,
        nargs="+",
        default=None,
        metavar="SLOT",
        help="Delta state slot for each packed token; defaults to 0..batch*mtp-1.",
    )
    parser.add_argument(
        "--num-accepted-tokens",
        type=int,
        nargs="+",
        default=None,
        metavar="COUNT",
        help="Accepted tokens per sequence; defaults to accepting all tokens.",
    )
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if args.batch <= 0:
        raise ValueError(f"batch must be positive, got {args.batch}")
    if not 1 <= args.mtp <= 8:
        raise ValueError(f"mtp must be in [1, 8], got {args.mtp}")
    if args.hidden_size <= 0:
        raise ValueError(f"hidden_size must be positive, got {args.hidden_size}")
    if args.query_heads <= 0 or args.value_heads <= 0:
        raise ValueError("query_heads and value_heads must be positive")
    if args.query_heads > 256 or args.value_heads > 256:
        raise ValueError("query_heads and value_heads must not exceed 256")
    if args.value_heads % args.query_heads != 0:
        raise ValueError(
            "value_heads must be a multiple of query_heads, "
            f"got {args.value_heads} and {args.query_heads}"
        )
    if not 1 <= args.key_dim <= 512 or not 1 <= args.value_dim <= 512:
        raise ValueError("key_dim and value_dim must be in [1, 512]")
    conv_size = (
        2 * args.query_heads * args.key_dim
        + args.value_heads * args.value_dim
    )
    if conv_size % 16 != 0:
        raise ValueError(
            "the causal_conv1d feature dimension must be a multiple of 16, "
            f"got {conv_size}"
        )
    if args.steps <= 0:
        raise ValueError(f"steps must be positive, got {args.steps}")
    if args.mtp > 1 and args.conv_kernel != 4:
        raise ValueError(
            "causal_conv1d speculative decode currently requires conv_kernel=4"
        )

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
    state_indices = args.ssm_state_indices or list(range(total_tokens))
    if len(state_indices) != total_tokens:
        raise ValueError(
            "ssm_state_indices must contain one slot per packed token, "
            f"expected {total_tokens}, got {len(state_indices)}"
        )
    if any(index < 0 for index in state_indices):
        raise ValueError(
            f"ssm_state_indices must be non-negative, got {state_indices}"
        )
    if len(set(state_indices)) != len(state_indices):
        raise ValueError("active candidate tokens must use distinct ssm_state_indices")
    if (
        args.delta_state_capacity is not None
        and max(state_indices) >= args.delta_state_capacity
    ):
        raise ValueError(
            "ssm_state_indices entries must be smaller than delta_state_capacity, "
            f"got capacity {args.delta_state_capacity} and indices {state_indices}"
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


def _tensor_changed(before, after) -> bool:
    return not bool((before == after).all().item())


class GatedRMSNorm(nn.Module):
    def __init__(self, head_dim: int, eps: float = 1e-6):
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
        return (normalized * F.silu(gate.float())).to(input_dtype)


class RecurrentGatedDeltaNetLayer(nn.Module):
    """One complete stateful GDN layer for decode and short MTP inputs."""

    def __init__(
        self,
        hidden_size: int,
        *,
        query_heads: int,
        value_heads: int,
        key_dim: int,
        value_dim: int,
        conv_kernel: int,
    ):
        super().__init__()
        self.query_heads = query_heads
        self.value_heads = value_heads
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.conv_kernel = conv_kernel

        query_size = query_heads * key_dim
        value_size = value_heads * value_dim
        conv_size = 2 * query_size + value_size

        self.in_proj_qkv = nn.Linear(hidden_size, conv_size, bias=False)
        self.in_proj_z = nn.Linear(hidden_size, value_size, bias=False)
        self.in_proj_b = nn.Linear(hidden_size, value_heads, bias=False)
        self.in_proj_a = nn.Linear(hidden_size, value_heads, bias=False)
        self.conv_weight = nn.Parameter(torch.empty(conv_kernel, conv_size))
        nn.init.uniform_(
            self.conv_weight,
            -1.0 / math.sqrt(conv_kernel),
            1.0 / math.sqrt(conv_kernel),
        )
        self.dt_bias = nn.Parameter(torch.ones(value_heads))
        self.A_log = nn.Parameter(torch.log(torch.rand(value_heads) * 15.99 + 0.01))
        self.norm = GatedRMSNorm(value_dim)
        self.out_proj = nn.Linear(value_size, hidden_size, bias=False)

    @torch.no_grad()
    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        conv_states: torch.Tensor,
        delta_states: torch.Tensor,
        cache_indices: list[int],
        actual_seq_lengths: torch.Tensor,
        ssm_state_indices: torch.Tensor,
        accepted_tokens_host: list[int] | None,
        num_accepted_tokens: torch.Tensor,
    ) -> torch.Tensor:
        from fla_npu.ops.ascendc import causal_conv1d, recurrent_gated_delta_rule

        batch, mtp, _ = hidden_states.shape

        mixed_qkv = self.in_proj_qkv(hidden_states)
        z = self.in_proj_z(hidden_states).reshape(
            batch, mtp, self.value_heads, self.value_dim
        )
        b = self.in_proj_b(hidden_states)
        a = self.in_proj_a(hidden_states)

        mixed_qkv = causal_conv1d(
            mixed_qkv,
            self.conv_weight,
            bias=None,
            conv_states=conv_states,
            cache_indices=cache_indices,
            num_accepted_tokens=accepted_tokens_host,
            activation_mode=1,
            pad_slot_id=-1,
            run_mode=1,
            head_num=0,
        )

        query_size = self.query_heads * self.key_dim
        value_size = self.value_heads * self.value_dim
        query, key, value = torch.split(
            mixed_qkv,
            (query_size, query_size, value_size),
            dim=-1,
        )
        query = query.reshape(-1, self.query_heads, self.key_dim)
        key = key.reshape(-1, self.query_heads, self.key_dim)
        value = value.reshape(-1, self.value_heads, self.value_dim).contiguous()

        beta = torch.sigmoid(b).to(torch.bfloat16)
        g = -self.A_log.float().exp() * F.softplus(
            a.float() + self.dt_bias.float()
        )
        query = F.normalize(query.float(), p=2, dim=-1).to(torch.bfloat16)
        key = F.normalize(key.float(), p=2, dim=-1).to(torch.bfloat16)
        beta = beta.reshape(-1, self.value_heads).contiguous()
        g = g.reshape(-1, self.value_heads).contiguous()

        recurrent_out = recurrent_gated_delta_rule(
            query,
            key,
            value,
            delta_states,
            beta=beta,
            scale=1.0 / math.sqrt(self.key_dim),
            actual_seq_lengths=actual_seq_lengths,
            ssm_state_indices=ssm_state_indices,
            num_accepted_tokens=num_accepted_tokens,
            g=g,
            gk=None,
        )
        recurrent_out = recurrent_out.reshape(
            batch, mtp, self.value_heads, self.value_dim
        )
        recurrent_out = self.norm(recurrent_out, z)
        return self.out_proj(recurrent_out.reshape(batch, mtp, -1))


def main() -> None:
    args = _parse_args()
    _validate_args(args)

    # Register the packaged custom OPP before torch_npu exposes same-named CANN ops.
    import fla_npu.ops.ascendc  # noqa: F401
    import torch_npu

    device = torch.device(f"npu:{args.device}")
    torch_npu.npu.set_device(device)
    torch.npu.set_compile_mode(jit_compile=False)
    torch.manual_seed(args.seed)
    torch.npu.manual_seed_all(args.seed)

    batch, mtp = args.batch, args.mtp
    total_tokens = batch * mtp
    state_dtype = torch.bfloat16 if args.state_dtype == "bf16" else torch.float32
    accepted_values = args.num_accepted_tokens or [mtp] * batch
    cache_indices = args.cache_indices or list(range(batch))
    state_indices = args.ssm_state_indices or list(range(total_tokens))
    conv_state_capacity = args.conv_state_capacity or (max(cache_indices) + 1)
    delta_state_capacity = args.delta_state_capacity or (max(state_indices) + 1)

    layer = RecurrentGatedDeltaNetLayer(
        args.hidden_size,
        query_heads=args.query_heads,
        value_heads=args.value_heads,
        key_dim=args.key_dim,
        value_dim=args.value_dim,
        conv_kernel=args.conv_kernel,
    ).to(device=device, dtype=torch.bfloat16)
    layer.eval()

    conv_size = (
        2 * args.query_heads * args.key_dim
        + args.value_heads * args.value_dim
    )
    conv_state_len = args.conv_kernel + mtp - 2
    conv_states = torch.zeros(
        conv_state_capacity,
        conv_state_len,
        conv_size,
        dtype=torch.bfloat16,
        device=device,
    )
    delta_states = torch.zeros(
        delta_state_capacity,
        args.value_heads,
        args.value_dim,
        args.key_dim,
        dtype=state_dtype,
        device=device,
    )

    actual_seq_lengths = torch.tensor(
        [0] + [mtp] * batch,
        dtype=torch.int32,
        device=device,
    )
    ssm_state_indices = torch.tensor(
        state_indices,
        dtype=torch.int32,
        device=device,
    )
    num_accepted_tokens = torch.tensor(
        accepted_values,
        dtype=torch.int32,
        device=device,
    )

    print(
        "config:",
        f"B={batch}",
        f"MTP={mtp}",
        f"hidden={args.hidden_size}",
        f"Hk={args.query_heads}",
        f"Hv={args.value_heads}",
        f"Dk={args.key_dim}",
        f"Dv={args.value_dim}",
        f"conv_state_capacity={conv_state_capacity}",
        f"delta_state_capacity={delta_state_capacity}",
        f"cache_indices={cache_indices}",
        f"ssm_state_indices={state_indices}",
        f"num_accepted_tokens={accepted_values}",
    )
    accepted_tokens_host = accepted_values if mtp > 1 else None
    with torch.inference_mode():
        for step in range(args.steps):
            hidden_states = torch.randn(
                batch,
                mtp,
                args.hidden_size,
                dtype=torch.bfloat16,
                device=device,
            )
            conv_before = conv_states.clone()
            delta_before = delta_states.clone()

            output = layer(
                hidden_states,
                conv_states=conv_states,
                delta_states=delta_states,
                cache_indices=cache_indices,
                actual_seq_lengths=actual_seq_lengths,
                ssm_state_indices=ssm_state_indices,
                accepted_tokens_host=accepted_tokens_host,
                num_accepted_tokens=num_accepted_tokens,
            )
            torch_npu.npu.synchronize()

            print(
                f"step {step}:",
                f"output={tuple(output.shape)} {output.dtype}",
                f"finite={bool(torch.isfinite(output.float()).all().item())}",
                f"conv_state_changed={_tensor_changed(conv_before, conv_states)}",
                f"delta_state_changed={_tensor_changed(delta_before, delta_states)}",
            )


if __name__ == "__main__":
    main()
