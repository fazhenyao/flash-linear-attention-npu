# Copyright (c) 2026 Tianjin University, Ltd.

"""Train-time chunk KDA and complete KDA mixer examples on Ascend NPU."""

from __future__ import annotations

import argparse
import importlib.metadata
import math
import os

# Register fla_npu's packaged OPP before torch_npu initializes the NPU runtime.
os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

import torch
import torch.nn as nn
import torch.nn.functional as F


TRITON_ASCEND_VERSION = "3.2.1"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run chunk KDA forward/backward or a complete train-time KDA mixer."
    )
    parser.add_argument("--case-name", default="")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--query-heads", type=int, default=2)
    parser.add_argument("--value-heads", type=int, default=2)
    parser.add_argument("--key-dim", type=int, default=128)
    parser.add_argument("--value-dim", type=int, default=128)
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--scale", type=float, default=None)
    parser.add_argument("--chunk-size", type=int, choices=(64,), default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--varlen", action="store_true")
    parser.add_argument(
        "--cu-seqlens",
        default="",
        help="Comma-separated variable-length offsets, for example 0,64,128.",
    )
    parser.add_argument(
        "--mean-len",
        type=int,
        default=64,
        help="Approximate sequence length when --varlen is used without explicit offsets.",
    )
    parser.add_argument(
        "--initial-state",
        choices=("none", "zeros", "random"),
        default="none",
    )
    parser.add_argument("--output-final-state", action="store_true")
    parser.add_argument("--safe-gate", action="store_true")
    parser.add_argument("--lower-bound", type=float, default=-5.0)
    parser.add_argument("--allow-neg-eigval", action="store_true")
    parser.add_argument("--disable-recompute", action="store_true")
    parser.add_argument(
        "--forward-only",
        action="store_true",
        help="Run the forward path without executing backward.",
    )
    parser.add_argument("--demo-model", action="store_true")
    parser.add_argument(
        "--use-short-conv",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--conv-kernel", type=int, choices=(2, 3, 4), default=4)
    parser.add_argument("--conv-bias", action="store_true")
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    positive = {
        "batch": args.batch,
        "tokens": args.tokens,
        "hidden_size": args.hidden_size,
        "query_heads": args.query_heads,
        "value_heads": args.value_heads,
        "key_dim": args.key_dim,
        "value_dim": args.value_dim,
        "mean_len": args.mean_len,
    }
    for name, value in positive.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")
    if args.query_heads != args.value_heads:
        raise ValueError(
            "the pinned train-time ChunkKDAFunction requires query_heads == "
            f"value_heads, got {args.query_heads} and {args.value_heads}"
        )
    if args.key_dim > 256 or args.key_dim % 16 != 0:
        raise ValueError("key_dim must be a multiple of 16 and no larger than 256")
    if args.value_dim > 256 or args.value_dim % 16 != 0:
        raise ValueError("value_dim must be a multiple of 16 and no larger than 256")
    if args.varlen and args.batch != 1:
        raise ValueError("variable-length input requires batch=1")
    if args.cu_seqlens and not args.varlen:
        raise ValueError("--cu-seqlens requires --varlen")
    if args.safe_gate and not -5.0 <= args.lower_bound < 0.0:
        raise ValueError("lower_bound must be in [-5, 0) when safe_gate is enabled")


def _parse_cu_seqlens(value: str, total_tokens: int) -> list[int] | None:
    if not value.strip():
        return None
    offsets = [int(item.strip()) for item in value.split(",") if item.strip()]
    if len(offsets) < 2:
        raise ValueError("cu_seqlens must contain at least two offsets")
    if offsets[0] != 0 or offsets[-1] != total_tokens:
        raise ValueError(
            f"cu_seqlens must start at 0 and end at tokens={total_tokens}, got {offsets}"
        )
    if any(left >= right for left, right in zip(offsets, offsets[1:])):
        raise ValueError(f"cu_seqlens must be strictly increasing, got {offsets}")
    return offsets


def _build_cu_seqlens(args: argparse.Namespace) -> list[int] | None:
    offsets = _parse_cu_seqlens(args.cu_seqlens, args.tokens)
    if offsets is not None or not args.varlen:
        return offsets
    offsets = list(range(0, args.tokens, args.mean_len))
    if not offsets or offsets[0] != 0:
        offsets.insert(0, 0)
    if offsets[-1] != args.tokens:
        offsets.append(args.tokens)
    return offsets


def _load_chunk_kda():
    try:
        version = importlib.metadata.version("triton-ascend")
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError(
            f"flash_kda.py requires triton-ascend=={TRITON_ASCEND_VERSION}"
        ) from exc
    if version != TRITON_ASCEND_VERSION:
        raise RuntimeError(
            f"flash_kda.py requires triton-ascend=={TRITON_ASCEND_VERSION}, got {version}"
        )

    try:
        from fla_npu.adapters import install_triton_ascend_kda_adapter

        install_triton_ascend_kda_adapter()
        from triton_ascend_kernels.attention.fla.kda import chunk_kda
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "flash_kda.py requires Ascend/triton-ascend-kernels commit "
            "4cd4b506d4153ac18ac1ca8f4c770eac9fd3fcc8 on PYTHONPATH"
        ) from exc
    return chunk_kda


class AscendCCausalConv1dFunction(torch.autograd.Function):
    """Training-mode causal conv with explicit Ascend C backward binding."""

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        head_num: int,
        cu_seqlens: torch.Tensor | None,
    ) -> torch.Tensor:
        from fla_npu.ops.ascendc import causal_conv1d

        op_weight = weight.transpose(0, 1).contiguous()
        width, feature_dim = op_weight.shape
        cu_list = None if cu_seqlens is None else cu_seqlens.detach().cpu().tolist()
        is_varlen = cu_list is not None
        op_x = x.reshape(-1, feature_dim).contiguous() if is_varlen else x.contiguous()
        sequence_count = len(cu_list) - 1 if cu_list is not None else int(x.shape[0])
        conv_states = torch.zeros(
            sequence_count,
            width - 1,
            feature_dim,
            dtype=x.dtype,
            device=x.device,
        )
        preactivation = causal_conv1d(
            op_x,
            op_weight,
            bias=bias,
            conv_states=conv_states,
            query_start_loc=cu_list,
            activation_mode=0,
            pad_slot_id=-1,
            run_mode=0,
            head_num=head_num,
        )
        if is_varlen:
            preactivation = preactivation.unsqueeze(0)

        ctx.save_for_backward(x, op_weight, preactivation)
        ctx.has_bias = bias is not None
        ctx.is_varlen = is_varlen
        ctx.query_start_loc = cu_list
        if is_varlen:
            ctx.input_layout = "TND" if head_num == 0 else "NTD"
        else:
            ctx.input_layout = "BSH" if head_num == 0 else "BNSD"
        return F.silu(preactivation)

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        from fla_npu.ops.ascendc import causal_conv1d_bwd

        x, op_weight, preactivation = ctx.saved_tensors
        op_x = x.reshape(-1, x.shape[-1]).contiguous() if ctx.is_varlen else x.contiguous()
        op_grad = grad.squeeze(0).contiguous() if ctx.is_varlen else grad.contiguous()
        op_y = (
            preactivation.squeeze(0).contiguous()
            if ctx.is_varlen
            else preactivation.contiguous()
        )
        dx, dw, db, _ = causal_conv1d_bwd(
            x=op_x,
            y=op_y,
            weight=op_weight,
            dy=op_grad,
            initial_state=None,
            dht=None,
            query_start_loc=ctx.query_start_loc,
            activation=1,
            input_layout=ctx.input_layout,
        )
        return (
            dx.reshape_as(x),
            dw.transpose(0, 1).contiguous(),
            db if ctx.has_bias else None,
            None,
            None,
        )


def causal_conv1d_train(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    *,
    head_num: int,
    cu_seqlens: torch.Tensor | None,
) -> torch.Tensor:
    return AscendCCausalConv1dFunction.apply(
        x,
        weight,
        bias,
        head_num,
        cu_seqlens,
    )


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


class DemoKimiDeltaAttention(nn.Module):
    """Complete train-time KDA mixer without the surrounding Transformer block."""

    def __init__(
        self,
        hidden_size: int,
        *,
        heads: int,
        key_dim: int,
        value_dim: int,
        use_short_conv: bool,
        conv_kernel: int,
        conv_bias: bool,
        safe_gate: bool,
        lower_bound: float,
        allow_neg_eigval: bool,
        disable_recompute: bool,
    ):
        super().__init__()
        self.heads = heads
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.use_short_conv = use_short_conv
        self.safe_gate = safe_gate
        self.lower_bound = lower_bound
        self.allow_neg_eigval = allow_neg_eigval
        self.disable_recompute = disable_recompute

        key_size = heads * key_dim
        value_size = heads * value_dim
        gate_size = heads * key_dim
        conv_size = 2 * key_size + value_size
        self.key_size = key_size
        self.value_size = value_size

        self.in_proj_qkv = nn.Linear(hidden_size, conv_size, bias=False)

        if use_short_conv:
            self.conv_weight = nn.Parameter(torch.empty(conv_size, conv_kernel))
            bound = 1.0 / math.sqrt(conv_kernel)
            nn.init.uniform_(self.conv_weight, -bound, bound)
            self.conv_bias = nn.Parameter(torch.zeros(conv_size)) if conv_bias else None

        self.in_proj_a = nn.Sequential(
            nn.Linear(hidden_size, value_dim, bias=False),
            nn.Linear(value_dim, gate_size, bias=False),
        )
        self.in_proj_b = nn.Linear(hidden_size, heads, bias=False)
        self.in_proj_z = nn.Sequential(
            nn.Linear(hidden_size, value_dim, bias=False),
            nn.Linear(value_dim, value_size, bias=True),
        )

        if safe_gate:
            self.A_log = nn.Parameter(torch.zeros(heads, dtype=torch.float32))
        else:
            initial_a = torch.empty(heads, dtype=torch.float32).uniform_(1, 16)
            self.A_log = nn.Parameter(torch.log(initial_a))
        dt = torch.exp(
            torch.rand(gate_size, dtype=torch.float32)
            * (math.log(0.1) - math.log(0.001))
            + math.log(0.001)
        ).clamp(min=1e-4)
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))

        self.norm = SigmoidGatedRMSNorm(value_dim)
        self.out_proj = nn.Linear(value_size, hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        chunk_kda,
        cu_seqlens: torch.Tensor | None,
        cu_seqlens_cpu: torch.Tensor | None,
        initial_state: torch.Tensor | None,
        output_final_state: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch, tokens, _ = hidden_states.shape
        mixed_qkv = self.in_proj_qkv(hidden_states)
        z = self.in_proj_z(hidden_states).reshape(
            batch, tokens, self.heads, self.value_dim
        )
        b = self.in_proj_b(hidden_states)
        a = self.in_proj_a(hidden_states)

        if self.use_short_conv:
            mixed_qkv = causal_conv1d_train(
                mixed_qkv,
                self.conv_weight,
                self.conv_bias,
                head_num=0,
                cu_seqlens=cu_seqlens,
            )
        else:
            mixed_qkv = F.silu(mixed_qkv)

        query, key, value = torch.split(
            mixed_qkv,
            (self.key_size, self.key_size, self.value_size),
            dim=-1,
        )
        query = query.reshape(batch, tokens, self.heads, self.key_dim).contiguous()
        key = key.reshape(batch, tokens, self.heads, self.key_dim).contiguous()
        value = value.reshape(batch, tokens, self.heads, self.value_dim).contiguous()

        raw_gate = a.reshape(
            batch, tokens, self.heads, self.key_dim
        ).contiguous()
        beta = torch.sigmoid(b.float())
        if self.allow_neg_eigval:
            beta = beta * 2.0
        beta = beta.to(hidden_states.dtype).contiguous()

        core_out, final_state = chunk_kda(
            q=query,
            k=key,
            v=value,
            g=raw_gate,
            beta=beta,
            A_log=self.A_log,
            dt_bias=self.dt_bias,
            scale=self.key_dim**-0.5,
            initial_state=initial_state,
            output_final_state=output_final_state,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            cu_seqlens=cu_seqlens,
            cu_seqlens_cpu=cu_seqlens_cpu,
            safe_gate=self.safe_gate,
            lower_bound=self.lower_bound,
            disable_recompute=self.disable_recompute,
            return_intermediate_states=False,
            transpose_state_layout=True,
        )
        core_out = self.norm(core_out, z)
        output = self.out_proj(core_out.reshape(batch, tokens, -1))
        return output, final_state


def _move_model_parameters(
    model: DemoKimiDeltaAttention,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    model.to(device=device)
    for name, parameter in model.named_parameters():
        if name not in {"A_log", "dt_bias"}:
            parameter.data = parameter.data.to(dtype=dtype)


def _make_initial_state(
    mode: str,
    *,
    sequence_count: int,
    heads: int,
    value_dim: int,
    key_dim: int,
    device: torch.device,
) -> torch.Tensor | None:
    if mode == "none":
        return None
    shape = (sequence_count, heads, value_dim, key_dim)
    state = (
        torch.zeros(shape, dtype=torch.float32, device=device)
        if mode == "zeros"
        else torch.randn(shape, dtype=torch.float32, device=device) * 0.02
    )
    return state.requires_grad_(True)


def _print_grad(name: str, tensor: torch.Tensor) -> None:
    grad = tensor.grad
    print(
        f"{name}.grad:",
        "None" if grad is None else f"finite={bool(torch.isfinite(grad.float()).all().item())}",
        "" if grad is None else f"norm={float(grad.float().norm().item()):.6g}",
    )


def _run_core(
    args: argparse.Namespace,
    *,
    chunk_kda,
    device: torch.device,
    dtype: torch.dtype,
    cu_seqlens: torch.Tensor | None,
    cu_seqlens_cpu: torch.Tensor | None,
    sequence_count: int,
) -> None:
    shape_k = (args.batch, args.tokens, args.query_heads, args.key_dim)
    shape_v = (args.batch, args.tokens, args.value_heads, args.value_dim)
    q = (torch.randn(shape_k, dtype=dtype, device=device) * 0.02).requires_grad_(True)
    k = (torch.randn(shape_k, dtype=dtype, device=device) * 0.02).requires_grad_(True)
    v = (torch.randn(shape_v, dtype=dtype, device=device) * 0.02).requires_grad_(True)
    raw_gate = (
        torch.randn(shape_k, dtype=dtype, device=device) * 0.02
    ).requires_grad_(True)
    raw_beta = torch.randn(
        args.batch,
        args.tokens,
        args.value_heads,
        dtype=dtype,
        device=device,
        requires_grad=True,
    )
    beta = torch.sigmoid(raw_beta.float())
    if args.allow_neg_eigval:
        beta = beta * 2.0
    beta = beta.to(dtype).contiguous()
    A_log = torch.zeros(
        args.value_heads,
        dtype=torch.float32,
        device=device,
        requires_grad=True,
    )
    dt_bias = torch.zeros(
        args.value_heads * args.key_dim,
        dtype=torch.float32,
        device=device,
        requires_grad=True,
    )
    initial_state = _make_initial_state(
        args.initial_state,
        sequence_count=sequence_count,
        heads=args.value_heads,
        value_dim=args.value_dim,
        key_dim=args.key_dim,
        device=device,
    )

    out, final_state = chunk_kda(
        q=q,
        k=k,
        v=v,
        g=raw_gate,
        beta=beta,
        A_log=A_log,
        dt_bias=dt_bias,
        scale=args.scale if args.scale is not None else args.key_dim**-0.5,
        initial_state=initial_state,
        output_final_state=args.output_final_state,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        cu_seqlens=cu_seqlens,
        cu_seqlens_cpu=cu_seqlens_cpu,
        safe_gate=args.safe_gate,
        lower_bound=args.lower_bound,
        disable_recompute=args.disable_recompute,
        return_intermediate_states=False,
        transpose_state_layout=True,
    )
    loss = out.float().square().mean()
    if not args.forward_only:
        loss.backward()
    torch.npu.synchronize()

    print("forward:", tuple(out.shape), out.dtype, f"loss={float(loss.item()):.6g}")
    print("final_state:", None if final_state is None else tuple(final_state.shape))
    if args.forward_only:
        print("backward: skipped (--forward-only)")
        return
    for name, tensor in (
        ("q", q),
        ("k", k),
        ("v", v),
        ("raw_gate", raw_gate),
        ("raw_beta", raw_beta),
        ("A_log", A_log),
        ("dt_bias", dt_bias),
    ):
        _print_grad(name, tensor)
    if initial_state is not None:
        _print_grad("initial_state", initial_state)


def _run_model(
    args: argparse.Namespace,
    *,
    chunk_kda,
    device: torch.device,
    dtype: torch.dtype,
    cu_seqlens: torch.Tensor | None,
    cu_seqlens_cpu: torch.Tensor | None,
    sequence_count: int,
) -> None:
    model = DemoKimiDeltaAttention(
        args.hidden_size,
        heads=args.query_heads,
        key_dim=args.key_dim,
        value_dim=args.value_dim,
        use_short_conv=args.use_short_conv,
        conv_kernel=args.conv_kernel,
        conv_bias=args.conv_bias,
        safe_gate=args.safe_gate,
        lower_bound=args.lower_bound,
        allow_neg_eigval=args.allow_neg_eigval,
        disable_recompute=args.disable_recompute,
    )
    _move_model_parameters(model, device=device, dtype=dtype)
    model.train()
    hidden_states = torch.randn(
        args.batch,
        args.tokens,
        args.hidden_size,
        dtype=dtype,
        device=device,
        requires_grad=True,
    )
    initial_state = _make_initial_state(
        args.initial_state,
        sequence_count=sequence_count,
        heads=args.value_heads,
        value_dim=args.value_dim,
        key_dim=args.key_dim,
        device=device,
    )
    output, final_state = model(
        hidden_states,
        chunk_kda=chunk_kda,
        cu_seqlens=cu_seqlens,
        cu_seqlens_cpu=cu_seqlens_cpu,
        initial_state=initial_state,
        output_final_state=args.output_final_state,
    )
    loss = output.float().square().mean()
    if not args.forward_only:
        loss.backward()
    torch.npu.synchronize()

    print(
        "model forward:",
        tuple(output.shape),
        output.dtype,
        f"loss={float(loss.item()):.6g}",
    )
    print("final_state:", None if final_state is None else tuple(final_state.shape))
    if args.forward_only:
        print("backward: skipped (--forward-only)")
        return
    _print_grad("hidden_states", hidden_states)
    missing = []
    nonfinite = []
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            missing.append(name)
        elif not bool(torch.isfinite(parameter.grad.float()).all().item()):
            nonfinite.append(name)
    print(
        "parameter gradients:",
        f"total={sum(1 for _ in model.parameters())}",
        f"missing={missing}",
        f"nonfinite={nonfinite}",
    )
    if missing or nonfinite:
        raise RuntimeError(
            "complete KDA mixer did not produce finite gradients for all parameters"
        )
    if initial_state is not None:
        _print_grad("initial_state", initial_state)


def main() -> None:
    args = _parse_args()
    _validate_args(args)

    import fla_npu.ops.ascendc  # noqa: F401
    import torch_npu

    device = torch.device(f"npu:{args.device}")
    torch_npu.npu.set_device(device)
    torch.npu.set_compile_mode(jit_compile=False)
    torch.manual_seed(args.seed)
    torch.npu.manual_seed_all(args.seed)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    cu_values = _build_cu_seqlens(args)
    cu_seqlens = (
        None
        if cu_values is None
        else torch.tensor(cu_values, dtype=torch.int64, device=device)
    )
    cu_seqlens_cpu = (
        None if cu_values is None else torch.tensor(cu_values, dtype=torch.int64)
    )
    sequence_count = args.batch if cu_values is None else len(cu_values) - 1
    chunk_kda = _load_chunk_kda()

    print(
        "config:",
        f"case={args.case_name or '<direct>'}",
        f"mode={'model' if args.demo_model else 'core'}",
        f"B={args.batch}",
        f"T={args.tokens}",
        f"H={args.query_heads}",
        f"K={args.key_dim}",
        f"V={args.value_dim}",
        f"dtype={args.dtype}",
        f"varlen={args.varlen}",
        f"sequences={sequence_count}",
        f"safe_gate={args.safe_gate}",
        f"disable_recompute={args.disable_recompute}",
        f"forward_only={args.forward_only}",
    )
    if args.demo_model:
        _run_model(
            args,
            chunk_kda=chunk_kda,
            device=device,
            dtype=dtype,
            cu_seqlens=cu_seqlens,
            cu_seqlens_cpu=cu_seqlens_cpu,
            sequence_count=sequence_count,
        )
    else:
        _run_core(
            args,
            chunk_kda=chunk_kda,
            device=device,
            dtype=dtype,
            cu_seqlens=cu_seqlens,
            cu_seqlens_cpu=cu_seqlens_cpu,
            sequence_count=sequence_count,
        )


if __name__ == "__main__":
    main()
