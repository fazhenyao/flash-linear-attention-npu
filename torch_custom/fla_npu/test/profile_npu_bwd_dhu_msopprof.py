#!/usr/bin/env python3
"""使用 msopprof 测量 ChunkGatedDeltaRuleBwdDhu，并允许从命令行传入 shape。"""

from __future__ import annotations

import argparse
import math
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


SCRIPT = Path(__file__).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    shape = parser.add_argument_group("operator shape")
    shape.add_argument("--batch", type=int, default=1)
    shape.add_argument("--tokens", type=int, default=8192)
    shape.add_argument("--key-heads", type=int, default=32)
    shape.add_argument("--value-heads", type=int, default=32)
    shape.add_argument("--key-dim", type=int, default=128)
    shape.add_argument("--value-dim", type=int, default=128)
    shape.add_argument("--chunk-size", type=int, choices=(64, 128), default=64)
    shape.add_argument("--dtype", choices=("fp16", "bf16"), default="bf16")
    shape.add_argument(
        "--sequence-count",
        type=int,
        default=0,
        help="大于 0 时启用 varlen，并将 T 尽量均分为该数量的序列",
    )
    shape.add_argument(
        "--seq-lengths",
        default="",
        help="逗号分隔的 varlen 序列长度；长度之和必须等于 --tokens，优先于 --sequence-count",
    )

    operator = parser.add_argument_group("operator options")
    operator.add_argument("--device", type=int, default=int(os.environ.get("TEST_DEVICE_ID", 0)))
    operator.add_argument("--gate-mode", choices=("none", "g", "gk", "both"), default="gk")
    operator.add_argument(
        "--g-dtype",
        choices=("same", "fp32"),
        default="same",
        help="scalar gate g 的 dtype；gk 始终为 FP32",
    )
    operator.add_argument("--use-exp2", action=argparse.BooleanOptionalAction, default=True)
    operator.add_argument("--scale", type=float, default=None)

    profiler = parser.add_argument_group("msopprof options")
    profiler.add_argument("--output", default="outputs/bwd_dhu_msopprof")
    profiler.add_argument("--launch-count", type=int, default=20)
    profiler.add_argument("--warm-up", type=int, default=5)
    profiler.add_argument("--aic-metrics", default="BasicInfo")
    profiler.add_argument(
        "--replay-mode", choices=("application", "kernel", "range"), default="application"
    )
    profiler.add_argument("--kill", choices=("on", "off"), default="on")
    profiler.add_argument(
        "--msopprof",
        default="",
        help="msopprof 可执行文件；默认从 PATH 或 ASCEND_HOME_PATH 自动查找",
    )
    profiler.add_argument("--dry-run", action="store_true", help="只打印 msopprof 命令")

    parser.add_argument("--runner", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    positive = {
        "batch": args.batch,
        "tokens": args.tokens,
        "key_heads": args.key_heads,
        "value_heads": args.value_heads,
        "key_dim": args.key_dim,
        "value_dim": args.value_dim,
        "launch_count": args.launch_count,
    }
    for name, value in positive.items():
        if value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive, got {value}")
    if args.warm_up < 0:
        raise ValueError(f"--warm-up must be non-negative, got {args.warm_up}")
    if args.value_heads % args.key_heads != 0:
        raise ValueError(
            f"--value-heads must be divisible by --key-heads, got "
            f"{args.value_heads} and {args.key_heads}"
        )
    if args.key_dim > 128:
        raise ValueError(f"bwd_dhu requires key_dim <= 128, got {args.key_dim}")
    if args.value_dim > 256:
        raise ValueError(f"bwd_dhu requires value_dim <= 256, got {args.value_dim}")


def sequence_lengths(args: argparse.Namespace) -> list[int] | None:
    if args.seq_lengths:
        lengths = [int(item.strip()) for item in args.seq_lengths.split(",") if item.strip()]
        if not lengths or any(length <= 0 for length in lengths):
            raise ValueError(f"invalid --seq-lengths: {args.seq_lengths!r}")
    elif args.sequence_count > 0:
        if args.sequence_count > args.tokens:
            raise ValueError("--sequence-count cannot exceed --tokens")
        lengths = [args.tokens // args.sequence_count] * args.sequence_count
        for index in range(args.tokens % args.sequence_count):
            lengths[index] += 1
    else:
        return None

    if args.batch != 1:
        raise ValueError("bwd_dhu varlen mode requires --batch 1")
    if sum(lengths) != args.tokens:
        raise ValueError(
            f"sum(--seq-lengths) must equal --tokens: {sum(lengths)} != {args.tokens}"
        )
    return lengths


def make_varlen_metadata(lengths: list[int], chunk_size: int) -> tuple[list[int], list[int]]:
    cu_seqlens = [0]
    chunk_indices: list[int] = []
    for sequence_index, length in enumerate(lengths):
        cu_seqlens.append(cu_seqlens[-1] + length)
        for chunk_index in range(math.ceil(length / chunk_size)):
            chunk_indices.extend((sequence_index, chunk_index + 1))
    return cu_seqlens, chunk_indices


def find_msopprof(explicit: str) -> str:
    if explicit:
        path = Path(explicit).expanduser()
        if path.is_file():
            return str(path.resolve())
        resolved = shutil.which(explicit)
        if resolved:
            return resolved
        raise FileNotFoundError(f"msopprof not found: {explicit}")

    resolved = shutil.which("msopprof")
    if resolved:
        return resolved
    ascend_home = os.environ.get("ASCEND_HOME_PATH", "").strip()
    if ascend_home:
        candidate = Path(ascend_home) / "tools" / "msopprof" / "bin" / "msopprof"
        if candidate.is_file():
            return str(candidate)
    raise FileNotFoundError(
        "msopprof is not in PATH. Source the target CANN set_env.sh or pass --msopprof explicitly."
    )


def runner_arguments(args: argparse.Namespace) -> list[str]:
    values = [
        "--runner",
        "--batch", str(args.batch),
        "--tokens", str(args.tokens),
        "--key-heads", str(args.key_heads),
        "--value-heads", str(args.value_heads),
        "--key-dim", str(args.key_dim),
        "--value-dim", str(args.value_dim),
        "--chunk-size", str(args.chunk_size),
        "--dtype", args.dtype,
        "--device", str(args.device),
        "--gate-mode", args.gate_mode,
        "--g-dtype", args.g_dtype,
        "--sequence-count", str(args.sequence_count),
        "--use-exp2" if args.use_exp2 else "--no-use-exp2",
    ]
    if args.seq_lengths:
        values.extend(("--seq-lengths", args.seq_lengths))
    if args.scale is not None:
        values.extend(("--scale", str(args.scale)))
    return values


def run_profiler(args: argparse.Namespace) -> None:
    msopprof = find_msopprof(args.msopprof)
    output = Path(args.output).expanduser()
    if not output.is_absolute():
        output = (Path.cwd() / output).resolve()

    # msopprof rejects an application located in a group/other-writable directory. Repository test
    # directories are commonly mode 0775, so execute an identical private copy of this runner.
    repo_root = SCRIPT.parents[3]
    with tempfile.TemporaryDirectory(prefix=".msopprof_bwd_dhu_", dir=repo_root) as temp_dir:
        private_dir = Path(temp_dir)
        private_dir.chmod(0o700)
        private_script = private_dir / SCRIPT.name
        shutil.copy2(SCRIPT, private_script)
        private_script.chmod(0o700)

        application_parts = [sys.executable, str(private_script), *runner_arguments(args)]
        application = " ".join(shlex.quote(part) for part in application_parts)
        command = [
            msopprof,
            f"--application={application}",
            f"--output={output}",
            f"--aic-metrics={args.aic_metrics}",
            f"--launch-count={args.launch_count}",
            f"--warm-up={args.warm_up}",
            f"--replay-mode={args.replay_mode}",
            f"--kill={args.kill}",
        ]
        print("[RUNNER] " + str(private_script), flush=True)
        print("[MSOPPROF] " + " ".join(shlex.quote(part) for part in command), flush=True)
        if args.dry_run:
            return
        subprocess.run(command, cwd=private_dir, env=os.environ.copy(), check=True)


def run_operator(args: argparse.Namespace) -> None:
    import torch
    from fla_npu.ops import ascendc as ascendc_ops

    lengths = sequence_lengths(args)
    cu_seqlens = None
    chunk_indices = None
    if lengths is not None:
        cu_seqlens, chunk_indices = make_varlen_metadata(lengths, args.chunk_size)

    torch.npu.config.allow_internal_format = False
    torch.npu.set_compile_mode(jit_compile=False)
    torch.npu.set_device(args.device)
    device = f"npu:{args.device}"
    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16

    q = torch.empty(
        (args.batch, args.key_heads, args.tokens, args.key_dim), dtype=dtype, device=device
    )
    k = torch.empty_like(q)
    w = torch.empty(
        (args.batch, args.value_heads, args.tokens, args.key_dim), dtype=dtype, device=device
    )
    d_o = torch.empty(
        (args.batch, args.value_heads, args.tokens, args.value_dim), dtype=dtype, device=device
    )
    dv = torch.empty_like(d_o)

    g = None
    if args.gate_mode in ("g", "both"):
        g_dtype = dtype if args.g_dtype == "same" else torch.float32
        g = torch.zeros(
            (args.batch, args.value_heads, args.tokens), dtype=g_dtype, device=device
        )
    gk = None
    if args.gate_mode in ("gk", "both"):
        gk = torch.zeros(
            (args.batch, args.value_heads, args.tokens, args.key_dim),
            dtype=torch.float32,
            device=device,
        )

    scale = args.scale if args.scale is not None else 1.0 / math.sqrt(args.key_dim)
    chunk_count = (
        len(chunk_indices) // 2
        if chunk_indices is not None
        else math.ceil(args.tokens / args.chunk_size)
    )
    tiling_key = 3 if g is None else (2 if g.dtype == torch.float32 else 1)
    print(
        "[CASE] "
        f"B={args.batch} T={args.tokens} Hk/Hv={args.key_heads}/{args.value_heads} "
        f"K/V={args.key_dim}/{args.value_dim} chunk={args.chunk_size} "
        f"dtype={args.dtype} gate={args.gate_mode} g_dtype={args.g_dtype} "
        f"use_exp2={args.use_exp2} varlen={lengths is not None} "
        f"sequences={len(lengths) if lengths else 0} chunks={chunk_count} tiling_key={tiling_key}",
        flush=True,
    )

    outputs = ascendc_ops.npu_chunk_gated_delta_rule_bwd_dhu(
        q,
        k,
        w,
        d_o,
        dv,
        scale=scale,
        chunk_size=args.chunk_size,
        g=g,
        gK=gk,
        h0=None,
        dht=None,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        use_exp2=args.use_exp2,
        transpose_state_layout=False,
    )
    torch.npu.synchronize()
    print(f"[DONE] output_shapes={[tuple(item.shape) if item is not None else None for item in outputs]}")


def main() -> None:
    args = parse_args()
    validate_args(args)
    sequence_lengths(args)
    if args.runner:
        run_operator(args)
    else:
        run_profiler(args)


if __name__ == "__main__":
    main()
