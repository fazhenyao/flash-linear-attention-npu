#!/usr/bin/env python3
"""使用 123.xlsx shape 验证 bwd_dhu 的 FP64/NPU-aligned 双标杆精度。"""
from __future__ import annotations

import argparse
import importlib.util
import math
import os
from dataclasses import dataclass
from pathlib import Path

import openpyxl
import torch

from fla_npu.ops import ascendc as ascendc_ops


_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parents[2]
_GOLDEN_PATH = _SCRIPT_DIR / "test_bwd_dhu.py"


def load_golden_module():
    spec = importlib.util.spec_from_file_location("test_bwd_dhu_dual_golden", _GOLDEN_PATH)
    golden = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(golden)
    return golden


@dataclass(frozen=True)
class ProfileShape:
    batch: int
    tokens: int
    key_heads: int
    value_heads: int
    key_dim: int
    value_dim: int
    sequence_count: int
    dtype: torch.dtype


def _parse_shape(text: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in text.split(","))


def load_profile_shape(xlsx_path: Path) -> ProfileShape:
    workbook = openpyxl.load_workbook(xlsx_path, read_only=True, data_only=True)
    sheet = workbook.active
    rows = sheet.iter_rows(values_only=True)
    header = next(rows)
    record = next(rows)
    columns = dict(zip(header, record))

    shapes = [_parse_shape(item) for item in str(columns["Input Shapes"]).split(";")]
    dtypes = str(columns["Input Data Types"]).split(";")
    if len(shapes) < 5 or len(shapes[0]) != 4:
        raise ValueError(f"unexpected Input Shapes in {xlsx_path}: {columns['Input Shapes']}")

    # profiling 表来自上游 kernel，主输入布局为 [B,T,H,K/V]；NPU 测试侧转换为 [B,H,T,K/V]。
    q_shape, k_shape, w_shape, do_shape, state_shape = shapes[:5]
    if q_shape != k_shape or q_shape != w_shape:
        raise ValueError(f"q/k/w shape mismatch in profile: {q_shape}, {k_shape}, {w_shape}")
    if q_shape[:3] != do_shape[:3]:
        raise ValueError(f"q/dO B,T,H mismatch in profile: {q_shape}, {do_shape}")

    dtype = torch.bfloat16 if dtypes[0] == "DT_BF16" else torch.float16
    return ProfileShape(
        batch=q_shape[0],
        tokens=q_shape[1],
        key_heads=q_shape[2],
        value_heads=do_shape[2],
        key_dim=q_shape[3],
        value_dim=do_shape[3],
        sequence_count=state_shape[0],
        dtype=dtype,
    )


def make_cu_seqlens(tokens: int, sequence_count: int) -> list[int]:
    lengths = [tokens // sequence_count] * sequence_count
    for index in range(tokens % sequence_count):
        lengths[index] += 1
    cu_seqlens = [0]
    for length in lengths:
        cu_seqlens.append(cu_seqlens[-1] + length)
    return cu_seqlens


def create_gk(batch: int, heads: int, tokens: int, key_dim: int) -> torch.Tensor:
    # 沿时间和 K 维平滑变化且保持负值，避免长递推中指数放大。
    time_gate = torch.linspace(-0.03, -0.002, tokens, dtype=torch.float32)
    key_gate = torch.linspace(-0.004, 0.0, key_dim, dtype=torch.float32)
    return (time_gate[:, None] + key_gate[None, :]).view(1, 1, tokens, key_dim).expand(
        batch, heads, tokens, key_dim
    ).contiguous()


def dual_check(name: str, actual: torch.Tensor, fp64_ref: torch.Tensor, npu_ref: torch.Tensor) -> bool:
    import ct

    result = ct.dual(
        actual.detach().cpu().float(),
        fp64_ref.detach().cpu().float(),
        npu_ref.detach().cpu().float(),
        level="L1",
    )
    success = bool(result.get("success"))
    print(
        f"[{name}] {'PASS' if success else 'FAIL'} "
        f"checks={result.get('checks', {})} ratios={result.get('ratios', {})}",
        flush=True,
    )
    return success


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xlsx", type=Path, default=_REPO_ROOT / "123.xlsx")
    parser.add_argument("--device", type=int, default=int(os.environ.get("TEST_DEVICE_ID", 0)))
    parser.add_argument("--chunk-size", type=int, choices=(64, 128), default=64)
    parser.add_argument("--gate-mode", choices=("none", "g", "gk", "both"), default="gk")
    parser.add_argument("--use-exp2", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fixed", action="store_true", help="忽略 state shape 的序列数，按定长模式验证")
    parser.add_argument("--dry-run", action="store_true", help="仅解析并打印 Excel shape，不创建 NPU tensor")
    args = parser.parse_args()

    shape = load_profile_shape(args.xlsx)
    if shape.value_heads % shape.key_heads != 0:
        raise ValueError(f"Hv must be divisible by Hk, got Hk={shape.key_heads}, Hv={shape.value_heads}")

    print(f"[XLSX] {args.xlsx}: {shape}", flush=True)
    if args.dry_run:
        return

    golden = load_golden_module()

    torch.npu.config.allow_internal_format = False
    torch.npu.set_compile_mode(jit_compile=False)
    torch.npu.set_device(args.device)
    torch.manual_seed(args.seed)

    q, k, w, d_o, dv, generated_g = golden.create_bwd_dhu_random_inputs(
        shape.batch,
        shape.key_heads,
        shape.value_heads,
        shape.tokens,
        shape.key_dim,
        shape.value_dim,
        shape.dtype,
        torch.float32,
    )
    g = generated_g if args.gate_mode in ("g", "both") else None
    gk = create_gk(
        shape.batch, shape.value_heads, shape.tokens, shape.key_dim
    ) if args.gate_mode in ("gk", "both") else None
    if args.gate_mode == "gk":
        assert g is None
        assert gk is not None and gk.dtype == torch.float32

    cu_seqlens = None
    chunk_indices = None
    if not args.fixed:
        if shape.batch != 1:
            raise ValueError("NPU varlen mode requires B=1")
        cu_seqlens = make_cu_seqlens(shape.tokens, shape.sequence_count)
        chunk_indices = golden.prepare_chunk_indices(cu_seqlens, args.chunk_size)

    scale = golden.scale_for_compute_dtype(
        golden.effective_scale(1.0 / math.sqrt(shape.key_dim), shape.key_dim), shape.dtype
    )
    print(
        "[CASE] "
        f"B={shape.batch} T={shape.tokens} Hk/Hv={shape.key_heads}/{shape.value_heads} "
        f"K/V={shape.key_dim}/{shape.value_dim} chunk={args.chunk_size} "
        f"varlen={cu_seqlens is not None} seqs={shape.sequence_count} dtype={shape.dtype} "
        f"gate={args.gate_mode} use_exp2={args.use_exp2}",
        flush=True,
    )
    if cu_seqlens is not None:
        print(f"cu_seqlens={cu_seqlens}, chunk_count={len(chunk_indices) // 2}", flush=True)

    common = dict(
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        g=g,
        gk=gk,
        scale=scale,
        chunk_size=args.chunk_size,
        use_exp2=args.use_exp2,
    )
    dh_fp64, _, dv2_fp64 = golden.chunk_gated_delta_rule_bwd_dhu_cpu(
        q, k, w, d_o, dv, golden_mode="fp64", **common
    )
    dh_npu_ref, _, dv2_npu_ref = golden.chunk_gated_delta_rule_bwd_dhu_cpu(
        q, k, w, d_o, dv, golden_mode="npu", **common
    )

    dh_npu, _, dv2_npu = ascendc_ops.npu_chunk_gated_delta_rule_bwd_dhu(
        q.npu(), k.npu(), w.npu(), d_o.npu(), dv.npu(),
        scale=scale,
        chunk_size=args.chunk_size,
        g=g.npu() if g is not None else None,
        gK=gk.npu() if gk is not None else None,
        h0=None,
        dht=None,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        use_exp2=args.use_exp2,
        transpose_state_layout=False,
    )

    dh_ok = dual_check("dh", dh_npu, dh_fp64, dh_npu_ref)
    dv2_ok = dual_check("dv2", dv2_npu, dv2_fp64, dv2_npu_ref)
    if not (dh_ok and dv2_ok):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
