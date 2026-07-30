# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Tianjin University, Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import torch
import os
import math
# import custom_ops
from fla_npu.ops import ascendc as ascendc_ops


os.environ['TBE_PARALLEL_COMPILE_ENABLE'] = '0'
os.environ['PARALLEL_COMPILE'] = '0'

torch.npu.config.allow_internal_format = False
torch.npu.set_compile_mode(jit_compile=False)
torch.npu.set_device(int(os.environ.get("TEST_DEVICE_ID", 0)))


def _reference(q, k, w, do, dv, *, g=None, gk=None, h0=None, dht=None, scale=1.0, chunk_size=64):
    B, Hk, T, K = q.shape
    Hv, V = do.shape[1], do.shape[-1]
    NT = (T + chunk_size - 1) // chunk_size
    hq = torch.arange(Hv) // (Hv // Hk)
    b_dh = dht.float().clone() if dht is not None else torch.zeros(B, Hv, K, V)
    dh = torch.empty(B, Hv, NT, K, V)
    dv2 = torch.empty_like(dv)
    for chunk_idx in range(NT - 1, -1, -1):
        start = chunk_idx * chunk_size
        end = min(start + chunk_size, T)
        last = end - 1
        dh[:, :, chunk_idx] = b_dh
        q_chunk = q[:, :, start:end].index_select(1, hq).float()
        k_chunk = k[:, :, start:end].index_select(1, hq).float()
        w_chunk = w[:, :, start:end].float()
        do_chunk = do[:, :, start:end].float()
        b_dv = torch.matmul(k_chunk, b_dh)
        if g is not None:
            g_chunk = g[:, :, start:end].float()
            g_last = g[:, :, last].float()
            b_dv *= torch.exp2(g_last[:, :, None, None] - g_chunk[:, :, :, None])
        b_dv += dv[:, :, start:end].float()
        dv2[:, :, start:end] = b_dv.to(dv.dtype)
        if g is not None:
            b_dh *= torch.exp2(g_last[:, :, None, None])
            q_chunk *= torch.exp2(g_chunk[:, :, :, None])
        if gk is not None:
            b_dh *= torch.exp2(gk[:, :, last].float())[:, :, :, None]
        b_dh += torch.matmul(q_chunk.transpose(-1, -2), do_chunk) * scale
        b_dh -= torch.matmul(w_chunk.transpose(-1, -2), b_dv)
    return dh, b_dh if h0 is not None else None, dv2


def test_gate_modes():
    torch.manual_seed(1)
    B, Hk, Hv, T, K, V = 1, 1, 2, 128, 128, 64
    dtype = torch.bfloat16
    scale = 1.0 / math.sqrt(K)
    q = torch.randn(B, Hk, T, K, dtype=dtype) * 0.1
    k = torch.randn_like(q) * 0.1
    w = torch.randn(B, Hv, T, K, dtype=dtype) * 0.1
    do = torch.randn(B, Hv, T, V, dtype=dtype) * 0.1
    dv = torch.randn_like(do) * 0.1
    scalar_g = torch.cumsum(-torch.rand(B, Hv, T, dtype=torch.float32) * 0.02, dim=-1)
    key_g = torch.cumsum(-torch.rand(B, Hv, T, K, dtype=torch.float32) * 0.02, dim=-2)

    cases = [
        ("gdn", q, k, scalar_g, None),
        ("kda", q * 0.75, k * 0.5, None, key_g),
        ("ungated", q, k, None, None),
    ]
    for name, q_in, k_in, g, gk in cases:
        dh_ref, _, dv2_ref = _reference(q_in, k_in, w, do, dv, g=g, gk=gk, scale=scale)
        dh, _, dv2 = ascendc_ops.npu_chunk_gated_delta_rule_bwd_dhu(
            q_in.npu(), k_in.npu(), w.npu(), do.npu(), dv.npu(), scale, 64,
            g=None if g is None else g.npu(), gK=None if gk is None else gk.npu(),
        )
        torch.npu.synchronize()
        torch.testing.assert_close(dh.cpu().float(), dh_ref, rtol=0.03, atol=0.03, msg=lambda msg: f"{name} dh: {msg}")
        torch.testing.assert_close(dv2.cpu().float(), dv2_ref.float(), rtol=0.03, atol=0.03,
                                   msg=lambda msg: f"{name} dv2: {msg}")


def prepare_chunk_indices_original(cu_seqlens, chunk_size=64):
    """完全保持你原始代码的chunk_indices生成逻辑"""
    chunk_indices = []

    # 遍历每个序列
    for seq_idx in range(len(cu_seqlens) - 1):
        seq_len = cu_seqlens[seq_idx + 1] - cu_seqlens[seq_idx]
        chunk_num = (seq_len + chunk_size - 1) // chunk_size  # 向上取整

        # 添加 [seq_idx, chunk_idx] 对
        for chunk_idx in range(chunk_num):
            chunk_indices.append(seq_idx)
            chunk_indices.append(chunk_idx + 1)  # chunk_idx从1开始

    return chunk_indices


def run_case(
    case_name,
    B,
    T,
    H,
    K,
    V,
    chunk_size,
    dtype,
    is_varlen=False,
    cu_seqlens=None
):
    print(f"\n===== {case_name} =====")
    print(f"B={B}, T={T}, H={H}, K={K}, V={V}, chunk_size={chunk_size}, dtype={dtype}, is_varlen={is_varlen}")

    torch.manual_seed(0)

    q = torch.randn(B, H, T, K, dtype=dtype)
    k = torch.randn(B, H, T, K, dtype=dtype)
    w = torch.randn(B, H, T, K, dtype=dtype)
    do = torch.randn(B, H, T, V, dtype=dtype)
    dv = torch.randn(B, H, T, V, dtype=dtype)
    g = torch.randn(B, H, T, dtype=dtype)
    scale = k.shape[-1] ** -0.5

    chunk_indices = None
    if is_varlen:
        chunk_indices = prepare_chunk_indices_original(cu_seqlens, chunk_size)
        print("cu_seqlens =", cu_seqlens)
        print("chunk_indices len =", len(chunk_indices))

    print("before custom op")
    dh, dh0, dv2 = ascendc_ops.npu_chunk_gated_delta_rule_bwd_dhu(
        q.npu(),
        k.npu(),
        w.npu(),
        do.npu(),
        dv.npu(),
        scale=scale,
        chunk_size=chunk_size,
        g=g.npu(),
        gK=None,
        h0=None,
        dht=None,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices
    )
    print("after custom op")
    print("dh shape =", dh.shape)
    print("dv2 shape =", dv2.shape)
    if dh0 is None:
        print("dh0 = None")
    else:
        print("dh0 shape =", dh0.shape)


if __name__ == "__main__":

    # Case 1: 定长
    run_case(
        case_name="Case 1: fixed length",
        B=1,
        T=32768,
        H=32,
        K=128,
        V=128,
        chunk_size=64,
        dtype=torch.float16,
        is_varlen=False,
        cu_seqlens=None
    )
    print("Case 1 finished")

    # Case 2: 完全保持你原始代码不动的变长case
    run_case(
        case_name="Case 2: varlen original",
        B=1,
        T=32768,
        H=32,
        K=128,
        V=128,
        chunk_size=64,
        dtype=torch.float16,
        is_varlen=True,
        cu_seqlens=[0, 1024, 2048, 4096, 8192, 10240, 20480, 32768]
    )
    print("Case 2 finished")

    # Case 3: 额外补一组 L2 风格变长 case
    run_case(
        case_name="Case 3: varlen L2 style",
        B=1,
        T=65536,
        H=16,
        K=128,
        V=128,
        chunk_size=64,
        dtype=torch.float16,
        is_varlen=True,
        cu_seqlens=[0, 2048, 4096, 6144, 8192, 12288, 16384, 20480, 24576, 28672, 32768, 36864, 40960, 45056, 49152, 53248, 57344, 61440, 65536]
    )
    print("Case 3 finished")
