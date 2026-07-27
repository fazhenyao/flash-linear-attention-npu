from __future__ import annotations

import math
import os

import torch

from fla_npu.ops.ascendc import npu_chunk_gated_delta_rule_bwd_dhu


def _reference(q, k, w, do, dv, scale, chunk_size, *, g=None, gk=None, h0=None, dht=None, cu=None):
    bsz, hk, tokens, kdim = q.shape
    hv, vdim = dv.shape[1], dv.shape[-1]
    ratio = hv // hk
    sequences = [(b, 0, tokens) for b in range(bsz)] if cu is None else [
        (0, cu[i], cu[i + 1]) for i in range(len(cu) - 1)
    ]
    total_chunks = sum(math.ceil((eos - bos) / chunk_size) for _, bos, eos in sequences)
    dh = torch.empty((bsz, total_chunks, hv, kdim, vdim), dtype=q.dtype)
    dv2 = torch.empty_like(dv)
    dh0 = torch.empty((len(sequences), hv, kdim, vdim), dtype=torch.float32) if h0 is not None else None

    chunk_base = 0
    for seq, (batch, bos, eos) in enumerate(sequences):
        chunks = math.ceil((eos - bos) / chunk_size)
        for h in range(hv):
            qh = h // ratio
            state = torch.zeros((kdim, vdim), dtype=torch.float32)
            if dht is not None:
                state.copy_(dht[seq, h])
            for chunk in range(chunks - 1, -1, -1):
                start = bos + chunk * chunk_size
                end = min(start + chunk_size, eos)
                dh[batch, chunk_base + chunk, h] = state.to(q.dtype)

                k_block = k[batch, qh, start:end].float()
                q_block = q[batch, qh, start:end].float()
                w_block = w[batch, h, start:end].float()
                do_block = do[batch, h, start:end].float()
                raw = k_block.to(q.dtype).float() @ state.to(q.dtype).float()

                if g is not None:
                    gate = torch.exp2(g[batch, h, start:end].float())
                    last_gate = gate[-1]
                    raw *= (last_gate / gate).unsqueeze(1)
                else:
                    gate = None
                    last_gate = torch.tensor(1.0)

                cur_dv2 = dv[batch, h, start:end].float() + raw
                dv2[batch, h, start:end] = cur_dv2.to(q.dtype)
                state *= last_gate
                if gk is not None:
                    state *= torch.exp2(gk[batch, h, end - 1].float()).unsqueeze(1)

                q_gated = q_block if gate is None else q_block * gate.unsqueeze(1)
                plus = q_gated.to(q.dtype).float().T @ do_block.to(q.dtype).float()
                minus = w_block.to(q.dtype).float().T @ cur_dv2.to(q.dtype).float()
                state += plus * scale - minus
            if dh0 is not None:
                dh0[seq, h] = state
        chunk_base += chunks
    return dh, dh0, dv2


def _assert_close(name, actual, expected, atol=4e-3, rtol=4e-2):
    actual = actual.cpu().float()
    expected = expected.float()
    diff = (actual - expected).abs()
    print(f"  {name}: max_abs={diff.max().item():.6e} mean_abs={diff.mean().item():.6e}")
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


def _chunk_indices(cu, chunk_size):
    result = []
    for seq in range(len(cu) - 1):
        for chunk in range(math.ceil((cu[seq + 1] - cu[seq]) / chunk_size)):
            result.extend((seq, chunk))
    return result


def _run_case(name, *, dtype, hv, tokens, vdim, chunk_size, use_g, use_gk, state, cu=None, gate_dtype=torch.float32):
    print(f"CASE {name}")
    torch.manual_seed(7)
    bsz, hk, kdim = 1, 1 if hv > 1 else hv, 128
    scale = kdim**-0.5
    q = (torch.randn(bsz, hk, tokens, kdim) * 0.004).to(dtype)
    k = (torch.randn_like(q.float()) * 0.004).to(dtype)
    w = (torch.randn(bsz, hv, tokens, kdim) * 0.004).to(dtype)
    do = (torch.randn(bsz, hv, tokens, vdim) * 0.004).to(dtype)
    dv = (torch.randn(bsz, hv, tokens, vdim) * 0.006).to(dtype)
    g = None
    gk = None
    if use_g:
        g = torch.linspace(-1e-4, -1e-2, tokens).view(1, 1, -1).expand(bsz, hv, -1).contiguous().to(gate_dtype)
    if use_gk:
        gk = (torch.randn(bsz, hv, tokens, kdim) * 2e-4).cumsum(dim=2).to(gate_dtype)
    state_batch = len(cu) - 1 if cu is not None else bsz
    h0 = torch.empty((state_batch, hv, kdim, vdim), dtype=dtype) if state else None
    dht = torch.randn(state_batch, hv, kdim, vdim) * 2e-3 if state else None
    indices = _chunk_indices(cu, chunk_size) if cu is not None else None

    ref_dh, ref_dh0, ref_dv2 = _reference(
        q, k, w, do, dv, scale, chunk_size, g=g, gk=gk, h0=h0, dht=dht, cu=cu,
    )
    dh, dh0, dv2 = npu_chunk_gated_delta_rule_bwd_dhu(
        q.npu(), k.npu(), w.npu(), do.npu(), dv.npu(), scale, chunk_size,
        g=None if g is None else g.npu(),
        gK=None if gk is None else gk.npu(),
        h0=None if h0 is None else h0.npu(),
        dht=None if dht is None else dht.npu(),
        cu_seqlens=cu,
        chunk_indices=indices,
    )
    torch.npu.synchronize()

    assert tuple(dh.shape) == tuple(ref_dh.shape)
    assert (dh0 is None) == (ref_dh0 is None)
    _assert_close("dh", dh, ref_dh)
    if dh0 is not None:
        _assert_close("dh0", dh0, ref_dh0)
    _assert_close("dv2", dv2, ref_dv2)


def main():
    torch.npu.set_device(int(os.environ.get("TEST_DEVICE_ID", "0")))
    cases = [
        ("bf16_g_gk_state_gva", dict(dtype=torch.bfloat16, hv=2, tokens=128, vdim=128, chunk_size=64,
                                      use_g=True, use_gk=True, state=True)),
        ("bf16_g_only", dict(dtype=torch.bfloat16, hv=1, tokens=128, vdim=128, chunk_size=64,
                              use_g=True, use_gk=False, state=False)),
        ("bf16_gk_only", dict(dtype=torch.bfloat16, hv=1, tokens=128, vdim=128, chunk_size=64,
                               use_g=False, use_gk=True, state=True)),
        ("bf16_gk_only_hv2", dict(dtype=torch.bfloat16, hv=2, tokens=128, vdim=128, chunk_size=64,
                                   use_g=False, use_gk=True, state=True)),
        ("bf16_g_gk_hv1", dict(dtype=torch.bfloat16, hv=1, tokens=128, vdim=128, chunk_size=64,
                                use_g=True, use_gk=True, state=True)),
        ("bf16_gk_only_bf16_gate", dict(dtype=torch.bfloat16, hv=1, tokens=128, vdim=128, chunk_size=64,
                                         use_g=False, use_gk=True, state=True, gate_dtype=torch.bfloat16)),
        ("fp16_no_gate", dict(dtype=torch.float16, hv=1, tokens=128, vdim=128, chunk_size=64,
                               use_g=False, use_gk=False, state=True)),
        ("bf16_v256", dict(dtype=torch.bfloat16, hv=1, tokens=128, vdim=256, chunk_size=64,
                            use_g=True, use_gk=False, state=False)),
        ("fp16_chunk128", dict(dtype=torch.float16, hv=1, tokens=256, vdim=128, chunk_size=128,
                                use_g=True, use_gk=False, state=False)),
        ("bf16_varlen", dict(dtype=torch.bfloat16, hv=2, tokens=192, vdim=128, chunk_size=64,
                              use_g=True, use_gk=True, state=True, cu=[0, 80, 192])),
    ]
    selected = os.environ.get("DHU_TEST_CASE")
    for name, kwargs in cases:
        if selected is None or selected == name:
            _run_case(name, **kwargs)


if __name__ == "__main__":
    main()
