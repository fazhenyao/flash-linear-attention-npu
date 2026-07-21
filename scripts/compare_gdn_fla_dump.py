#!/usr/bin/env python3

"""Replay FLA fwd_h/bwd_dhu dumps with the Ascend C operators."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch


def _load_payload(dump_path: Path, outputs_path: Path | None) -> tuple[str, dict, dict, int]:
    payload = torch.load(dump_path, map_location='cpu', weights_only=False)
    if 'inputs' in payload and 'outputs' in payload:
        return payload['op'], payload['inputs'], payload['outputs'], int(payload.get('gate_exponent_base', 2))

    if outputs_path is None:
        raise ValueError('Legacy dumps require --outputs with the matching output .pt file.')
    outputs = torch.load(outputs_path, map_location='cpu', weights_only=False)
    op = 'bwd_dhu' if 'q' in payload else 'fwd_h'
    return op, payload, outputs, 2


def _tensor(data: dict, *names: str) -> torch.Tensor | None:
    for name in names:
        value = data.get(name)
        if isinstance(value, torch.Tensor):
            return value
    return None


def _as_int(value, default: int) -> int:
    if value is None:
        return default
    if isinstance(value, torch.Tensor):
        return int(value.item())
    return int(value)


def _as_bool(value, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, torch.Tensor):
        return bool(value.item())
    return bool(value)


def _as_int_list(value) -> list[int] | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        value = value.reshape(-1).tolist()
    return [int(item) for item in value]


def _make_chunk_indices(cu_seqlens: list[int] | None, chunk_size: int) -> list[int] | None:
    if cu_seqlens is None:
        return None
    indices = []
    for sequence_id, (bos, eos) in enumerate(zip(cu_seqlens, cu_seqlens[1:])):
        for chunk_id in range((eos - bos + chunk_size - 1) // chunk_size):
            indices.extend((sequence_id, chunk_id))
    return indices


def _to_npu(tensor: torch.Tensor | None, *, head_first: bool = False, gate_scale: float = 1.0):
    if tensor is None:
        return None
    if head_first:
        tensor = tensor.transpose(1, 2)
    if gate_scale != 1.0:
        tensor = tensor * gate_scale
    return tensor.contiguous().to('npu')


def _check_supported(op: str, inputs: dict) -> None:
    if _as_bool(inputs.get('state_v_first')):
        raise ValueError('state_v_first=True is not supported by the Ascend C comparison path.')

    k = _tensor(inputs, 'k')
    value = _tensor(inputs, 'v', 'u', 'do')
    if k is None or value is None:
        raise ValueError('The dump is missing k or a value-side tensor.')
    K, V = k.shape[-1], value.shape[-1]
    chunk_size = _as_int(inputs.get('BT'), 64)
    cu_seqlens = _as_int_list(inputs.get('cu_seqlens'))
    chunk_indices = _as_int_list(inputs.get('chunk_indices'))
    if cu_seqlens is not None and k.shape[0] != 1:
        raise ValueError(f'Ascend C varlen mode requires B=1, got B={k.shape[0]}.')
    if chunk_size not in (64, 128):
        raise ValueError(f'Ascend C supports chunk_size 64 or 128, got {chunk_size}.')
    if op == 'fwd_h' and (K != 128 or V not in (128, 256)):
        raise ValueError(f'Ascend C fwd_h requires K=128 and V in (128, 256), got K={K}, V={V}.')
    if op == 'bwd_dhu' and (K > 128 or V > 256):
        raise ValueError(f'Ascend C bwd_dhu requires K<=128 and V<=256, got K={K}, V={V}.')
    if op == 'fwd_h' and _tensor(inputs, 'g') is None and _tensor(inputs, 'gk') is None:
        raise ValueError('Ascend C fwd_h requires g or gk.')
    if op == 'fwd_h' and cu_seqlens is not None and len(chunk_indices or []) % 4 != 0:
        raise ValueError('Ascend C fwd_h varlen mode currently requires an even number of chunks.')
    if op == 'bwd_dhu' and _tensor(inputs, 'g') is None:
        raise ValueError('Ascend C bwd_dhu currently requires scalar gate g.')
    dht = _tensor(inputs, 'dht')
    if op == 'bwd_dhu' and dht is not None and dht.dtype != k.dtype:
        raise ValueError(
            'Ascend C bwd_dhu requires dht to match q/k dtype for a bitwise-equivalent replay; '
            f'got {dht.dtype} versus {k.dtype}. Dump a case without dht, or generate dht in {k.dtype}.'
        )


def _compare(name: str, actual, expected, *, rtol: float, atol: float) -> bool:
    if actual is None or expected is None:
        passed = actual is None and expected is None
        print(f'{name}: {"PASS" if passed else "FAIL"} (actual={actual is not None}, expected={expected is not None})')
        return passed

    actual = actual.detach().float().cpu()
    expected = expected.detach().float().cpu()
    if actual.shape != expected.shape:
        print(f'{name}: FAIL shape actual={tuple(actual.shape)} expected={tuple(expected.shape)}')
        return False

    diff = (actual - expected).abs()
    finite = torch.isfinite(actual) & torch.isfinite(expected)
    max_abs = diff[finite].max().item() if finite.any() else float('nan')
    mean_abs = diff[finite].mean().item() if finite.any() else float('nan')
    passed = torch.allclose(actual, expected, rtol=rtol, atol=atol, equal_nan=True)
    print(
        f'{name}: {"PASS" if passed else "FAIL"} shape={tuple(actual.shape)} '
        f'max_abs={max_abs:.6e} mean_abs={mean_abs:.6e} rtol={rtol:g} atol={atol:g}'
    )
    return passed


def _common_metadata(inputs: dict) -> tuple[int, list[int] | None, list[int] | None]:
    chunk_size = _as_int(inputs.get('BT'), 64)
    cu_seqlens = _as_int_list(inputs.get('cu_seqlens'))
    chunk_indices = _as_int_list(inputs.get('chunk_indices'))
    if chunk_indices is None:
        chunk_indices = _make_chunk_indices(cu_seqlens, chunk_size)
    return chunk_size, cu_seqlens, chunk_indices


def _replay_fwd(inputs: dict, expected: dict, gate_scale: float, rtol: float, atol: float) -> bool:
    from fla_npu.ops.ascendc import chunk_gated_delta_rule_fwd_h

    chunk_size, cu_seqlens, chunk_indices = _common_metadata(inputs)
    h, v_new, final_state = chunk_gated_delta_rule_fwd_h(
        _to_npu(_tensor(inputs, 'k'), head_first=True),
        _to_npu(_tensor(inputs, 'w'), head_first=True),
        _to_npu(_tensor(inputs, 'v', 'u'), head_first=True),
        g=_to_npu(_tensor(inputs, 'g'), head_first=True, gate_scale=gate_scale),
        gk=_to_npu(_tensor(inputs, 'gk'), head_first=True),
        initial_state=_to_npu(_tensor(inputs, 'h0')),
        output_final_state=_tensor(expected, 'ht', 'final_state') is not None,
        chunk_size=chunk_size,
        save_new_value=True,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        use_exp2=False,
        transpose_state_layout=False,
    )
    torch.npu.synchronize()
    results = (
        _compare('h', h.transpose(1, 2), _tensor(expected, 'h'), rtol=rtol, atol=atol),
        _compare('v_new', v_new.transpose(1, 2), _tensor(expected, 'v_new'), rtol=rtol, atol=atol),
        _compare('ht', final_state, _tensor(expected, 'ht', 'final_state'), rtol=rtol, atol=atol),
    )
    return all(results)


def _replay_bwd(inputs: dict, expected: dict, gate_scale: float, rtol: float, atol: float) -> bool:
    from fla_npu.ops.ascendc import chunk_gated_delta_rule_bwd_dhu

    chunk_size, cu_seqlens, chunk_indices = _common_metadata(inputs)
    scale = inputs.get('scale')
    if isinstance(scale, torch.Tensor):
        scale = scale.item()
    if scale is None:
        scale = _tensor(inputs, 'q').shape[-1] ** -0.5
    dh, dh0, dv2 = chunk_gated_delta_rule_bwd_dhu(
        _to_npu(_tensor(inputs, 'q'), head_first=True),
        _to_npu(_tensor(inputs, 'k'), head_first=True),
        _to_npu(_tensor(inputs, 'w'), head_first=True),
        _to_npu(_tensor(inputs, 'do'), head_first=True),
        _to_npu(_tensor(inputs, 'dv'), head_first=True),
        float(scale),
        chunk_size,
        g=_to_npu(_tensor(inputs, 'g'), head_first=True, gate_scale=gate_scale),
        gK=_to_npu(_tensor(inputs, 'gk'), head_first=True),
        h0=_to_npu(_tensor(inputs, 'h0')),
        dht=_to_npu(_tensor(inputs, 'dht')),
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        use_exp2=False,
        transpose_state_layout=False,
    )
    torch.npu.synchronize()
    results = (
        _compare('dh', dh.transpose(1, 2), _tensor(expected, 'dh'), rtol=rtol, atol=atol),
        _compare('dh0', dh0, _tensor(expected, 'dh0'), rtol=rtol, atol=atol),
        _compare('dv2', dv2.transpose(1, 2), _tensor(expected, 'dv2'), rtol=rtol, atol=atol),
    )
    return all(results)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dump', type=Path, help='Combined FLA dump generated with FLA_GDN_DUMP_DIR.')
    parser.add_argument('--outputs', type=Path, help='Output file for the legacy split-dump format.')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--rtol', type=float, default=6e-3)
    parser.add_argument('--atol', type=float, default=6e-3)
    args = parser.parse_args()

    op, inputs, expected, gate_base = _load_payload(args.dump, args.outputs)
    _check_supported(op, inputs)
    gate_scale = math.log(2.0) if gate_base == 2 else 1.0

    import torch_npu  # noqa: F401

    torch.npu.set_device(args.device)
    passed = (
        _replay_fwd(inputs, expected, gate_scale, args.rtol, args.atol)
        if op == 'fwd_h'
        else _replay_bwd(inputs, expected, gate_scale, args.rtol, args.atol)
    )
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
