"""Exercise published Python backends and raw ACLNN descriptors on the NPU."""
import argparse
import ctypes
import sys
import warnings
from pathlib import Path

import torch
import torch_npu

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from executor_recurrent_kda import build_inputs, call_options, run_cpu
from gen_recurrent_kda import base_spec
from fla_npu.ops.ascendc import _aclnn_ctypes, _stable
from atk.tasks.post_process.mixed_tolerance_benchmark_compare import MixedToleranceBenchmarkAccuracyCompare


def direct_aclnn(q, k, v, g, beta, initial_state, **options):
    """Bypass both wrappers' shape normalization and State allocation policies."""
    out = torch.empty_like(v, memory_format=torch.contiguous_format)
    shape, strides = initial_state.shape, initial_state.stride()
    span = sum((size - 1) * stride for size, stride in zip(shape, strides)) + 1
    storage = torch.full((span + 32,), 17.0, dtype=initial_state.dtype, device=initial_state.device)
    final = torch.as_strided(storage, shape, strides, storage_offset=16)
    names = ("cu_seqlens", "ssm_state_indices", "A_log", "dt_bias", "num_accepted_tokens")
    inputs = (q, k, v, g, beta, initial_state, *(options.get(name) for name in names))
    layout = ctypes.create_string_buffer(options.get("layout", "BSND").encode())

    def arguments(ctx):
        args = [ctx.tensor(tensor, name) for tensor, name in zip(
            inputs, ("q", "k", "v", "g", "beta", "initial_state", *names))]
        args += [ctypes.cast(layout, ctypes.c_char_p), ctypes.c_double(options["scale"])]
        args += [ctypes.c_bool(options.get(name, False)) for name in (
            "output_final_state", "inplace_final_state", "use_qk_l2norm_in_kernel",
            "use_gate_in_kernel", "use_beta_sigmoid_in_kernel", "allow_neg_eigval", "safe_gate")]
        return [*args, ctypes.c_double(options.get("lower_bound", -5.0)),
                ctypes.c_bool(options["state_v_first"]), ctx.tensor(out, "attn_out"),
                ctx.tensor(final, "final_state")]

    _aclnn_ctypes._call_aclnn("aclnnRecurrentKda", arguments, (out, initial_state, final))
    torch.npu.synchronize()
    assert torch.all(storage[:16] == 17) and torch.all(storage[-16:] == 17)
    return out, final


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.npu.set_device(args.device)
    device = torch.device(f"npu:{args.device}")
    compare = MixedToleranceBenchmarkAccuracyCompare(None)
    count = 0
    backends = {"stable": _stable.npu_recurrent_kda,
                "ctypes": _aclnn_ctypes.npu_recurrent_kda, "aclnn": direct_aclnn}
    for backend, call in backends.items():
        for v_first in (True, False):
            for dtype in ("fp32", "bf16"):
                for scenario in ("empty_request", "sparse_pool", "rank3_no_cu", "rank4_cu"):
                    spec = base_spec(B=2, T=2, H=1, HV=1, state_dtype=dtype,
                                     state_v_first=v_first, inplace=False, pattern="signal",
                                     use_qk_l2norm_in_kernel=True, strided_state=True)
                    if scenario == "empty_request":
                        spec.update(lengths=[0, 2])
                    elif scenario == "sparse_pool":
                        spec.update(indices="packed", accepted=1)
                    elif scenario == "rank3_no_cu":
                        spec.update(B=1, cu=False)
                    inputs = build_inputs(spec, device)
                    state = inputs["initial_state"]
                    # Preserve values while exercising a nonzero logical offset
                    # and gapped outer stride on each entry.
                    backing = torch.zeros((state.shape[0] + 2, 2, 128, 128),
                                          dtype=state.dtype, device=device)
                    inputs["initial_state"] = backing[1:-1, :1]
                    inputs["initial_state"].copy_(state)
                    before = inputs["initial_state"].clone()
                    if scenario.startswith("rank"):
                        for name in ("q", "k", "v", "g", "beta"):
                            tensor = inputs[name]
                            inputs[name] = (tensor.squeeze(0) if scenario == "rank3_no_cu" else
                                            tensor.reshape(spec["B"], spec["T"], *tensor.shape[1:]))
                    expected_shape = inputs["v"].shape
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always", DeprecationWarning)
                        output, final = call(**inputs, **call_options(spec))
                    torch.npu.synchronize()
                    assert output.shape == expected_shape, (backend, scenario, output.shape)
                    if backend != "aclnn":
                        deprecated = [w for w in caught if issubclass(w.category, DeprecationWarning)]
                        assert len(deprecated) == int(scenario.startswith("rank")), (backend, scenario, caught)
                    golden = run_cpu(spec, high_precision=True)
                    assert compare.compute_accuracy_result(
                        output.cpu().reshape(golden[0].shape), golden[0].float(), "output_0.pt").result
                    assert compare.compute_accuracy_result(final.cpu(), golden[1].float(), "output_1.pt").result
                    assert torch.equal(inputs["initial_state"], before)
                    assert final.data_ptr() != inputs["initial_state"].data_ptr()
                    if scenario == "empty_request":
                        assert torch.equal(final[0], before[0])
                    elif scenario == "sparse_pool":
                        used = set(inputs["ssm_state_indices"].cpu().tolist())
                        untouched = [i for i in range(before.shape[0]) if i not in used]
                        assert torch.equal(final[untouched], before[untouched])
                    count += 1
                    print("REVIEW_CASE_OK", backend, v_first, dtype, scenario, flush=True)
    rejected = 0
    for backend, call in backends.items():
        for invalid in ("head_ratio", "q_dtype", "state_stride", "cu_start", "cu_order", "cu_capacity"):
            # Non-inplace stable calls intentionally clone/materialize State.
            # Exercise illegal kernel strides through the inplace path instead.
            spec = base_spec(T=2, HV=3 if invalid == "head_ratio" else 4, inplace=invalid == "state_stride")
            inputs = build_inputs(spec, device)
            reason = "HV must be divisible by H"
            if invalid == "q_dtype":
                inputs["q"] = inputs["q"].float()
                reason = "bfloat16" if backend == "ctypes" else "query"
            elif invalid == "state_stride":
                state = inputs["initial_state"]
                inputs["initial_state"] = torch.zeros((*state.shape[:-1], 256),
                                                      dtype=state.dtype, device=device)[..., ::2]
                reason = "inner state matrix dense"
            elif invalid.startswith("cu_"):
                offsets = {"cu_start": [1, 2], "cu_order": [0, -1], "cu_capacity": [0, 3]}[invalid]
                inputs["cu_seqlens"] = torch.tensor(offsets, dtype=torch.int64, device=device)
                reason = "cu_seqlens"
            try:
                call(**inputs, **call_options(spec))
                torch.npu.synchronize()
            except (RuntimeError, ValueError) as exc:
                message = str(exc)
                assert reason in message, (backend, invalid, message)
                python_guard = backend == "ctypes" and invalid in ("head_ratio", "q_dtype")
                if not python_guard:
                    assert "161002" in message, (backend, invalid, message)
                if invalid.startswith("cu_"):
                    assert all(text in message for text in ("EZ0037", "index=", "actual=", "capacity=2")), message
                rejected += 1
            else:
                raise AssertionError((backend, "accepted invalid", invalid))
            healthy = base_spec(T=1, H=1, HV=1, pattern="signal", state_zero=True)
            actual = call(**build_inputs(healthy, device), **call_options(healthy))
            torch.npu.synchronize()
            golden = run_cpu(healthy, high_precision=True)
            assert all(compare.compute_accuracy_result(a.cpu(), b.float(), f"output_{i}.pt").result
                       for i, (a, b) in enumerate(zip(actual, golden)))
    print(f"REVIEW_REGRESSIONS_OK cases={count} negative={rejected} backends=stable,ctypes,aclnn")


if __name__ == "__main__":
    main()
