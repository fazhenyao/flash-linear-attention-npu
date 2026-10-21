"""Installed public API checks beyond ATK's always-return-state contract."""
import argparse
import importlib.util
import sys
from pathlib import Path

import torch
import torch_npu

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from executor_recurrent_kda import build_inputs, call_options, run_cpu
from gen_recurrent_kda import base_spec
from fla_npu.ops.ascendc import recurrent_kda
from atk.tasks.post_process.mixed_tolerance_benchmark_compare import (
    MixedToleranceBenchmarkAccuracyCompare,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=6)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.npu.set_device(args.device)
    device = torch.device(f"npu:{args.device}")
    compare = MixedToleranceBenchmarkAccuracyCompare(None)
    count = 0
    for layout in ["BSND", "TND"]:
        for v_first in [False, True]:
            for inplace in [False, True]:
                for return_state in [False, True]:
                    spec = base_spec(T=17, layout=layout, state_v_first=v_first,
                                     inplace=inplace, pattern="signal", state_zero=True,
                                     use_qk_l2norm_in_kernel=True)
                    inputs = build_inputs(spec, device)
                    state = inputs["initial_state"]
                    before, version = state.clone(), state._version
                    options = call_options(spec)
                    options["output_final_state"] = return_state
                    output, final = recurrent_kda(**inputs, **options)
                    torch.npu.synchronize()
                    golden = run_cpu(spec, high_precision=True)
                    assert compare.compute_accuracy_result(output.cpu(), golden[0].float(), "output_0.pt").result
                    assert state._version - version == int(inplace)
                    if inplace:
                        assert not torch.equal(state, before)
                    else:
                        assert torch.equal(state, before)
                    if return_state:
                        assert (final.data_ptr() == state.data_ptr()) == inplace
                        assert compare.compute_accuracy_result(final.cpu(), golden[1].float(), "output_1.pt").result
                    else:
                        assert final is None
                    count += 1
            for metadata in [False, True]:
                for return_state in [False, True]:
                    spec = base_spec(B=2, T=17, cu=metadata, V=128 if v_first else 256, layout=layout, state_v_first=v_first,
                                     initial_none=True, pattern="signal", use_qk_l2norm_in_kernel=True)
                    options = call_options(spec)
                    options["output_final_state"] = return_state
                    output, final = recurrent_kda(**build_inputs(spec, device), **options)
                    golden = run_cpu(spec, high_precision=True)
                    assert compare.compute_accuracy_result(output.cpu(), golden[0].float(), "output_0.pt").result
                    assert (final is not None) == return_state
                    if final is not None:
                        assert final.dtype == torch.float32
                        assert compare.compute_accuracy_result(final.cpu(), golden[1].float(), "output_1.pt").result
                    count += 1
    for tokens in [1, 9, 17]:
        for x in [0.0, 2.0 ** -12]:
            spec = base_spec(T=tokens, H=1, HV=1, pattern="small", state_zero=True,
                             use_qk_l2norm_in_kernel=True)
            inputs = build_inputs(spec, device)
            inputs["q"].zero_(); inputs["k"].zero_()
            inputs["q"][..., 0] = x; inputs["k"][..., 0] = x
            inputs["g"].zero_(); inputs["beta"].fill_(0.5); inputs["v"].fill_(2)
            inputs["initial_state"][..., 0] = 0.25
            output, final = recurrent_kda(**inputs, **call_options(spec))
            z, current, expected = x / (x*x + 1e-6)**0.5, 0.25, []
            for _ in range(tokens):
                current = current + 0.5 * (2 - current*z) * z
                expected.append(current * z * 128**-0.5)
            target = torch.tensor(expected).reshape(1, tokens, 1, 1).expand_as(output.cpu())
            assert torch.allclose(output.cpu().float(), target, rtol=0.008, atol=1e-6)
            assert torch.allclose(final.cpu()[..., 0], torch.full_like(final.cpu()[..., 0], current),
                                  rtol=1e-5, atol=1e-6)
            assert torch.count_nonzero(final[..., 1:]) == 0
            count += 1
    for invalid in ["head_ratio", "q_dtype", "state_stride", "missing_a_log"]:
        spec = base_spec(T=17, inplace=True, HV=3 if invalid == "head_ratio" else 4,
                         use_gate_in_kernel=invalid == "missing_a_log")
        inputs = build_inputs(spec, device)
        if invalid == "q_dtype":
            inputs["q"] = inputs["q"].float()
        elif invalid == "state_stride":
            state = inputs["initial_state"]
            inputs["initial_state"] = torch.zeros((*state.shape[:-1], state.shape[-1] * 2),
                                                  device=device)[..., ::2]
        elif invalid == "missing_a_log":
            inputs["A_log"] = None
        try:
            recurrent_kda(**inputs, **call_options(spec))
            torch.npu.synchronize()
        except (RuntimeError, ValueError):
            count += 1
        else:
            raise AssertionError(f"accepted invalid {invalid}")
    # Reuse the repository's mutation/grad regression for just the affected op.
    path = Path(__file__).resolve().parents[3] / "stable_abi/regression_mutation_contract.py"
    module_spec = importlib.util.spec_from_file_location("mutation_contract", path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    module.scenario_recurrent_kda()
    print(f"PUBLIC_CONTRACT_OK cases={count} mutation_grad=pass")


if __name__ == "__main__":
    main()
