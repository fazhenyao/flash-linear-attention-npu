"""recurrent_kda 的 ATK 泛化用例生成器。"""

from __future__ import annotations

import json
from copy import deepcopy

try:
    from atk.case_generator.generator.base_generator import CaseGenerator
    from atk.case_generator.generator.generate_types import GENERATOR_REGISTRY
    from atk.configs.case_config import CaseConfig
except ModuleNotFoundError as exc:
    if exc.name != "atk":
        raise
    CaseGenerator = None
    GENERATOR_REGISTRY = None
    CaseConfig = None

OP_NAME = "recurrent_kda"
def base_spec(**values):
    spec = dict(dtype="bf16", B=1, T=2, H=2, HV=4, K=128, V=128,
                layout="BSND", state_v_first=True, soc="ascend950", op=OP_NAME)
    spec.update(values)
    return spec


def profiles():
    output = [base_spec(name="bf16_small", B=2, state_zero=True)]
    for flags in range(8):
        output.append(base_spec(name=f"fused_{flags}", T=9,
            use_qk_l2norm_in_kernel=bool(flags & 1),
            use_gate_in_kernel=bool(flags & 2),
            use_beta_sigmoid_in_kernel=bool(flags & 4),
            g_dtype="bf16" if flags % 2 else "fp32",
            beta_dtype="bf16" if flags % 3 else "fp32",
            qkv_view="fused", inplace=bool(flags & 1)))
    for i, tokens in enumerate([1, 2, 4, 8, 15, 16, 17, 31, 32, 33, 65, 129, 257, 1024]):
        output.append(base_spec(name=f"window_{tokens}", T=tokens,
            layout="TND" if i % 2 else "BSND", state_v_first=bool(i % 2),
            state_dtype="bf16" if i % 3 else "fp32",
            use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True,
            use_beta_sigmoid_in_kernel=True, safe_gate=bool(i % 2)))
    for ratio in [1, 3, 16]:
        output.append(base_spec(name=f"ratio_{ratio}", H=1, HV=ratio, T=17,
            use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True))
    for pattern in ["zero", "small"]:
        output.append(base_spec(name=f"norm_{pattern}", T=17, pattern=pattern,
            use_qk_l2norm_in_kernel=True))
    for view in ["fused", "head_gap", "feature_gap"]:
        output.append(base_spec(name=f"view_{view}", T=17, B=2, qkv_view=view,
            strided_state=True, use_gate_in_kernel=True, use_qk_l2norm_in_kernel=True))
    for packed in ["packed", "speculative"]:
        for layout in ["BSND", "TND"]:
            output.append(base_spec(name=f"slots_{packed}_{layout}", T=17,
                lengths=[0, 9, 17, 0, 33], layout=layout, padding=5,
                indices=packed, accepted=9, metadata_dtype="int32" if layout == "TND" else "int64",
                state_v_first=layout == "TND", state_dtype="bf16" if layout == "TND" else "fp32",
                strided_state=True, inplace=layout == "BSND",
                use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True,
                safe_gate=True, use_beta_sigmoid_in_kernel=True, allow_neg_eigval=True))
    output += [
        base_spec(name="dense_no_metadata", B=2, T=17, cu=False),
        base_spec(name="tnd_no_metadata", T=33, layout="TND", cu=False),
        base_spec(name="zero_initial", T=17, initial_none=True, use_qk_l2norm_in_kernel=True),
        base_spec(name="large_gate", T=17, use_gate_in_kernel=True, large_gate=True),
        base_spec(name="v256_regression", T=9, V=256, state_v_first=False,
                  state_dtype="bf16", use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True),
        base_spec(name="empty_sequences", T=17, lengths=[0, 17, 0, 9], padding=3),
        base_spec(name="norm_signal", T=17, pattern="signal", state_zero=True,
                  use_qk_l2norm_in_kernel=True),
    ]
    return output


PROFILES = profiles()


def model_specs():
    result = []
    for batch in [32, 64, 72]:
        for hv in [32, 48, 128]:
            for tokens in [1, 2, 4, 8]:
                result.append(base_spec(name=f"B{batch}_H16_HV{hv}_T{tokens}",
                    B=batch, T=tokens, H=16, HV=hv, inplace=True, qkv_view="fused",
                    use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True,
                    use_beta_sigmoid_in_kernel=True, seed=20261020, case_id=len(result)))
    return result


def _dtype(dtype):
    return {"bf16": "bf16", "fp16": "fp16", "fp32": "fp32"}.get(dtype, "bf16")

def _spec(index):
    profile = deepcopy(PROFILES[index % len(PROFILES)])
    profile.update({"op": OP_NAME, "case_id": index, "seed": 20260817 + index, "route": "ascendc", "soc": "ascend950"})
    return profile

if GENERATOR_REGISTRY is not None:
    @GENERATOR_REGISTRY.register("generator_recurrent_kda")
    class Generator(CaseGenerator):
        def __init__(self, config):
            super().__init__(config)

        def after_case_config(self, case_config: CaseConfig) -> CaseConfig:
            index = max(int(self.index) - 1, 0)
            spec = _spec(index)
            case_config.id = index
            case_config.default_seed = spec["seed"]
            case_config.name = f"{OP_NAME}_{index:04d}_{spec.get('name', 'case')}"
            for item in case_config.inputs:
                cfg = item[0] if isinstance(item, list) else item
                if cfg.name == "low_precision_marker":
                    cfg.dtype = _dtype(spec.get("dtype", "bf16"))
                elif cfg.name == "case_spec":
                    cfg.range_values = json.dumps(spec, ensure_ascii=False, separators=(",", ":"))
                elif cfg.name in spec:
                    cfg.range_values = spec[cfg.name]
            return case_config


def generate_packages():
    # Accuracy candidates, user model shapes, and manually selected key paths
    # have separate sources; performance/mss are never synthesized by gen_cases.
    from pathlib import Path
    root = Path(__file__).resolve().parent
    template = json.loads((root / "atk_recurrent_kda.json").read_text(encoding="utf-8"))[0]
    groups = {
        "atk_recurrent_kda.json": [_spec(i) for i in range(len(PROFILES) * 3)],
        "atk_recurrent_kda_perf.json": model_specs(),
        "atk_recurrent_kda_mss.json": [base_spec(**item) for item in [
            dict(name="key0_short", T=1, inplace=True),
            dict(name="key0_window", T=17, use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True),
            dict(name="key0_tail", T=33, layout="TND", qkv_view="fused", state_dtype="bf16"),
            dict(name="key0_transpose", T=17, state_v_first=False, strided_state=True),
            dict(name="key0_slots", T=17, indices="speculative", inplace=True,
                 use_beta_sigmoid_in_kernel=True, allow_neg_eigval=True),
            dict(name="key0_packed", T=17, lengths=[0,9,17], layout="TND", indices="packed"),
            dict(name="key0_fallback", T=9, qkv_view="feature_gap"),
            dict(name="key0_v256", T=9, V=256, state_dtype="bf16", state_v_first=False),
        ]],
    }
    from copy import deepcopy
    for filename, specs in groups.items():
        cases = []
        for index, spec in enumerate(specs):
            spec.setdefault("seed", 20261020); spec["case_id"] = index
            case = deepcopy(template)
            case.update(id=index, default_seed=spec["seed"], name=f"recurrent_kda_{index:04d}_{spec['name']}")
            for value in case["inputs"]:
                if value["name"] == "case_spec":
                    value["range_values"] = json.dumps(spec, separators=(",", ":"))
                elif value["name"] in spec:
                    value["range_values"] = spec[value["name"]]
            cases.append(case)
        # One case per line keeps changes to the spec/seed matrix reviewable.
        content = "[\n" + ",\n".join(
            "  " + json.dumps(case, ensure_ascii=False, separators=(",", ":"))
            for case in cases) + "\n]\n"
        (root / filename).write_text(content, encoding="utf-8")
        print(filename, len(cases))


if __name__ == "__main__":
    generate_packages()
