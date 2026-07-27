import torch

from fla_npu.ops.ascendc import npu_chunk_gated_delta_rule_bwd_dhu


def main():
    torch.npu.set_device(0)
    shape_q = (1, 32, 8192, 128)
    shape_v = (1, 32, 8192, 128)
    q = torch.randn(shape_q, dtype=torch.bfloat16, device="npu") * 0.005
    k = torch.randn_like(q) * 0.005
    w = torch.randn_like(q) * 0.005
    do = torch.randn(shape_v, dtype=torch.bfloat16, device="npu") * 0.005
    dv = torch.randn_like(do) * 0.005
    g = torch.linspace(-0.0001, -0.01, 8192, device="npu").view(1, 1, -1).expand(1, 32, -1).contiguous()
    for _ in range(5):
        npu_chunk_gated_delta_rule_bwd_dhu(q, k, w, do, dv, 0.0883883476, 64, g=g)
    torch.npu.synchronize()


if __name__ == "__main__":
    main()
