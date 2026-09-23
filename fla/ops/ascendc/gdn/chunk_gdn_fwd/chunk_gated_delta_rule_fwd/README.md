# ChunkGatedDeltaRuleFwd

## 功能

将 chunk 内累加、KKT、三角求逆、W/U 重算、状态更新 H、输出 O 六个阶段融合在同一算子内。A2/A3 使用 arch22，A5 使用 arch35；内部不调用 prepare、独立 H 或独立 O 的 ACLNN 接口。

支持 FP16/BF16、K=128、V=128/256、chunk_size=64/128、MHA/GVA、定长/变长序列及可选初始/最终状态。Hv 必须能被 Hk 整除。

## 输入与布局

- q/k/v：四维，FP16/BF16；q/k shape 相同。
- BNSD/NTD：q/k 为 [B,Hk,T,K]，v 为 [B,Hv,T,V]；A2/A3/A5 支持。
- BSND/TND：q/k 为 [B,T,Hk,K]，v 为 [B,T,Hv,V]；A2 原生支持，其他产品本路径暂不支持。
- g/beta：固定 [B,T,Hv]，FP32 或与 q 同 dtype；内部转为 FP32 后计算。
- initial_state：可选 [N,Hv,K,V]，FP32 或与 q 同 dtype。
- cu_seqlens/chunk_indices：同时提供或同时省略；变长输入物理 B=1，累计长度从0开始、以T结束，chunk索引按 sequence-major 顺序。

## 输出

- o：固定 [B,T,Hv,V]，与 q 同 dtype。
- final_state：可选 [N,Hv,K,V]，与初始状态同 dtype，无初始状态时为 FP32。
- g_cumsum：可选 [B,T,Hv]，FP32。
- A：可选 [B,Hv,T,chunk_size]，与 q 同 dtype。

Python 保留十元组接口：
`(o, final_state, g_cumsum, A, beta_eff, h, q_hat, k_hat, q_rstd, k_rstd)`。
`disable_recompute=True` 导出 g_cumsum/A，False 时不导出；内部仍计算必要中间量。
`output_final_state` 控制最终状态。q_hat/k_hat 为输入 q/k 的别名，其余扩展输出为 None。

## 参数范围

当前六合一路径仅支持 `use_exp2=False`、`use_qk_l2norm_in_kernel=False`、
`use_gate_in_kernel=False`、`use_beta_sigmoid_in_kernel=False`、`allow_neg_eigval=False`、
`return_intermediate_states=False`、`state_v_first=False`，a_log/dt_bias 必须为空。
ACLNN 参数数量和顺序保持不变；不支持的扩展组合明确报错，不静默忽略。

## 验证

精度参考为 CPU FP64 recurrence 与六个公开小算子：chunk_local_cumsum、
chunk_scaled_dot_kkt、solve_tri、recompute_w_u_fwd、chunk_gated_delta_rule_fwd_h、chunk_fwd_o。
公开小算子仅用于测试标杆，不是融合算子的构建或运行依赖。

ATK 用例和执行入口见 [ATK说明](../../../../../../tests/atk/chunk_gated_delta_rule_fwd/README.md)。
ABI 静态合同检查：

```bash
python3 tests/atk/chunk_gated_delta_rule_fwd/aclnn_abi_contract.py
```
