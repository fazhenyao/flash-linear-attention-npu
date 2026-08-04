# bwd_dhu 第三阶段：状态输入输出对齐

## 目标

在第一阶段 Gate/gk 对齐和第二阶段 Kernel 内统一 `*ln(2)` 的基础上，激活已有接口中的
`h0`、`dht` 和 `dh0`，对齐上游 `chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64` 的状态递推语义。

本阶段不新增或调整公开参数。`transpose_state_layout` 保持原有兼容行为：无论传入 `True` 或
`False`，均不改变布局，状态固定采用 K-first `[N, Hv, K, V]`。

## 数值语义

```text
b_dh = dht if dht is not None else 0

for chunk_idx in reversed(range(chunk_num)):
    dh[chunk_idx] = b_dh
    dv2 = gated(k @ b_dh) + dv
    b_dh = gated(b_dh) + gated_q.T @ do * scale - w.T @ dv2

dh0 = b_dh if h0 is not None else None
```

- `dht` 是反向扫描初值。
- `h0` 数值不参与计算，只决定是否生成 `dh0`。
- 固定长度时 `N=B`；变长时 `N=len(cu_seqlens)-1`。
- wrapper 将 `h0/dht` 规范化为 FP32，Kernel 状态输入和 `dh0` 输出均为 FP32。
- `dh`、`dv2` 继续保持 q/value 侧低精度 dtype 和既有布局。

## Kernel 分工

### AIV

1. 每个 `(sequence, value_head)` 开始时，将 FP32 `dht` 转为计算 dtype 后写入最后一个
   chunk 的 `dh`；无 `dht` 时写零。
2. 对每个 chunk 生成 gated q，融合 AIC 计算的 `k @ dh` 得到 `dv2`。
3. 对当前 `dh` 应用 scalar/key-wise Gate，融合 AIC 的两个状态更新项。
4. 非首 chunk 将更新结果写入前一个 chunk 的 `dh`；首 chunk 在存在 `h0` 时以 FP32 写入 `dh0`。

### AIC

1. 所有 chunk 均计算 `k @ dh`，包括使用 `dht` 初值的最后一个 chunk。
2. 所有 chunk 均计算 `gated_q.T @ do` 和 `w.T @ dv2`；首 chunk 的结果用于可选 `dh0`。
3. 通过既有跨核事件与 AIV 逐 chunk 串行同步。

## Tiling 与兼容性

- TilingData 增加 `hasDht`、`needDh0` 两个存在性标志。
- TilingKey 仍为 3 个，仅由 `g` 的存在性和 dtype 决定；状态输入不增加模板 dtype 分支。
- `h0/dht/dh0` 均校验为 `[N,Hv,K,V]`，raw aclnn 状态 tensor 使用 FP32。
- Python 稳定入口与 legacy wrapper 接受原调用参数并在内部转 FP32。
- `transpose_state_layout` 不报错、不转置，保持此前被忽略的行为。

## 验证矩阵

- fixed/varlen。
- `h0/dht` 四种存在性组合。
- `dht=0` 与省略 `dht` 的退化一致性。
- 改变 `h0` 数值不影响 `dh/dh0/dv2`。
- `transpose_state_layout=True/False` 结果与布局一致。
- Gate 模式覆盖 `g`、`gk`、无 Gate，并保持三个既有 TilingKey。
