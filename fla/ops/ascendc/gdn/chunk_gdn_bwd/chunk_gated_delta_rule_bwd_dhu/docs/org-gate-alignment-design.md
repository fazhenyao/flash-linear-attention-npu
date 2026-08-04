# ChunkGatedDeltaRuleBwdDhu 与上游 Triton Gate 语义对齐方案

## 1. 背景与目标

本文给出 NPU 仓库 `ChunkGatedDeltaRuleBwdDhu` 与上游 `flash-linear-attention` 仓库
`chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64` 的阶段性对齐方案。本阶段只处理以下差异：

- `g` 的数值域：上游 kernel 使用 `exp2`，当前 Ascend C kernel 使用自然指数 `exp`。
- 支持 `g=None`。
- 实现已有可选输入 `gk` 的 key-wise gate 语义。

本方案暂不处理 `dht`、`dh0`、`h0` 和 `state_v_first`。对齐验证继续限定：

```text
h0=None
dht=None
state_v_first=False
```

方案不得修改以下既有公共接口的参数名称、数量、顺序、默认值和原始默认行为：

- `ChunkGatedDeltaRuleBwdDhu` op prototype。
- `aclnnChunkGatedDeltaRuleBwdDhu*` 接口。
- `fla_npu.ops.ascendc.chunk_gated_delta_rule_bwd_dhu` 接口。
- legacy `torch.ops.npu.npu_chunk_gated_delta_rule_bwd_dhu` schema。

## 2. 目标数学语义

对第 `j` 个 chunk，令当前反向状态累加器为 `bdh`，chunk 最后一个有效 token 的 scalar gate 为
`g_last`，key-wise gate 为 `gk_last[k]`。目标递推为：

```text
bdv = k_chunk @ bdh

if g is not None:
    bdv *= gate(g_last - g_chunk)

dv2_chunk = bdv + dv_chunk

if g is not None:
    bdh *= gate(g_last)

if gk is not None:
    bdh *= gate(gk_last)[:, None]

q_gated = q_chunk
if g is not None:
    q_gated *= gate(g_chunk)[:, None]

bdh += q_gated.T @ dO_chunk * scale
bdh -= w_chunk.T @ dv2_chunk
```

四种 gate 组合的目标行为如下：

| `g` | `gk` | `bdv` gate | `bdh` gate | `q` gate |
| --- | --- | --- | --- | --- |
| `None` | `None` | 恒等 | 恒等 | 恒等 |
| 有 | `None` | scalar `g` | scalar `g` | scalar `g` |
| `None` | 有 | 恒等 | key-wise `gk` | 恒等 |
| 有 | 有 | scalar `g` | scalar `g` 与 key-wise `gk` | scalar `g` |

`gk` 只作用于传递到前一个 chunk 的 `bdh`。必须先使用当前 `bdh` 计算 `k @ bdh`，再应用
`gk_last`；否则会改变上游 kernel 的数学语义。

## 3. Gate 数值域

### 3.1 内部统一自然指数域

Ascend C kernel 内部继续统一执行自然指数：

```text
gate(x) = exp(x)
```

这样可以保留当前 raw aclnn 行为以及公开 Python 参数 `use_exp2=False` 的原始默认行为，不需要修改算子
prototype 或 aclnn ABI。

### 3.2 上游 `exp2` 数值域适配

上游 kernel 执行：

```text
gate(x) = exp2(x)
```

当 Python 调用方设置已有参数 `use_exp2=True` 时，公共 wrapper 在进入 aclnn 前将 gate 转为自然指数域：

```python
LN2 = 0.6931471805599453

g_kernel = g.float() * LN2 if g is not None else None
gk_kernel = gk.float() * LN2 if gk is not None else None
```

利用：

```text
exp(x * ln(2)) = exp2(x)
```

因此参数语义为：

| `use_exp2` | 调用方 gate 数值域 | wrapper 行为 | kernel 行为 |
| --- | --- | --- | --- |
| `False` | 自然指数域 | 原样传入 | `exp` |
| `True` | 上游 log2/`exp2` 域 | FP32 乘 `ln(2)` | `exp` |

raw aclnn 接口不增加 `use_exp2` 属性，继续接收自然指数域 gate。数值域适配只发生在已有 Python API 和
legacy compatibility wrapper 中。

### 3.3 dtype 策略

转换应在 FP32 完成。`g` 保留输入 dtype 和 FP32 两种类型；`gk` 固定使用 FP32：

```text
g dtype = q dtype 或 FP32
gk dtype = FP32
```

支持组合为：

```text
g  in {None, q dtype, FP32}
gk in {None, FP32}
```

当 `use_exp2=True` 时，wrapper 生成的内部 gate tensor保持 FP32，不将乘 `ln(2)` 后的结果 cast 回
FP16/BF16。这样避免在进入 kernel 前引入额外低精度舍入。

## 4. `g=None` 支持方案

### 4.1 Tiling

当前 tiling 在缺少 `g` 时直接失败。调整后：

- `g` 继续保持 optional prototype。
- tiling data 增加内部标志 `hasG`。
- 仅在 `hasG=true` 时检查 `g` shape 和 dtype。
- `g=None` 时不得访问 `g` 的 global memory 地址。

采用 3 个 tiling key，只编码 `g` 的存在性和 dtype：

```text
Key 1: g 为 q dtype，gk 为 None 或 FP32
Key 2: g 为 FP32，    gk 为 None 或 FP32
Key 3: g 为 None，    gk 为 None 或 FP32
```

tiling data 仍保留 `hasG` 和 `hasGk`，用于区分 optional pointer 是否有效。TilingKey 决定存在的 gate
指针元素类型，二者职责不同。

由于 `gk` 存在时固定为 FP32，三个 key 的 `GKT` 均实例化为 `float`。`gk=None` 与
`gk=FP32` 通过运行时 `hasGk` 区分，不需要为 `gk` 的存在性增加 TilingKey。

Kernel 模板扩展为：

```cpp
template <typename DT, typename GT, typename GKT>
ChunkGatedDeltaRuleBwdDhuKernelImpl(...)
```

建议分发关系：

| Key | `DT` | `GT` | `GKT` | `hasG` | `hasGk` |
| --- | --- | --- | --- | --- | --- |
| 1 | `DTYPE_Q` | `DTYPE_Q` | `float` | 1 | 0/1 |
| 2 | `DTYPE_Q` | `float` | `float` | 1 | 0/1 |
| 3 | `DTYPE_Q` | `DTYPE_Q` | `float` | 0 | 0/1 |

Key 3 的 `GT` 是占位模板类型；`hasG=0` 时 kernel 不得解引用 `g`。

### 4.2 Vector 路径

保留当前 AIC/AIV 同步协议和 gated-Q workspace，减少对已收敛 Cube 流水的影响。

当前 `CalcGatedQ` 应扩展为以下逻辑：

```text
if hasG:
    g_exp = exp(g)
    q_workspace = q * g_exp
    g_last_exp = exp(g_last)
else:
    q_workspace = q
    g_last_exp = 1
```

即使 `g=None`，AIV 仍把原始 `q` 写入现有 gated-Q workspace，并设置原有
`CROSS_CORE_V2C_GQ` flag。Cube 继续从同一 workspace 执行 `q.T @ dO`，无需新增一套 Cube 数据路径。

`CalcDv2` 调整为：

```text
bdv = k @ bdh
if hasG:
    bdv *= exp(g_last - g)
dv2 = bdv + dv
```

`UpdateDh` 调整为：

```text
if hasG:
    bdh *= exp(g_last)
bdh += term1 * scale - term2
```

本阶段仍限定 `dht=None`，因此最后一个 chunk 的 `bdh` 为零，现有 `dv2[last] = dv[last]` 快速路径仍成立。

## 5. `gk` 支持方案

### 5.1 数据布局与地址

NPU 输入布局保持现有接口约定：

```text
gk: [B, Hv, T, K]
```

对 value head `hv`，其 sequence 起点地址为：

```text
gmOffsetGk = ((b * Hv + hv) * T + sequence_start) * K
```

当前 chunk 的最后一个有效 token：

```text
last_idx = min((chunk_idx + 1) * chunk_size, sequence_length) - 1
```

对应 key-wise gate 地址：

```text
gk_last = gk[gmOffsetGk + last_idx * K : ... + K]
```

两个 AIV 子核分别处理 K 的一半，第二个子核需要额外增加 `halfK` 偏移。

### 5.2 Kernel 接入

当前 kernel entry 保留了 `gk` 参数但明确忽略。实现时需要依次透传到：

```text
chunk_gated_delta_rule_bwd_dhu entry
  -> ChunkGatedDeltaRuleBwdDhuKernelImpl
    -> GDRVec::Init
      -> InitGlobalTensor
```

Cube 不需要读取 `gk`；`gk` 只影响 Vector 侧的 `bdh` 更新。

### 5.3 应用位置

在 `UpdateDh` 中，读取当前 `bdh` 后、叠加 `term1` 和 `term2` 前执行：

```text
if hasG:
    bdh *= exp(g_last)

if hasGk:
    bdh *= exp(gk_last)[:, None]

bdh += term1 * scale
bdh -= term2
```

不能在 `CalcDv2` 之前应用 `gk`，否则 `k @ bdh` 会错误地使用已经衰减的状态。

### 5.4 UB 规划

不应物化完整的 `[halfK, V]` FP32 gate。建议分配或复用：

```text
gkLocal:     [halfK]，FP32
gkBcLocal:   按 BlockMul 行广播所需的最小 FP32 block
```

随后将 `exp(gk_last)` 沿 V 维广播，与 `bdhCastLocal[halfK, V]` 相乘。

`gk` 处理位于 `UpdateDh` 阶段，可以优先复用 `CalcGatedQ` 或 `CalcDv2` 已结束生命周期的 UB 区域。
tiling 的 `CalcUb()` 仍需按实际 live range 重新计算峰值，覆盖：

- `g=None/gk=None`
- `g!=None/gk=None`
- `g=None/gk!=None`
- `g!=None/gk!=None`

三个 key 中 `hasGk` 均是运行时分支。如果 profiling 证明该分支明显影响无 `gk` 热路径，可以后续为
`gk=None` 增加专用 key；这不属于第一版正确性要求。

## 6. Host 与接口校验

不修改已有输入、输出和属性，只补充 optional 参数的真实语义与校验。

aclnn/op_host 应检查：

```text
g:
  None，或 [B, Hv, T]
  dtype 为 q dtype 或 FP32

gk:
  None，或 [B, Hv, T, K]
  dtype 固定为 FP32
```

op_def dtype 组合需要覆盖以下 4 组，其中 `gk` 为 optional，因此每组同时覆盖对应的 `gk=None` 情况：

| `q/k/w/dO/dv` | `g` | `gk` |
| --- | --- | --- |
| BF16 | BF16 | FP32 |
| FP16 | FP16 | FP32 |
| BF16 | FP32 | FP32 |
| FP16 | FP32 | FP32 |

实际 `DataType(...)` 注册应展开为无歧义的 dtype tuple，确保每个输入/输出 dtype 列表索引一一对应。

现有约束继续保持：

```text
Hv % Hk == 0
chunk_size in {64, 128}
varlen 时 B == 1
```

ctypes 稳定入口与 legacy wrapper 都需要激活已有 `use_exp2` 参数，但不得改变默认值：

```text
use_exp2=False 保持原自然指数域行为
use_exp2=True 启用与上游 kernel 的 log2 域对齐
```

## 7. 实施步骤

### 阶段一：支持 `g=None`

1. tiling data 增加 `hasG`，允许 optional `g` 缺失。
2. 增加 no-g tiling key。
3. AIV 在 no-g 模式下将原始 `q` 写入 gated-Q workspace。
4. `CalcDv2` 和 `UpdateDh` 跳过 scalar gate 运算。
5. 保留现有 AIC/AIV flag 和 workspace 协议。

### 阶段二：支持 `gk`

1. tiling data 增加 `hasGk`，并校验存在的 `gk` 固定为 FP32。
2. 将 TilingKey 扩展为 3 种 `g` 存在性/dtype 组合，所有 key 的 `GKT` 固定为 `float`。
3. 补充 `gk` shape/dtype 校验，并在 op_def 中增加 FP32 `gk` 组合。
4. 将 kernel 模板扩展为 `<DT, GT, GKT>`，并将 `gk` 透传至 `GDRVec`。
5. 计算当前 chunk 的 `gk_last` 地址。
6. 在 `UpdateDh` 中应用 key-wise gate。
7. 重新核算 UB 峰值。

### 阶段三：激活 `use_exp2`

1. ctypes wrapper 在 `use_exp2=True` 时将 `g/gk` 乘 `ln(2)`。
2. legacy wrapper 实现相同转换。
3. 保证转换后的 `g/gk` 使用 FP32 tensor 进入对应 FP32 tiling key。
4. 更新 README、aclnn 文档和调用示例。

## 8. 验证方案

### 8.1 与上游 kernel 对齐

上游布局：

```text
q/k:   [B, T, Hk, K]
w:     [B, T, Hv, K]
dO/dv: [B, T, Hv, V]
g:     [B, T, Hv]
gk:    [B, T, Hv, K]
dh:    [B, NT, Hv, K, V]
```

NPU 布局：

```text
q/k:   [B, Hk, T, K]
w:     [B, Hv, T, K]
dO/dv: [B, Hv, T, V]
g:     [B, Hv, T]
gk:    [B, Hv, T, K]
dh:    [B, Hv, NT, K, V]
```

对齐测试需转置输入和输出，并向 NPU Python API 传入相同的上游 log2 域 gate：

```python
dh_npu, _, dv2_npu = chunk_gated_delta_rule_bwd_dhu(
    ...,
    g=g_npu,
    gK=gk_npu,
    use_exp2=True,
)
```

### 8.2 必测矩阵

Gate 组合：

```text
g=None,     gk=None
g!=None,    gk=None
g=None,     gk!=None
g!=None,    gk!=None
```

Shape 与模式至少覆盖：

- `K=64/128`。
- `V=64/128/256`。
- `Hk=Hv` 与 `Hv>Hk`。
- `chunk_size=64/128`。
- `T` 整除和不整除 `chunk_size`。
- dense 与 varlen。
- FP16 与 BF16。
- `g` 为输入 dtype 和 FP32，`gk` 固定为 FP32。

验证内容：

- `dh` 与上游输出在布局转换后精度一致。
- `dv2` 与上游输出在布局转换后精度一致。
- `use_exp2=False` 的历史自然指数域 case 不回归。
- `g=None` 时 kernel 不读取空指针。
- `gk` 只影响跨 chunk 的 `bdh`，不影响当前 chunk 的 `k @ bdh`。
- varlen 尾 chunk 和 dense 尾 chunk 正确。
- A2、A3、A5 目标按风险执行单算子回归。

## 9. 风险与边界

- 本方案只对齐 gate 语义；非零 `dht`、`dh0`、`h0` 和 `state_v_first` 仍不在支持范围内。
- `use_exp2=True` 的转换必须同时覆盖 `g` 和 `gk`，否则组合 gate 的数值域不一致。
- `use_exp2=True` 时不得将乘 `ln(2)` 后的 `g/gk` 回写为 FP16/BF16，否则会放大长序列递推误差。
- no-g 路径继续经过 gated-Q workspace，第一版以降低实现风险为优先；如 profiling 证明复制开销显著，可另行设计 Cube 直读 q 的专用 tiling key。
- `gk` 必须在 `k @ bdh` 之后应用，这是与上游语义对齐的正确性红线。

## 10. 结论

本方案在保持既有 ABI 和默认行为的前提下，将 Ascend C kernel 内部统一为自然指数域：

```text
kernel: 始终执行 exp
use_exp2=False: gate 原样传入
use_exp2=True: wrapper 将上游 log2 gate 乘 ln(2)
g=None: 使用恒等 scalar gate
gk: 仅在 UpdateDh 阶段沿 K 维衰减 bdh
TilingKey: 3 种组合，仅区分 g 的 None、q dtype 和 FP32；gk 固定为 FP32 并由 hasGk 区分是否存在
```

完成后，NPU 算子可在本文限定范围内覆盖上游 kernel 的四种 `g/gk` 组合，同时不修改已发布的算子
prototype、aclnn ABI 和 Python 参数列表。
