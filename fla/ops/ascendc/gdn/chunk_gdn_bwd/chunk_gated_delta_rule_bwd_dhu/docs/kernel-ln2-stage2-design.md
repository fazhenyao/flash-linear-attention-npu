# 第二阶段：Kernel 内统一采用 `exp2` Gate 语义

## 1. 阶段决策

第二阶段统一定义所有 `ChunkGatedDeltaRuleBwdDhu` 调用入口的 Gate 数值域为 log2 域：

```text
gate(x) = exp2(x)
```

Ascend C Kernel 继续使用自然指数指令 `Exp`，因此只要对应 Gate 存在，AIV 都固定执行：

```text
x_fp32 *= ln(2)
gate = exp(x_fp32)
```

利用：

```text
exp(x * ln(2)) = exp2(x)
```

该乘法不再由 `use_exp2` 控制。`use_exp2=True` 和 `use_exp2=False` 最终执行完全相同的 Gate 数学语义。

## 2. 与第一阶段的差异

第一阶段行为为：

| 调用方式 | wrapper | Kernel | 最终语义 |
| --- | --- | --- | --- |
| `use_exp2=False` | Gate 原样传入 | `exp(x)` | 自然指数域 |
| `use_exp2=True` | 全量 Gate `*ln2` | `exp(x)` | `exp2` 域 |
| raw aclnn | Gate 原样传入 | `exp(x)` | 自然指数域 |

第二阶段统一为：

| 调用方式 | wrapper | Kernel | 最终语义 |
| --- | --- | --- | --- |
| `use_exp2=False` | 不再 `*ln2` | tile 内固定 `*ln2` 后 `exp` | `exp2` 域 |
| `use_exp2=True` | 不再 `*ln2` | tile 内固定 `*ln2` 后 `exp` | `exp2` 域 |
| raw aclnn | Gate 原样传入 | tile 内固定 `*ln2` 后 `exp` | `exp2` 域 |

因此第二阶段不是单纯的性能融合，而是统一 Gate 输入语义。历史自然指数域行为不再保留。

## 3. 接口与兼容性影响

### 3.1 原型和 ABI

不需要增加输入、输出或属性：

```text
op prototype：不变
aclnn 函数签名：不变
Python 参数列表：不变
TilingKey：不变
TilingData：无需增加 useExp2/gateScale
```

### 3.2 行为兼容性

虽然 ABI 不变，但以下既有语义会改变：

- raw aclnn 的 `g/gk` 从自然指数域改为 log2 域。
- Python `use_exp2=False` 从自然指数域改为 `exp2` 域。
- `use_exp2` 的默认值仍为 `False`，但默认调用结果改为 `exp2` 语义。
- legacy `torch.ops.npu` 的 `use_exp2=False` 行为同步改变。

这属于公共接口的默认行为和既有参数语义变化。该变更已经获得 `@weinachuan` 针对本次实施的明确确认。

### 3.3 `use_exp2` 参数处理

为保持源码调用兼容，现有 `use_exp2` 参数不删除、不改名、不改顺序、不改默认值。

第二阶段将其定义为兼容保留参数：

```text
use_exp2=True  -> exp2 语义
use_exp2=False -> exp2 语义
```

wrapper 不应因为参数为 `False` 发出警告或拒绝调用，否则会破坏现有默认调用。README 和 Python API 文档
必须明确说明该参数在本算子中已不再切换数值域。

## 4. Host、Tiling 与 TilingKey

### 4.1 Host

Host 不需要读取或下发 `use_exp2`。现有 shape、dtype 和 optional Gate 校验保持：

```text
g  in {None, q dtype, FP32}
gk in {None, FP32}
```

### 4.2 TilingKey

继续保持第一阶段的 3 个 TilingKey：

```text
Key 1: g 为 q dtype，gk 为 None 或 FP32
Key 2: g 为 FP32，    gk 为 None 或 FP32
Key 3: g 为 None，    gk 为 None 或 FP32
```

无需增加 `use_exp2` 维度，也无需从 3 个扩展到 6 个。所有 Gate 路径都固定执行 `*ln2`。

### 4.3 TilingData

不增加：

```text
useExp2
gateScale
```

继续使用：

```text
hasG
hasGk
```

它们只控制 optional Gate 指针是否有效。

## 5. AIV 各 Stage 的修改

统一常量：

```cpp
constexpr float LN2 = 0.6931471805599453f;
```

### 5.1 Stage：`CalcGatedQ`

对存在的 scalar Gate `g`，路径改为：

```text
GM g
  -> CopyIn GT
  -> Cast FP32（g 已是 FP32 时省略 Cast）
  -> Muls(LN2)
  -> Exp
  -> q * gate
```

伪代码：

```cpp
if (hasG) {
    CopyInAndCast(gCastLocal, gGm, validLen);
    Muls(gCastLocal, gCastLocal, LN2, validLen);
    Exp(gExpLocal, gCastLocal, validLen);
}
```

必须覆盖：

- 完整 chunk 的两个 AIV subBlock。
- tail chunk 的有效长度。
- `curCalcBT == 0` 时只读取 `g_last` 的分支。
- `g` 为 q dtype 和 FP32 两条模板路径。

`gLast` 必须从已经 `*ln2` 的 `gCastLocal` 读取。因此：

```text
gLast = original_g_last * ln2
gLastExp = exp(gLast) = exp2(original_g_last)
```

### 5.2 Stage：`CalcDv2`

不再增加乘法。该阶段复用 `CalcGatedQ` 中已经转换的 `gCastLocal` 和 `gLast`：

```text
bdv *= exp(gLast - gCastLocal)
     = exp((original_g_last - original_g) * ln2)
     = exp2(original_g_last - original_g)
```

这里不得再次执行 `*ln2`。

### 5.3 Stage：`UpdateDh` 的 scalar Gate

直接复用：

```text
gLastExp = exp2(original_g_last)
bdh *= gLastExp
```

不新增乘法。

### 5.4 Stage：`UpdateDh` 的 key-wise Gate

对存在的 `gk`，路径改为：

```text
GM gk_last
  -> CopyIn FP32
  -> Muls(LN2)
  -> Exp
  -> 沿 V 广播
  -> bdh * gate
```

伪代码：

```cpp
if (hasGk) {
    CopyIn(gkLocal, gkGm[gkLastOffset], halfK);
    Muls(gkLocal, gkLocal, LN2, halfK);
    Exp(gkLocal, gkLocal, halfK);
    BroadcastAndMulBdh(gkLocal, bdhLocal);
}
```

必须保持第一阶段已经修复的 UB 生命周期顺序：

```text
1. gk CopyIn
2. gk * ln2
3. Exp
4. Broadcast 到不与 bdh 冲突的区域
5. 再 CopyIn bdh
6. bdh * gate
```

`gCastLocal` 与 `bdhCastLocal` 复用 UB 起始区域，若先搬入 bdh 再读取 gk，会重新产生 K 行 `0/halfK`、
V 前 64 列的聚集覆盖错误。

### 5.5 AIC

AIC 不读取 `g/gk`，无需修改。Cube matmul、workspace、AIC/AIV flag 和 stage 顺序保持第一阶段实现。

## 6. Wrapper 修改

### 6.1 ctypes 稳定入口

删除全量转换：

```python
if use_exp2:
    g = g.float() * ln2
    gK = gK * ln2
```

保留 `gK` 的 FP32 规范化：

```python
gK = gK.float() if gK is not None else None
```

`g` 保留输入 dtype，不再因为 `use_exp2=True` 被 wrapper 强制转换为 FP32。

### 6.2 legacy wrapper

删除：

```cpp
g_.to(at::kFloat) * LN2
gKKernel * LN2
```

继续将 `gk` 规范化为 FP32，然后直接调用原 aclnn。

### 6.3 raw aclnn

原接口直接进入统一 `exp2` Kernel。调用方必须传入 log2 域 Gate。

## 7. dtype 与精度

### 7.1 `g` 为 FP16/BF16

```text
输入低精度 Gate -> Kernel Cast FP32 -> FP32 *ln2 -> Exp
```

不会创建完整 FP32 GM 临时 tensor。乘法和指数都在 UB FP32 上执行。

### 7.2 `g/gk` 为 FP32

```text
CopyIn FP32 -> FP32 *ln2 -> Exp
```

不允许将 Gate 降精度后再转换。

### 7.3 `g=None/gk=None`

不存在的 Gate 路径不执行 CopyIn、`Muls` 或 `Exp`：

```text
g=None  -> scalar Gate 恒等
gk=None -> key-wise Gate 恒等
```

## 8. 性能预期

相对第一阶段 `use_exp2=True` 路径：

- 消除 wrapper 产生的全量 `Mul_*` Kernel。
- 低精度 `g` 消除 wrapper 产生的全量 `Cast_*` Kernel。
- 消除完整 Gate FP32 临时 tensor及其 GM 读写。
- bwd_dhu 主 Kernel 的 AIV 增加 tile 级 `Muls`。
- 主 Kernel duration 可能上升，但公开 API 的设备侧总 duration 和峰值显存预计下降。

相对第一阶段 `use_exp2=False` 路径：

- 主 Kernel 新增 tile 级 `Muls(LN2)`。
- 数学结果由自然指数域改为 `exp2` 域，不能将差异判断为精度回归。

性能报告必须分别列出：

```text
主 Kernel duration
外部 Cast/Mul Kernel 数量与 duration
公开 API 设备侧总 duration
临时显存峰值
```

## 9. 验证方案

### 9.1 精度标杆

所有标杆统一采用：

```text
gate(x) = exp2(x)
```

不再保留自然指数域标杆作为功能预期。覆盖：

- `g=None/gk=None`。
- `g!=None/gk=None`。
- `g=None/gk!=None`。
- `g!=None/gk!=None`。
- `use_exp2=False/True`，两者输出应一致。
- `g` 为 q dtype/FP32，`gk` 为 FP32。
- FP16/BF16、dense/varlen、整 chunk/tail chunk。
- `K=64/128`、`V=64/128/256`、`chunk_size=64/128`。
- `Hk=Hv` 和 GVA `Hv>Hk`。

重点检查：

- `use_exp2=False` 与 `use_exp2=True` 在相同输入下结果一致。
- `dh/dv2` 通过 FP64/NPU-aligned 双标杆。
- `g=None,gk!=None` 的 `gk` 只在 `k @ bdh` 之后应用。
- tail chunk 的 `g_last/gk_last` 地址正确。
- K 行 `0/halfK`、V 前 64 列无 UB 覆盖型聚集误差。

### 9.2 接口回归

- 旧 Python 调用代码无需修改即可运行。
- legacy schema、参数顺序和默认值不变。
- raw aclnn 二进制签名不变。
- 文档明确 raw aclnn Gate 已统一为 log2 域。

### 9.3 性能

使用 `msopprof` 对比第一阶段 `use_exp2=True` 与第二阶段统一语义路径。不能用 Python wall time 作为结论。

## 10. 文档同步范围

实施时必须同步：

- 算子 README 的 Gate 数值域说明。
- raw aclnn 文档。
- Python API 中 `use_exp2` 的兼容保留语义。
- legacy API 文档。
- 精度标杆和测试用例。
- profiling 命令与性能对比口径。

不得继续描述：

```text
use_exp2=False -> exp
use_exp2=True  -> exp2
```

统一改为：

```text
所有入口 -> exp2
use_exp2  -> 兼容保留参数，不再切换 Gate 数值域
```

## 11. 风险与实施确认

- 该方案不修改函数签名，但改变了 raw aclnn 和公开 Python API 的既有默认行为。
- 依赖自然指数域 Gate 的现有调用会产生不同结果。
- `use_exp2=False` 不再具有历史语义，属于参数语义变化。
- 已按仓库兼容性红线获得 `@weinachuan` 针对本次语义变更的明确确认，可以同步修改 Kernel、wrapper、
  README 对外语义和测试预期。

## 12. 结论

第二阶段统一采用：

```text
Gate 输入域：log2
Kernel Gate：exp2
实现方式：AIV FP32 tile 固定 *ln2 后执行 Exp
use_exp2：兼容保留，不参与分支
TilingKey：保持 3 个
TilingData：不增加 useExp2/gateScale
AIC：不变
```

该设计最小化了 Kernel/Host 结构变化，并消除第一阶段 wrapper 的全量 Gate 预处理；代价是改变已有自然指数域
行为。本文记录第二阶段已经确认并据此实施的设计决策。
