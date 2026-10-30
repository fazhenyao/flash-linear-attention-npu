# Recurrent KDA 算子方案设计

方案设计规则版本：`V2`

文档状态：A5 目标方案。

修订日期：2026-10-29。开发流程按[03-方案设计](../../../../../../docs/agents/03-方案设计.md)，接口以[api.md](api.md)为准。

## 1. 目标

### 1.1 算子职责

Recurrent KDA 逐 token 更新 State 并输出结果，覆盖 issue #680 的三项要求：

- 核内融合 Q/K L2Norm、beta sigmoid 和 column gate 预处理。
- 支持 causal_conv1d 产生的合法非连续 Q/K/V view。
- 按 tiling 参数 `X` 指定容量的 tile 遍历序列，取消总 token 数 `[1,8]` 限制。

```text
Python/ACLNN校验及必要的连续化、State初始化
  → tiling → 单个AIV kernel：读取输入 → 预处理 → State递推 → 写回State/out
```

主方案：`tile → token → V行块（R=4）`；State 使用两个 bank 交替递推，另一个 bank 在可复用时预取下一任务初态，具体 bank 按当时的占用和读写完成状态分配，输入/out 双缓冲。每个 State bank 按 FP32 容量预留。每块四行的衰减、预测、更新和输出在寄存器中计算，复用当前 token 的预处理系数；out 在寄存器中收集整个 token 的结果后转换并写入 UB，整个 tile 完成后批量搬出。UB 仅保存输入、跨 token State 和待搬出结果。算法 workspace 为0，system workspace 另计。

面向 A5，维度规格为 `K=V=128`，保留固定长度 rank4、变长 packed rank3、FP16 辅助输入、两种 State dtype/layout 和原位/非原位接口。定长或变长只由是否传入 `cu_seqlens` 决定，不读取 `layout` 参数。本方案的接口语义调整及共用 host 规则同时适用于 A2/A3/A5；寄存器、bank 和 tile 流水优化面向 A5，对 A2/A3 的影响见第1.4节。

### 1.2 设计依据

- 需求：[issue #680](https://github.com/flashserve/flash-linear-attention-npu/issues/680)。
- 结构参考：[design_full.md](/C:/Users/Administrator/Downloads/0894ddf210974f0d9b955bcc25caa5cd/design_full.md)；寄存器计算参考：[regebase.md](/C:/Users/Administrator/Downloads/0894ddf210974f0d9b955bcc25caa5cd/regebase.md)。
- 接口：[api.md](api.md)；CPU 标杆：[recurrent_kda_reference](../../../../../../tests/atk/recurrent_kda/executor_recurrent_kda.py)。
- 实现：[host](../op_host/)、[kernel](../op_kernel/arch35/)；验证：[ATK](../../../../../../tests/atk/recurrent_kda/)、[validation.md](validation.md)。

沿用参考方案的单 Vector Stage、三个 State bank 的容量和 tile 完成后批量搬出 out；bank 调度采用两个 bank 交替递推、另一个可复用 bank 预取下一任务初态，递推和预取角色按当前 bank 状态分配，token 窗口容量改由 tiling 参数 `X` 指定。KDA 将 scalar gate 改为 K 维 column gate，并保留本接口的 State 选择与写回语义。参考方案中的固定 256 head 上限不属于本方案约束，head 数改由本方案第 1.3 节的表示范围、溢出检查和设备容量共同决定。

### 1.3 完整接口语义

本表及以下约束描述 A5 的目标支持范围，其中序列模式、gate 判定和参数组合语义由 A2/A3/A5 共用；硬件相关的支持范围与实现资源按第1.4节区分。Q/K/V 输入采用定长 BSND（`[B,S,N,D]`）或变长 TND（`[T,N,D]`），其中 N 为对应的 head 数，D 为 K 或 V。`q/k/v` 对应 ACLNN 的 query/key/value；是否传入 `cu_seqlens` 决定输入格式，`BlockNum` 为 State pool 槽容量。

| 参数 | shape | dtype | 语义 |
| --- | --- | --- | --- |
| `q` / query | 无 `cu_seqlens`：`[B,S,Nk,K]`；有 `cu_seqlens`：`[T,Nk,K]` | BF16 | query；合法非连续输入按各轴 stride 直读，条件见第2.5节 |
| `k` / key | 与 query 相同 | BF16 | key；stride 独立于 query |
| `v` / value | 无 `cu_seqlens`：`[B,S,Nv,V]`；有 `cu_seqlens`：`[T,Nv,V]` | BF16 | value；每次搬入当前 head 的 tile |
| `initial_state` / State pool | V-first `[BlockNum,Nv,V,K]`；K-first `[BlockNum,Nv,K,V]` | BF16/FP32 | 原位模式读写参数；Python 非原位模式可省略，由 wrapper 创建全零 FP32 State；ACLNN 必选 |
| `beta` | 无 `cu_seqlens`：`[B,S,Nv]`；有 `cu_seqlens`：`[T,Nv]` | BF16/FP16/FP32 | delta 强度；可解释为 sigmoid 前的 raw 值 |
| `cu_seqlens`，可选 | `[B+1]` | INT32/INT64 | 传入时启用变长 packed 模式；累积 token offset，首项0，允许空序列 |
| `ssm_state_indices`，可选 | packed `[>=T]`；speculative `[B,max_step]` | INT32/INT64 | token/checkpoint 到 State pool 槽号的映射 |
| `g` | 无 `cu_seqlens`：`[B,S,Nv,K]`；有 `cu_seqlens`：`[T,Nv,K]` | BF16/FP16/FP32 | 必选 column gate；step log gate 或 raw gate；没有独立 `gk` 参数 |
| `A_log`，可选 | `[Nv]` | FP32 | 传入时启用 raw gate，作为 head-wise 参数按 `hv` 取值；未传入时使用 step log gate |
| `dt_bias`，可选 | `[Nv,K]` 或 `[Nv*K]` | FP32 | raw gate 的 column bias；未传入 `A_log` 时忽略 |
| `num_accepted_tokens`，可选 | `[B]` | INT32/INT64 | 与 indices 配合选择初始 State 槽；不截短本次有效序列 |
| `layout`，兼容参数 | string attr | string | 方案忽略该参数；不参与定长/变长判定，也不改变 shape 或 stride 解释 |
| `scale`，可选 | scalar attr | float | query 输出缩放；Python 缺省为 `K**-0.5`，ACLNN 由调用方传入 |
| `output_final_state`，可选 | bool attr | bool | 默认 `false`；控制 Python 返回项中的 State 是否为 None |
| `inplace_final_state`，可选 | bool attr | bool | 默认 `true`；是否更新输入 State pool |
| `use_qk_l2norm_in_kernel`，可选 | bool attr | bool | 默认 `false`；FP32 Q/K L2Norm；对应 issue 的 `use_l2_norm_in_kernel` 功能 |
| `use_gate_in_kernel`，兼容参数 | bool attr | bool | 保留旧 ABI；数值路径忽略其值，由 `A_log` 是否传入决定 `g` 是 raw column gate 还是 step log gate |
| `use_beta_sigmoid_in_kernel`，可选 | bool attr | bool | 默认 `false`；是否使用 `sigmoid(beta)` |
| `allow_neg_eigval`，可选 | bool attr | bool | 默认 `false`；beta sigmoid 开启后是否再乘2 |
| `safe_gate`，可选 | bool attr | bool | 默认 `false`；传入 `A_log` 时控制 raw gate 的 safe 分支，否则忽略 |
| `lower_bound`，可选 | scalar attr | float | 缺省按 `-5.0`；仅传入 `A_log` 且 `safe_gate=true` 时使用并校验 `[-5,0)` |
| `state_v_first`，可选 | bool attr | bool | 默认 `false`；true 为 V-first，false 为 K-first |
| `out` | 与 value 相同 | BF16 | 每个有效 token 的 recurrent 输出 |
| `final_state` | 与 initial State 同 shape | 与 initial State 相同 | 原位模式 alias 输入，非原位模式为独立结果；未发布槽保持初值 |

约束与组合规则：

1. `Nk,Nv` 为正，`Nv%Nk==0`；删除旧方案中的 `Nk<=256`、`Nv<=256` 静态限制，不再以 256 作为 head 数上限。实际可支持范围由 `uint32` tiling 字段、GM stride、task 数、State pool 大小和设备可用内存共同决定；所有乘法、加法和字节数换算必须先在 `uint64` 中检查溢出，再转换为设备字段。当前设计仅支持 `K=V=128`。
2. 未传入 `cu_seqlens` 时，q/k/v/g/beta 必须为固定长度 rank4，shape 的 batch 和长度维一致，`T=B*S`；传入 `cu_seqlens` 时，q/k/v/g/beta 必须为 packed rank3，`cu_seqlens.shape=[B+1]`，`cu_seqlens[0]=0`、单调不减且末项不超过 packed token 容量。`layout` 不参与上述判断。
3. `g` 必选；`A_log` 传入时进入 raw gate 路径，`dt_bias` 可选，并按 `safe_gate` 选择 safe/unsafe 分支；`A_log` 不传入时 `g` 解释为 step log gate，忽略 `dt_bias`、`safe_gate` 及该分支的 `lower_bound`，不因这些参数的存在或取值触发 raw gate 组合或范围校验。host/wrapper 在相关校验和连续化前将未启用 raw gate 时的 `dt_bias` 视为空，不校验其 raw gate 专用 shape/dtype、不搬运也不读取；仅在传入 `A_log` 且 `safe_gate=true` 时校验 `lower_bound`。`use_gate_in_kernel` 仅作为兼容参数，不参与数值分支。迁移期 host 可检查旧 flag 与 `A_log` 是否一致并拒绝矛盾调用，kernel 不使用该 flag。Q/K L2Norm、beta sigmoid 和 raw-gate safe 分支的开关保持独立。
4. indices 为空时 `BlockNum=B`；否则有效槽号须在 `[0,BlockNum)`，不同活跃请求不得写同一槽。accepted 仅选择初态，必须与 indices 配合。
5. 变长模式的空序列跳过；`cu_seqlens` 末项不超过输入容量，padding tail 的 out 不保证。固定长度模式不使用 `cu_seqlens`，每个 batch 序列长度均为 `S`。窗口容量 `X` 不限制总长，地址仍受 tiling 容量和设备内存约束。
6. 原位更新或要求返回 State 时执行写回：有 indices 则逐 token 写快照，无 indices 则只写末态。Python 返回 `(out, final_state)`，`output_final_state=false` 时第二项为 None，原位更新仍按属性执行。
7. Q/K/V 合法非连续 view 按第2.5节直读或连续化；State 只允许矩阵外层间隔。非原位模式保留未写槽初值。

### 1.4 A2/A3 共用语义及影响

本次接口语义调整在 A2/A3/A5 上统一生效。Python wrapper、stable ABI/op_plugin、ACLNN、host tiling 和 kernel 使用一致的模式判定、参数归一化和组合校验规则，无需为 A2/A3 保留重构前的参数解释或校验分支。

| 项目 | 共用规则及 A2/A3 影响 |
| --- | --- |
| 序列模式 | 仅由 `cu_seqlens` 是否传入决定，忽略 `layout`；A2/A3 同步采用新的 shape 判定和组合校验 |
| gate 模式 | 由 `A_log` 是否传入决定，数值路径忽略 `use_gate_in_kernel`；A2/A3 通用 kernel 同步采用该判定 |
| 未传入 `A_log` | 忽略 `dt_bias`、`safe_gate` 和 `lower_bound` 的 raw gate 专用处理；各层均跳过相应校验、连续化、搬运和读取 |
| host 和 tiling | 共用的 head/stride 校验、连续化判定和预算辅助函数可统一修改；这些修改对 A2/A3 的支持范围、`vStep` 和缓冲配置的影响纳入回归 |
| 硬件实现 | A5 使用 arch35 的 R=4、寄存器 Cast、动态 bank 角色和 tile 流水；A2/A3 使用各自 kernel 路径，并遵循上述共用接口语义 |

host 仍按实际平台选择 kernel 及硬件资源预算。A5 的 `K=V=128`、`X<=8` 选型和第2.4节 UB 布局仅约束 A5 路径；A2/A3 的硬件支持范围和资源配置按对应路径确定。回归应验证各平台的新接口语义、精度及资源合法性，并检查共用代码变更造成的性能影响；有意调整的参数解释和校验按本方案验收。


## 2. Stage 0 完整详设

### 2.1 全局符号和任务域

符号表：

| 符号 | 含义 |
| --- | --- |
| `T` | token容量；无 `cu_seqlens` 时为 `B*S`，有 `cu_seqlens` 时为 packed 输入第0维容量 |
| `S` | 单条序列长度；定长模式各请求相同，变长模式按当前请求的有效 token 数计 |
| `Lmax` | 本次调用中所有请求的最大有效序列长度；定长为S，变长为相邻cu_seqlens差分的最大值；用于选择共用X |
| `B` | 请求数；无 `cu_seqlens` 时为固定长度输入第0维，有 `cu_seqlens` 时为 `numel()-1`，可含空序列 |
| `Nk/Nv` | query-key/value head数；不参与 UB 按 head 扩容，受 tiling、GM stride、task 数和设备内存可表示范围约束 |
| `K/V` | key/value维度；当前设计均为128 |
| `G` | Nv/Nk，hk=floor(hv/G) |
| `X` | token 窗口容量，从`{1,2,4,8}`选择，最大不超过8，由host tiling传入kernel；不是请求总长上限 |
| `hv` | value head id |
| `hk` | query-key head id，floor(hv/G) |
| `State` | 单个head的[V,K]数学State；寄存器计算为FP32，UB保持接口dtype和layout；无整份State寄存器常驻假设 |

每个请求的一个 value head 为独立任务，共 `B*Nv` 个；`Nv` 增大只增加任务数量和 GM 访问跨度，不增加单任务 UB 容量。任务按请求、head 顺序循环分给 AIV 核，启动核数取可用核数与任务数的较小值。任务内按 `tile → token → V行块（R=4）` 计算，同一 State 的 token 串行，无核间同步。head 数超过单轮 AIV 数时继续采用 grid-stride 轮转。

例如32条请求、每条8个token、Nk=16、Nv=48：共1536个任务；56个核分别处理27或28个任务，共28轮，末轮24个任务。G=3，hv=5对应hk=1。每个任务共 `ceil(8/X)` 个tile；示例取 `X=4` 时为2个tile。空序列直接跳过。

`X` 是内部 tiling 参数，不新增 Python/ACLNN 公开参数，候选集合为 `{1,2,4,8}`，硬上限为8。本次调用所有请求/head 共用一个 X，host 根据最大有效序列长度 `Lmax` 计算截断到8的选型长度，再结合实际可用 UB、DMA/VF 约束及精度与性能实测，从不超过选型长度的合法候选中选择 X：

```text
Lmax = S                                           # 定长
Lmax = max_b(cu_seqlens[b+1] - cu_seqlens[b])         # 变长
selection_length = min(max(Lmax, 1), 8)
X in {1,2,4,8}, X <= selection_length              # 还须满足UB与DMA/VF约束
```

变长模式使用各请求有效长度的最大值，不使用 packed 总容量 T、有效 token 总数或平均长度代替。host 选型前需获得实际 `Lmax`；从设备元数据获取或统计最大长度产生的开销计入调用链评估。`Lmax>8` 时 X 仍不超过8，长序列通过多个 tile 处理。全空序列时 `Lmax=0`，选型长度为1，在满足容量约束时取 `X=1`，kernel 跳过全部空任务。

选定 X 通过 tiling data 传入 kernel，kernel 按该值分配输入/out槽、计算槽偏移和推进窗口；每个请求仍按自己的有效长度执行，`validTokens=min(X, end-tileStart)`，短请求允许不足一个完整窗口。`X=4` 仅用于本文资源预算示例。host 必须校验 `X∈{1,2,4,8}`、`1<=X<=8`、字段可表示范围及第2.4节的容量条件，相关容量和偏移先在 `uint64` 中检查溢出。

### 2.2 Stage 结果

唯一 Stage 定义如下：

| Stage | 执行单元 | 功能 | 主要空间 |
| --- | --- | --- | --- |
| `Stage 0：Vector，逐 token 完成 Recurrent KDA` | AIV | 元数据、三项预处理、State递推、out/State写回 | State bank按FP32容量预留，UB随X变化（第2.4节）；X=4示例为207.3125 KiB；L1/L0与算法workspace均0 B |

Main Scalar负责元数据、DMA和bank调度；目标每个tile调用一次VF，完成全部数值计算。

### 2.3 Stage 0：Vector，逐 token Recurrent KDA 公式

对请求 `b`、value head `hv` 和展平 token 下标 `t`，令 `hk=floor(hv/G)`；t 位于当前请求的有效 token 范围。所有中间计算使用 FP32；State 在 GM 和 UB 中始终保持接口 dtype 和 layout，V-first/K-first 均按四个 V 行读入寄存器并完成相同的递推，不改变下述 `[V,K]` 数学 State。两种布局仅在 UB 行访问方式上不同，K-first 的跨距读写见第2.3.5节。

#### 2.3.1 Q/K 预处理

下面 j 是 K 维下标，范围为0到K-1，sum_j 表示沿 K 维求和。

```text
q_hat = FP32(query[t,hk,:])
k_hat = FP32(key[t,hk,:])
if use_qk_l2norm_in_kernel:
    q_hat /= sqrt(sum_j(q_hat[j]^2) + 1e-6)
    k_hat /= sqrt(sum_j(k_hat[j]^2) + 1e-6)
q_scaled = scale * q_hat
```

归一化、后续系数与 State 计算均保持 FP32；scale 只乘 query 一次。

#### 2.3.2 Column gate 预处理

`g` 是必选的 K 维 column gate；下面 j 为 K 列下标（0到K-1）。

先根据 `A_log` 是否传入确定有效参数：

```text
raw_gate = (A_log is not None)
effective_dt_bias = dt_bias if raw_gate else None
effective_safe_gate = raw_gate and safe_gate
```

若未传入 `A_log`（预计算 step log gate）：

```text
g_eff[j]   = FP32(g[t,hv,j])
alpha_k[j] = exp(g_eff[j])
```

此分支不读取 `dt_bias`，不使用 `safe_gate` 或 `lower_bound`，仅计算上述 `exp(g)`。

若传入 `A_log`（raw gate），先计算：

```text
x[j] = FP32(g[t,hv,j]) + (FP32(effective_dt_bias[hv,j]) if present else 0)
a    = exp(FP32(A_log[hv]))
```

按 `effective_safe_gate` 选择 unsafe 或 safe 分支：

```text
g_eff[j] = -a * (max(x[j],0) + log(1 + exp(-abs(x[j]))))  # unsafe
g_eff[j] = lower_bound * sigmoid(a*x[j])                 # safe
alpha_k[j] = exp(g_eff[j])
```

unsafe 使用稳定 softplus；safe 的 lower_bound 为 log decay 下界。`alpha_k` 沿 V 行广播。

#### 2.3.3 Beta 预处理

```text
beta_eff = sigmoid(FP32(beta[t,hv]))   if use_beta_sigmoid_in_kernel
           FP32(beta[t,hv])            otherwise
if use_beta_sigmoid_in_kernel and allow_neg_eigval:
    beta_eff = 2 * beta_eff
```

sigmoid 使用稳定实现；非有限值传播按既有精度规则验证。

#### 2.3.4 Recurrent update

令 `State_prev` 为 `[V,K]`，用 V 行 `i` 和 K 列 `j` 表示完整数学公式；i、j 的范围均为0到127：

```text
State_bar[i,j]    = State_prev[i,j] * alpha_k[j]
prediction[i] = sum_j State_bar[i,j] * k_hat[j]
delta[i]      = beta_eff * (FP32(value[t,hv,i]) - prediction[i])
State_new[i,j]    = State_bar[i,j] + delta[i] * k_hat[j]
out_fp32[i]   = sum_j State_new[i,j] * q_scaled[j]
out[t,hv,i] = cast_BF16_RINT(out_fp32[i])
```

每次处理四个相邻 V 行（`R=4`），交错执行四行各自的 decay、prediction、delta、update、out，复用当前 token 的 q/k、alpha_k 和 beta_eff。各行的独立 Mul/Add 和归约次序保持不变，中间结果不落 UB；更新后的 State 按接口 dtype 写入 next bank，下一 token 从该 bank 读取。

两种 State layout 使用相同的 dtype 处理：BF16 从 UB 读入寄存器后立即转为 FP32，全部计算在 FP32 寄存器中完成；out 使用未降精度的更新结果，State 在写回 UB 前于寄存器内按 RINT 转回 BF16。FP32 State 直接读入 FP32 寄存器并直接写回。UB 中仅保留接口 dtype 的跨 token State，不设置完整 FP32 转换副本。

128 个 V 行分为前后两个64行半区，每个半区循环16个四行块。每块的四个 FP32 out 归约结果通过三次 Interleave、一次掩码 Move 和一次寄存器 Gather 收集到对应的64元素输出累加器；Gather 每次旋转四个 lane，16块完成后得到自然顺序的半区结果。两个输出累加器保持 FP32，完整 token 的128行全部完成后，分别在寄存器内转换为 BF16，并通过两次 Store 写入本 tile 的 UB 输出槽。out 始终使用尚未降为 State 接口 dtype 的 FP32 更新结果计算；整个 tile 的 VF 完成后才批量提交 out 的 MTE3。

初始 State 选择与快照写回：

```text
start, end = 当前请求在展平输入中的起止位置           # 左闭右开 [start,end)
write_slot(t) = b                               # no indices
                indices[t]                     # packed indices
                indices[b,t-start]             # speculative indices
initial_slot  = b                               # no indices
                write_slot(start)              # indices without accepted
                write_slot(start+accepted[b]-1) # indices with accepted
需要写回State = (inplace_final_state or output_final_state)
               and (has_indices or t == end-1)
```

每个 token 更新后，在寄存器内将 FP32 `State_new` 转换到接口 dtype，再按接口 layout 写入 next bank，供后续递推及需要时的 MTE3 搬出。发布到 GM 的元素对应关系为：

```text
state_dst[write_slot(t),hv,i,j] = cast_state_dtype(State_new[i,j]) # V-first
state_dst[write_slot(t),hv,j,i] = cast_state_dtype(State_new[i,j]) # K-first
```

BF16 使用 RINT，FP32 不降精度；重复槽按 token 顺序覆盖。

#### 2.3.5 K-first 的 V 行寄存器读写

`state_v_first=false` 时，GM 初态、UB 跨 token State、快照和末态始终保持 K-first `[K,V]`。按四个相邻 V 行读入寄存器，复用第2.3.4节的 R=4 计算和 out 收集方式，再按 K-first 地址写回 next bank；不执行独立的初态、快照或末态转置。

UB 内矩阵紧密存放，第 `i` 个 V 行、第 `j` 个 K 元素的偏移以 State 元素为单位：

```text
V-first：offset(i,j) = i*K + j      # 连续行 Load/Store
K-first：offset(i,j) = j*V + i      # 按 V 行 Gather/Scatter

UB cur[K,V] -> Gather 四个 V 行到寄存器
  -> BF16 转 FP32；FP32 直接使用
  -> FP32 decay / prediction / delta / update / out
  -> State 写 UB 前：BF16 按 RINT 转回；FP32 直接使用
  -> 按 K-first 地址写入 UB next[K,V]
  -> 需要发布时由 MTE3 直接搬出
```

BF16 的相邻两个 V 元素共享一个32位字，读入时可成对收集，在寄存器内解包后分别转为 FP32；写回时先分别完成 FP32→BF16 RINT，再将相邻两个 BF16 的位模式打包为32位字写回。R=4 对应两对，打包不增加数值转换或舍入。FP32 按元素 Gather/Scatter，dtype 处理与 V-first 一致。

每个 token 对 State 的逻辑 UB 访问为每元素读一次、写一次，衰减、预测和更新的中间结果均留在寄存器中。完整 token 的 State 写入完成并发布 ready 后，有 indices 时直接从更新结果所在 bank 搬出快照；无 indices 时仅在末 token 搬出末态。覆盖该 bank 前等待相应 `MTE3_DONE`，无需额外快照布局副本或 GM 回读。

下一任务初态按接口布局预取到当时可复用的 bank；等待 DMA 完成后可在相应任务中按上述方式读入计算。bank 分配仍按第2.4.1节的状态与生命周期约束执行，不增加 UB 转换缓冲或算法 workspace。Gather/Scatter、BF16 成对打包和寄存器峰值在实现后单独验证。

### 2.4 A5 UB 空间布局

A5单AIV可用UB按248 KiB=253952 B预算，依据历史Ascend950PR设备记录；host实施时需核对实际设备。所有区间为绝对半开区间、32 B对齐，1 KiB=1024 B，表内KiB为精确值。两个dtype模板互斥，不累加。

K=V=128，X由tiling指定。每个State bank按完整FP32 State容量预留，实际内容保持接口dtype和layout。BF16初态、递推结果和快照在UB中均只保存BF16；加载后及写回前的Cast在寄存器内完成。K-first按V行Gather/Scatter，不增加独立转换或快照布局缓冲。Q/K/V/g/beta/out 按 tile 双缓冲，g/beta 按最大4 B元素预留；A_log/dt_bias 当前任务与后继任务各一槽。

空间按 `X` 计算：qIn/kIn/vIn/out 各为 `512*X B`，gIn 为 `1024*X B`。betaIn 使用多行短行 DMA，每个 token 的 beta 在 UB 中占一个32 B行间隔，加载步长为 `32/sizeof(beta_dtype)` 个元素；单槽预留 `beta_slot_bytes(X)=max(128,32*X) B`，双槽总预留 `R_beta(X)=2*max(128,32*X) B`。三个 State bank 共196608 B，A_log/dt_bias 双槽共1088 B，guard reserve 为2048 B。因此总峰值为：

```text
UB_bytes(X) = 196608 + 3072*X + R_beta(X) + 1088 + 2048
            = 199744 + 3072*X + R_beta(X)
UB_bytes(X) <= 实际可用UB字节数
```

地址按下表相同顺序，以各区实际大小递增排列并保持32 B对齐；输入/out双槽各按X计算大小，beta双槽等分预留。总空闲与最大连续空闲均为实际可用UB减去 `UB_bytes(X)`。下表仅代入 `X=4` 展示绝对地址，其他X须重新计算大小和偏移。

原始输入由MTE2写、VF读；结果由VF写、MTE3读。槽在最后一次读取完成后复用；gate参数在当前任务的全部tile间复用。UB不设置FP32预处理或数值中间区。

#### 2.4.1 FP32 State

每个bank容纳当前head的完整State，三个均64 KiB；FP32 State实际使用64 KiB，BF16 State实际使用32 KiB。初态、递推结果、快照及末态始终保持本次调用的接口dtype和layout。bank的用途按当时状态和生命周期分配，不以bank编号或任务固定。下表为 `X=4` 示例的同时存活峰值：

| UB 区间 | 大小 | 对象 | 生命周期 |
| --- | ---: | --- | --- |
| `[0,64 KiB)` | 64 KiB | `stateBank0 FP32` | 按当前状态承载递推、State 搬出或初态预取；旧内容消费完毕后复用 |
| `[64,128 KiB)` | 64 KiB | `stateBank1 FP32` | 按当前状态承载递推、State 搬出或初态预取；旧内容消费完毕后复用 |
| `[128,192 KiB)` | 64 KiB | `stateBank2 FP32` | 按当前状态承载递推、State 搬出或初态预取；旧内容消费完毕后复用 |
| `[192,194 KiB)` | 2 KiB | `qIn[2][X][128] BF16` | 双缓冲 tile |
| `[194,196 KiB)` | 2 KiB | `kIn[2][X][128] BF16` | 双缓冲 tile |
| `[196,198 KiB)` | 2 KiB | `vIn[2][X][128] BF16` | 双缓冲 tile |
| `[198,202 KiB)` | 4 KiB | `gIn[2][X][128] 原dtype reserve` | column gate 输入 tile |
| `[202,202.25 KiB)` | 0.25 KiB | `betaIn[2][X] 原dtype reserve` | 每 token 按32 B间隔存放，单槽至少128 B |
| `[202.25,203.3125 KiB)` | 1.0625 KiB | A_log / dt_bias（双槽），FP32 | per-head 参数 reserve |
| `[203.3125,205.3125 KiB)` | 2 KiB | `out[2][X][128] BF16` | 双缓冲 MTE3 |
| `[205.3125,207.3125 KiB)` | 2 KiB | `guard reserve` | 对齐、事件和实现安全边界；无 tensor |
| `[207.3125,248 KiB)` | 40.6875 KiB | contiguous reserve | event 退化、对齐和扩展余量 |

`X=4` 时FP32峰值212288 B=207.3125 KiB，按248 KiB预算余量41664 B=40.6875 KiB；总空闲与最大连续空闲相同。

`X=8` 时 beta 双槽共512 B，FP32峰值224832 B=219.5625 KiB，按248 KiB预算余量29120 B=28.4375 KiB；后续参数、out 和 guard 区间按该槽大小重新计算。

三个物理 bank 的递推和预取用途按当时状态分配。`cur` 指保存当前有效递推 State 的 bank，`next` 指本次更新的目标 bank；下一任务的初态预取使用另一个满足复用条件的 bank。A/B/C 仅为这些角色的示意名称，具体编号及后续用途由当前 State 是否仍有效、VF 是否访问、MTE2/MTE3 是否完成以及是否已存入待消费的预取初态决定。

调度保持两个 bank 交替递推、另一个 bank 承载下一任务初态的结构。选择 `next` 或预取目标前，必须确认旧 State 已完成最后一次消费、VF 访问及旧 DMA 已完成；已预取且尚未被对应任务消费的初态也必须保留。两种 layout 的逐 token 快照均直接使用更新结果所在 bank，覆盖快照源 bank 前等待对应 `MTE3_DONE`。

任务交接时，实际存放下一任务预取初态的 bank 成为其初始 `cur`，其余 bank 根据当时状态承担更新目标或后续预取用途。若某个 bank 仍被末态 MTE3 读取，则保留到完成后再评估复用；暂时没有可用预取目标时，推迟预取，待 bank 释放后重新选择。三个 bank 的容量与 UB 峰值不变。

### 2.5 Stage 0 流程和流水

1. MTE2 按接口 dtype 和 layout 搬入当前 head 的完整初始 State。`A_log` 已传入时读取 A_log/dt_bias，任务内复用。
2. MTE2 搬入本 tile 的 Q/K/V/g/beta。等待 INPUT_READY 后，若当前任务仍有下一 tile，则向另一输入槽预取下一 tile；若当前 tile 是任务最后一个 tile 且存在下一任务，则向另一输入槽预取下一任务的首个 tile 输入。
   存在下一任务时，在当前任务首个 tile 的输入搬完并提交后继 tile 输入后，按 bank 当前状态选择可复用的预取目标，装入下一任务初态，并向另一参数槽预取对应参数；若暂时没有可用目标，则先启动当前 tile 的 VF，在相关旧 MTE3 完成、bank 释放后重新选择并提交 State 预取。
3. 一次 VF 完成本 tile，内部逐 token 递推：
   - Cast，按开关完成 Q/K L2Norm、gate 和 beta 预处理。
   - 每次处理四个相邻 V 行，完成 State 衰减、prediction、delta、State 更新和 out 计算，公式及寄存器收集方式见第2.3节。
   - State 从 UB 读入寄存器后按需转为 FP32，更新结果在写回 UB 前按需转换到接口 dtype，并按原 layout 写入 next；out 在两个64元素 FP32 寄存器中收集，完整 token 计算完成后转换并通过两次 Store 写入当前 tile 的输出槽。State 按接口条件直接从结果 bank 提交 MTE3，整个 tile 的 VF 完成后再批量提交 out 的 MTE3。
   - 更新结果所在 bank 成为新的 cur，按 bank 状态确定下一次更新的 next，继续下一 token；覆盖尚被 MTE3 读取的 bank 前等待完成，并保留尚未消费的预取初态。
4. 切换输入槽，处理下一 tile。任务交接时等待下一任务初态预取完成并重新分配 bank；允许未完成的末态 MTE3 与下一任务计算重叠，覆盖其源 bank 前等待完成。最后一个任务结束后排空剩余 MTE3。

单个请求/head任务的逻辑流程如下。tile 指 token 窗口，容量由 tiling 参数 X 指定；A_log/dt_bias 在任务内复用。

```text
1. 读取请求的有效token数；若为0，立即结束当前任务
2. 计算start/end（展平token起止位置）、accepted位置、initial_slot（初态槽号）
3. 若初态已预取，等待其STATE_READY并将实际所在bank设为cur；否则选择可复用bank搬入当前head的完整initial State并设为cur
   按其余bank的当前状态安排next和后续预取用途；有未完成VF/DMA或待消费State的bank须保留
   State始终保持接口layout；每次计算加载后在寄存器内将BF16转为FP32，FP32直接使用
4. 若已传入 `A_log`，读取当前 hv 的 A_log/dt_bias（任务内首次加载）
5. 读取tiling参数X；for tileStart in range(start, end, X):
     validTokens = min(X, end - tileStart)    # 尾窗口仅搬运和计算有效token
     5.1 MTE2一次搬入本窗口当前head所需的Q/K/V/beta/g
     5.2 等待本窗口INPUT_READY；当前任务存在下一窗口时，向另一输入槽预取下一窗口；当前窗口为任务尾窗口且存在下一任务时，向另一输入槽预取下一任务的首个窗口输入
         存在下一任务时，首窗口在上述输入提交后，选择可复用bank预取下一任务初态并向另一参数槽预取参数
         若无可用bank，则在5.3启动VF后等待相关旧MTE3完成，重新选择释放的bank并提交预取
     5.3 发起一次顶层VF调用，传入validTokens；VF内部for t in 当前窗口的有效token:
           a. Q/K cast；按开关做FP32 L2Norm，query乘scale
           b. 按开关计算beta_eff和column gate系数alpha_k
           c. 依次遍历两个64行半区，每次读入四个相邻V行：V-first连续加载，K-first跨距Gather
              BF16读入寄存器后转FP32，FP32直接使用；随后State按K列衰减
           d. 四行分别计算prediction = 衰减后的State @ k_hat
           e. 四行分别计算delta = beta_eff * (value - prediction)
              State_new = 衰减后的State + delta @ k_hat.T
           f. 四行分别计算out = State_new @ q_scaled，收集到当前半区的FP32输出寄存器
              完整token的128行全部完成后，将两个输出寄存器转为BF16并分别Store到当前tile的UB输出槽
           g. 每个四行块在寄存器内按需将更新后的State转为BF16，FP32直接使用，再按接口layout写入next
              V-first连续Store；K-first按原地址Scatter，BF16相邻V元素成对打包后写回
              按第2.3.4节取得write_slot；有indices时按token直接提交结果bank的State异步搬出，无indices时仅提交末态
           h. 更新结果所在bank成为cur，按当前状态选择next，继续下一token；覆盖前等待该bank的快照MTE3完成，保留待消费的预取初态；窗口结束后切换input/out槽
      5.4 当前tile的VF完成后，等待必要的V→S同步，将该tile的out一次批量提交MTE3
6. 存在下一任务时，交接前等待其STATE_READY，将预取初态实际所在bank作为下一任务初始cur，按其余bank状态安排next和后续预取用途
   当前末态仍被MTE3读取的bank保持到MTE3_DONE后，才可用于后续任务预取；全部任务结束后排空剩余MTE3
```

同一 State 的 token 串行；流水重叠来自：

- 当前 token 计算与下一 tile 输入 MTE2。
- 当前任务计算与按 bank 状态选出的预取目标中下一任务初态及参数的 MTE2 预取。
- 当前 token 计算与上一 token 的 State MTE3，以及当前 tile VF 与上一 tile 的 out MTE3。
- 不同 AIV 核处理不同请求/head。

State bank 和输出槽在 MTE3 读完后才可覆盖，输入槽在当前 tile 的 VF 最后一次读取完成后才可复用。任务尾 tile 的另一输入槽可在当前 tile VF 运行期间承载下一任务首个 tile 的 Q/K/V/g/beta；该输入预取与下一任务 State/参数预取分别受输入槽和 State bank 生命周期保护。BF16→FP32在UB加载后的寄存器内完成，FP32→BF16在Store到UB前的寄存器内完成，寄存器上计算时用FP32计算。尾 tile 只处理有效 token，空序列跳过。

Q/K/V 支持定长 BSND、变长 TND，包括 causal_conv1d 输出切片形成的合法非连续输入。非连续输入保持 D、N、S/T、B 从内到外的存储顺序，允许轴间有间隔；各输入使用独立的真实 stride，D 轴 stride 为1且满足DMA约束时直接搬运。定长 BSND 输入跨物理 batch 时拆段，变长 TND 输入按 `cu_seqlens` 的 token 区间拆段。支持范围内安全但不能直接搬运的输入先连续化；真实元素地址重叠、越界或溢出报错，无法确认安全则报不支持，zero-stride保留既有连续化路径。

### 2.6 数据搬运汇总

按单个非空任务计，S为有效token数；State元素为BF16时2 B、FP32时4 B，g/beta按实际dtype计。表中为逻辑读写量，DMA可按tile/行拆分：

| 数据 | GM访问次数与字节 | UB驻留/复用 | 批量搬运与L2策略 |
| --- | --- | --- | --- |
| Q | 直接路径每元素1次，S*K*2 B | qIn双槽；当前token系数跨V行复用 | 每窗口最多X token，独立q token/head/batch stride；共享hk依赖L2 |
| K | 直接路径每元素1次，S*K*2 B | kIn双槽 | 同Q，用独立k stride |
| V | 直接路径每元素1次，S*V*2 B | vIn双槽 | 每窗口最多X token，多行DMA；固定长度 rank4 跨物理batch拆段，变长 packed rank3 按序列区间拆段 |
| beta | 每元素1次，S*beta元素字节数 B | betaIn双槽，每token占32 B行间隔，单槽至少128 B | 按hv收集最多X行；默认cache |
| g | 每元素1次，S*K*g元素字节数 B | gIn双槽 | 每窗口最多X个token的column向量，不物化FP32系数 |
| A_log | 传入时1次，4 B | 当前work参数槽 | 按hv，默认cache |
| dt_bias | 传入 A_log 且存在时1次，K*4 B | 当前work参数槽，所有tile复用 | 不按token重复搬入 |
| initial State | 每元素1次，V*K*State元素字节数 B | 共用State bank，保持接口dtype和layout；读入寄存器后按需转FP32 | 按真实State stride搬入完整State；V-first连续加载、K-first按V行Gather |
| updated State | 有indices且需要写回时为S*V*K*State元素字节数 B；仅末态为V*K*State元素字节数；否则0 | 写UB前在寄存器内按需转为接口dtype，保持原layout；结果bank同时供递推和MTE3读取，覆盖前等待MTE3完成 | 快照逐token、末态仅一次，直接按slot/stride搬出；每token的State在UB中逻辑读一次、写一次，不回读GM递推 |
| out | 每元素写1次，S*V*2 B | 每token在两个FP32寄存器中收集完整结果，再以两次Store写入out双tile槽；MTE3读完后复用 | 每个tile的VF完成后批量提交一次，搬运`valid`行、每行V个元素，按out token stride写回 |
| cu_seqlens | 现有合法性扫描及每work边界项 | Main Scalar | 不每token重扫所有B |
| State indices | 初态选择项及S个有效写槽；int32/64原值 | Scalar | 同序列token顺序；不分配全序列workspace |
| accepted | 非空work最多1项，4/8 B | Main Scalar | 仅选初态 |
| 算法workspace | 0 B | 无中间结果GM区 | system workspace另计 |

连续化额外增加每个输入一次GM读和一次GM写，各为逻辑元素数乘2 B，并计入调用链耗时与显存。不同head共享Q/K依赖L2，cache hint由profiling决定。

### 遗留问题

1. 验证BSND/TND输入的独立stride/offset、跨batch DMA拆段、连续化和真实重叠反例。
2. 验证一次tile VF内逐token通知Main Scalar并提交State MTE3，以及tile完成后批量提交out MTE3的可执行同步协议。
3. 验证BF16/FP32寄存器Cast、按bank状态分配的两个递推bank及下一任务初态预取的读写顺序，确保预取和更新不覆盖有效State或未完成的MTE3源bank；验证两种layout的R=4行访问、K-first Gather/Scatter地址、BF16成对解包/打包及RINT舍入、直接快照/末态写回、重复槽、外层非连续和跨任务预取交接。核对寄存器峰值、State每token逻辑读一次写一次及覆盖前等待，并单独验证K-first精度和性能；本方案变更不代表实现或测试已完成。
4. 落实按最大有效序列长度Lmax及X最大为8的限制选择共用X及其传递、UB大小和槽偏移计算；验证定长与变长最大长度获取、多条不同长度请求、包含空序列及全空序列、packed容量大于有效token数、候选X=1/2/4/8、Lmax超过8时的多tile处理、拒绝X超过8、短于X及不能整除X的序列、容量边界和跨窗口递推一致性。核对最大长度获取开销，并比较同条件性能。
5. A2/A3/A5均验证未传入A_log时，传入或不传dt_bias、切换safe_gate及改变lower_bound均不改变out和State，也不触发raw gate专用校验；传入A_log时仍按实际使用的dt_bias及safe分支校验参数。同时验证仅由cu_seqlens判定序列模式、由A_log判定gate模式。A2/A3按第1.4节回归精度、资源配置及性能，确认共用host变更在对应kernel路径上合法。
