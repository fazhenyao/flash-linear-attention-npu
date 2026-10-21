# RecurrentKda 设计方案

方案设计规则版本：`V2`

## Issue 680：A5 窗口递推设计

接口与差异以 [api.md](api.md) 为准。参考 PR #917 固定提交
`082b0c704407d0b38909fd4b0f58703b8ab52873` 的窗口递推、epsilon 和
CPU FP64 标杆；保留本算子现有 arch35 AIV-only 融合归约，不引入 Cube Stage。
以下为本次变更的完整受影响 Stage 设计，后文保留原有工程设计背景。

### 1. 目标

支持 A5 任意合法长度的 recurrent 序列，窗口仅为容量8的实现细节；状态在窗口间以FP32
驻留。保持既有 QKV stride 直接访问和 ACLNN 连续化回退。
模型 case 使用 issue 的 B={32,64,72},HK=16,HV={32,48,128},T={1,2,4,8},K=V=128。
预热10次、采样20次，统计目标 recurrent kernel 的 Task Duration(us) 平均值；
含连续化或非原位复制时另报告完整调用所含kernel之和。用户确认本轮目标为功能完整且
性能不劣于库上已有实现，以 `67dc99d7` 为同口径基线；不要求本轮验证原生实现的0.8倍。
逐case结果归档至 `tests/atk/recurrent_kda/README.md`。

主要优化空间来自 state 的重复快照和写回。FP32 state 每head为64KiB，无slot时
每work原来读取一次、写T次，优化为读取一次、末token写一次；slot模式仍逐token写。
单次模型state读写量从 `B*HV*65536*(1+T)` 降为 `B*HV*65536*2`，T=1不变，
T=8理论state流量下降约78%。计算和输出仍为O(T)，该估计不是实测达标结论。

### 2. Stage 0：Vector，窗口预处理与状态递推

定义 W=8，L=min(W,seq_end-window_start)，R=vStep，D=K，E=V。
work id=seq*HV+hv，hk=floor(hv/(HV/HK))。保持原grid-stride分发，
Nwork=seq_num*HV，blockDim=min(AIV_count,Nwork)。例如B=32,HV=48时Nwork=1536；
实际AIV_count由平台查询，本次A5_ci实测blockDim=56，即28个wave、尾wave24个work。
尾work用相同所有权规则。
每work按V方向R行分片，每片遍历全部窗口并保持 `[R,D]` FP32 state驻留。
窗口输入的搬运次数为ceil(E/R)；这是既有state切片容量不足时的明确重读代价，
不新增全量GM中转，不更改分核或引入跨核归约。

每窗口先读取真实token/head stride的QKV和紧凑gate/beta；FP32计算：
`q_hat=q/sqrt(sum(q*q)+1e-6)`，k同理；随后独立乘q scale。
raw gate的unsafe分支使用稳定softplus：`max(x,0)+log(1+exp(-abs(x)))`，
再乘 `-exp(A_log)`；safe分支保持lower_bound*sigmoid(exp(A_log)*x)。
unsafe softplus的逐元素链在同一RegBase VF内计算，完整64元素寄存器搬入/搬出，
K=128使每窗元素数恰为寄存器宽度的整数倍；中间值只驻留寄存器，结束后V屏障。
该调整合并稳定公式新增的中间UB读写和V屏障，不改变窗口、任务映射或缓冲容量。
gate decay和beta sigmoid均以FP32计算；有限FP32 A_log需exp结果可表示，
g的softplus不再因正向大值发生中间exp溢出。零范数由epsilon保护。
随后逐token按既有state decay、delta、outer和output次序计算。
Q/K 每窗分别批量平方与归约，delta/attn 各保存 L 个 norm；两组 epsilon/sqrt
完成后用RegBase广播行norm并在V侧逐行相除，不读取Scalar、不新增V→S等待。
L<=8<=R，结果槽容量不随序列长度增长。批归约沿用参考PR的批处理方法，
行缩放采用本仓既有A5 RegBase DataCopy/Div接口，保持独立q scale次序。
只有state输出cast成state dtype，跨窗口state保持FP32，避免BF16窗口边界误差。

每个AIV的UB地址由TPipe按以下绝对半开区间顺序分配。单位为byte，s=state dtype字节，
O_s/O_a为host选出的输出队列深度，A32(x)=ceil(x/32)*32；R为8或16的倍数。
U=A32(tiling.ubRestBytes)是compute buffer实际保留容量，tmpBytes为内部有效切片峰值。

```text
UB[0,2WD) q_raw BF16[W,D]；MTE2写，cast末次读，queue释放后复用
UB[2WD,4WD) k_raw BF16[W,D]；同上
UB[4WD,4WD+2WE) v_raw BF16[W,E]；同上
UB[4WD+2WE,8WD+2WE) gate_raw 最多FP32[W,D]；同上
UB[A,A+64W) beta_raw；A=8WD+2WE，FP16/BF16每token补齐32B，预留FP32展开上限
UB[B,B+2sRD) state_in[2]；B=A+64W，MTE2生产，V消费，输入queue释放后复用
UB[C,C+O_s*sRD) state_out[O_s]；C=B+2sRD，V生产，MTE3末次读
UB[F,F+X) transpose；F=C+O_s*sRD，K-first时X=sRD，否则X=0
UB[G,G+O_a*A32(2R)) out[O_a]；G=F+X，V生产，MTE3末次读
UB[H,H+U) compute reserve；H=G+O_a*A32(2R)，其中有效切片使用tmpBytes
UB[H+U,H+U+64) scalar；V/S局部同步后读取
```

compute基址H处：delta `[H,H+4R)`，attn `[H+4R,H+8R)`；接着为v `[W,E]`、
q `[W,D]`、k `[W,D]`、state `[R,D]`、beta `[W,16]`、gate `[W,D]`、
独立preprocess scratch `[W,D]`，均FP32。故A5/K128的
`tmpBytes=8R+4WE+16WD+4RD+64W`。
预处理scratch用于norm平方/归约、dt_bias和softplus，绝不alias活跃state。
delta/attn在窗口预处理时用于norm结果，首次递推后改为各V行结果，互斥生命周期。
host从可用UB扣除固定输入、独立scratch、scalar、2048B guard后求R，
并核验所有队列、compute峰值；guard覆盖allocator对齐余量。
不使用L1/L0，不新增算法workspace，系统workspace16MiB。

| Tensor | 每work GM次数及byte | 驻留与复用 |
| --- | --- | --- |
| Q/K | 各读ceil(E/R)次，各2TD | 每窗口cast后留UB到当前窗口末token |
| V | 读ceil(E/R)次，2TE | 每窗口UB |
| gate/beta | 每V片读一次全序列，按原dtype计量 | 每窗口UB，预处理后复用 |
| state | 每V片初读一次，共sED | FP32驻留跨全部窗口 |
| state输出 | 无slot末token写一次sED；有slot逐token写TsED | 输出queue快照，不覆盖驻留state |
| out | 写一次2TE | 输出queue，按token写有效V行 |

所有搬运保持现有默认cache策略。L=1..8，尾窗只处理有效行；空序列在输入/state读取前跳过，
padding tail不读写，无效slot按现有检查退出，不引入新增等待。

同步资源沿用queue的MTE2→V ready/V→MTE2 free及V→MTE3 ready/MTE3→V free；
显式局部event为MTE2_V、V_MTE2、V_S、V_MTE3、MTE3_V各至多1个，lazy分配、Set/Wait配对、
末任务释放。长窗覆盖FP32输入前等待此前V操作完成；预处理与state的地址互不重叠。
slot模式同一GM槽允许连续写，写回完成后再复用，避免尾部两个快照重叠写。

```text
validate metadata -> grid-stride work (空序列跳过)
  for V片: prefetch/dequeue initial state -> FP32驻留
    for window: queue读取L行输入 -> cast -> normalize/gate
      for token: compute -> queue输出；需state快照时queue保存并写回
      确认V末次消费 -> 下一窗口复用输入/scratch
    排空输出queue -> 下一V片/下一work
release paired local events
```

R01/R04/R06/R07/R09/R13的Cube协同部分对原AIV-only路径不适用；
R08/R11/R17按上述地址与生命周期核验，R10/R14记录V片重读，R20保留原所有权，
R21用epsilon和稳定softplus保护。其余未受影响的语义按既有设计回归。

## 1. 背景

KDA recurrent 用于小步长 decode/MTP 场景，按 token 顺序更新 KDA 状态并生成输出。上游常用 API 支持
raw gate、`A_log/dt_bias/lower_bound/safe_gate`、`beta sigmoid` 和两种 state 布局。该实现的目标
是在一个 KDA 独立算子中完成这些语义，避免把 gate 预处理拆成 `KdaGateCumsum + recurrent` 两段公共实现，也不复用
GDN recurrent 的函数接口。

## 2. 目标与非目标

### 2.1 目标

- 提供 Ascend C 实现类型的 `recurrent_kda`，覆盖 `fla_npu.ops.ascendc`、aclnn 和 `<<<>>>` 入口；legacy 通路不属于支持范围。
- 对齐常用上游 API：`raw gate`、`A_log`、`dt_bias`、`lower_bound`、`safe_gate`、`beta sigmoid`、`allow_neg_eigval`。
- 支持 `BSND` 和 `TND`，支持 device `cu_seqlens` 变长序列、容量化 state pool 和 speculative slot 索引。
- 保持 KDA recurrent 与 GDN recurrent 接口解耦。

### 2.2 非目标

- 不提供内部 state 矩阵任意转置或步长采样的低性能 gather/scatter 路径。
- 不支持把 `KdaGateCumsum` 的输出流水作为公开前置步骤。
- 不引入 chunk/Cube 并行算法；A5 长序列按窗口串行递推，A2/A3 当前工作树仍限制每段 `<=8`。

## 3. 能力边界

实现类型：`ascendc`

支持范围以当前公共接口和实现约束为准，Shape 符号引用 [KDA 模型符号表](../../README.md#model-shape-symbols)。

| 维度 | 支持范围 |
| --- | --- |
| SOC | `ascend910b`、`ascend910_93`、`ascend950` |
| Layout | `BSND`、`TND` |
| Q/K/V dtype | BF16 |
| K/V | `K=128,V=128` 或 `K=128,V=256` |
| Heads | `1<=H_k,H_v<=256`，且 `H_v % H_k == 0` |
| Gate/Beta dtype | FP32/BF16/FP16，kernel 在 UB 中转为 FP32 计算 |
| State dtype | FP32/BF16 |
| Gate 模式 | 预计算 step log gate；kernel 内 raw gate |
| State layout | V-first `[state_capacity,H_v,V,K]`；K-first `[state_capacity,H_v,K,V]` |

## 4. 数学与接口语义

若 `use_qk_l2norm_in_kernel=True`，先对每个 token 的 `q/k` 在最后一维做 L2 normalize。随后 `q` 乘以 `scale`。

`use_gate_in_kernel=false` 时，`g` 已经是 step log gate。`use_gate_in_kernel=true` 时，kernel 内转换 raw gate：

```text
gate = -exp(A_log) * softplus(g + dt_bias)                 # safe_gate=False
gate = lower_bound * sigmoid(exp(A_log) * (g + dt_bias))   # safe_gate=True
```

`use_beta_sigmoid_in_kernel=true` 时：

```text
beta = sigmoid(beta)
if allow_neg_eigval:
    beta = 2 * beta
```

每个 token 的 recurrent 更新为：

```text
S = exp(gate_t) * S
delta = beta_t * (v_t - S @ k_t)
S = S + outer(delta, k_t)
o_t = S @ q_t
```

`H_v % H_k == 0`，每个 value head 通过 `floor(h_v / (H_v / H_k))` 映射到一个 query/key head。

## 5. 整体架构

- `op_host/op_api/aclnn_recurrent_kda.cpp`：校验 dtype、layout、shape、可选输入组合；不读取 device metadata 的值；
  对规则非连续 Q/K/V 保留 stride/storage offset，已验证布局回退连续化，token/head 置换或
  正 stride 重叠 descriptor 明确拒绝；非连续 state 以
  `CreateView` 保留真实存储信息；创建 L0 executor。
- `op_host/recurrent_kda_tiling.cpp`：读取 Q/K/V 与输入/输出 state stride、shape、属性和可选输入存在性，
  填充 `RecurrentKdaTilingData`，计算 block dim 与 UB 切分。
- `op_kernel/recurrent_kda.cpp`：单个 AIV kernel 完成 q/k normalize、raw gate 转换、state decay、delta 更新、输出和最终状态写回。
- `torch_custom/fla_npu/csrc/src/stable_recurrent_kda.cpp` + `fla_npu/ops/ascendc/_stable.py`：提供解耦的 Python 入口（Stable-ABI 适配层），不注册 legacy `torch.ops.npu`；`_aclnn_ctypes.py` 保留为参考实现与回退。
- legacy `torch.ops.npu.npu_recurrent_kda` 不属于当前支持范围，不作为公共接口维护。

## 6. Tiling 设计

### 6.1 任务划分

Tiling 按逻辑序列和 value head 拆分任务。提供 `cu_seqlens` 时，`seq_num=len(cu_seqlens)-1`，使用与
fla-org 一致的累计边界语义，第 `i` 条序列范围为 `[cu_seqlens[i], cu_seqlens[i+1])`；未提供时，
BSND 按 batch 行划分序列，TND 视为一条序列。接口面向 decode / MTP 短序列：每个相邻 offset
的差值（varlen 每段长度）与 dense 单序列长度在 A2/A3 当前工作树限制为 `<=8`；A5 按窗口递推。提供 `cu_seqlens` 时，末项表示有效 token 数，可小于图捕获的
token capacity。

### 6.2 Tiling Data

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `vectorCoreNum` | `uint32_t` | 使用的 AIV core 数 |
| `ubCalSize` / `ubRestBytes` | `uint32_t` | UB 计算区大小与余量 |
| `t` / `seqLen` | `uint32_t` | token 总数与输入物理序列轴长度 |
| `nk` / `dk` | `uint32_t` | query/key head 数与 K 维 |
| `nv` / `dv` | `uint32_t` | value head 数与 V 维 |
| `sBlockNum` / `b` | `uint32_t` | state pool 容量与逻辑序列数 |
| `ssmStateStride` | `uint32_t` | 二维 speculative state 索引的行 stride；packed 一维索引时为 0 |
| `query/key/value TokenStride` | `uint64_t` | Q/K/V 相邻逻辑 token 的元素 stride |
| `query/key/value HeadStride` | `uint64_t` | Q/K/V 相邻 head 的元素 stride |
| `vStep` | `uint32_t` | 单次处理的 V 方向步长 |
| `stateOutBufferNum` / `attnOutBufferNum` | `uint32_t` | UB ring buffer 数 |
| `scale` / `lowerBound` | `float` | query scale 与 safe gate 下界 |
| `layout` | `uint32_t` | BSND/TND 编码 |
| `has*` | `uint32_t` | 可选输入存在性 |
| `use*` / `safeGate` / `stateVFirst` | `uint32_t` | 属性分支 |

### 6.3 模板化方案与 tiling key

当前 kernel 通过 `REGISTER_TILING_DEFAULT(RecurrentKdaTilingData)` 使用默认 tiling data，不引入额外 tiling key。
dtype 固定为 BF16 QKV，state dtype 由编译宏 `DTYPE_INITIAL_STATE` 实例化；layout、gate/beta 分支和可选输入存在性由
tiling data 驱动，避免把 runtime shape 组合扩张到 tiling key。

## 7. Kernel 设计

### 7.1 计算流程

1. 解析当前任务对应的逻辑序列、value head、query/key head 和状态槽。
2. 加载初始状态到 UB 工作区；Python 仅在非原位模式下允许 `initial_state=None`，并预先创建全零状态。
3. 按 Q/K/V 各自 token/head stride 计算 GM 首地址并二维搬运到紧凑 UB；随后按属性执行 q/k
   normalize、raw gate 转换和 beta sigmoid。
4. 对状态做 gate decay，计算 `S @ k`、delta 和 outer 更新。
5. 计算 `S @ q` 写回 `out`，并把状态原位写回命中的 state pool 槽。

### 7.2 内存规划

| 内存层级 | Buffer | 生命周期 | 用途 |
| --- | --- | --- | --- |
| GM | 输入/输出 tensor | kernel 全程 | q/k/v/g/beta/state/out 与 device metadata |
| GM | workspace | kernel 调用期间 | 无算法专用 GM 中转；按标准运行时预留 16 MiB system workspace |
| UB | state 工作区 | 单任务内复用 | 保存 `[V,K]` 局部状态片段 |
| UB | q/k/v/g/beta 临时区 | 单 token 内复用 | normalize、gate、beta 与输出计算 |

### 7.3 流水与同步

算子为 AIV-only recurrent 小步长 kernel，单任务内部按 token 串行维护状态依赖。调用侧保证不同活跃序列不共享写槽，
任务分配保证每个 `(seq/value head)` 的输出区域独立。`--cce-auto-sync=off` 保持不变，UB 复用由本 kernel 内顺序和
必要的 vector pipe barrier 维护，不依赖 GDN recurrent 的同步协议。

### 7.4 边界处理

- 提供 `cu_seqlens` 时，首项必须为 0，offset 必须单调不减，末项为有效 token 数且不得超过输入 token
  capacity。host 只检查 rank/dtype，device kernel 校验值。
- 相邻 offset 相等时表示空序列；该序列不产生 token 输出，也不读取 `ssm_state_indices/num_accepted_tokens` 或 state。
- 末项小于 capacity 时，仅 `[0,cu_seqlens[-1])` 的有效输出和 state side effect 有定义；padding tail 输出不作保证。
- 一维 `ssm_state_indices` 按 packed token 偏移读取；二维索引按 `[seq_idx,step]` 读取，第二维必须覆盖对应序列长度。
- `num_accepted_tokens` 只在提供 `ssm_state_indices` 时有效，用于定位 MTP decode 的初始状态槽。

## 8. 平台设计

| 平台 | SOC | 实现策略 | 分支差异 |
| --- | --- | --- | --- |
| A2 | `ascend910b` | 公共 AIV kernel | 无 |
| A3 | `ascend910_93` | 公共 AIV kernel | 无 |
| A5 | `ascend950` | arch35 特化 AIV kernel | RegBase/DataCopy 路径，并对 `K=128` 使用融合归约实现 |

## 9. 精度设计

Q/K/V 公开输入为 BF16，gate/beta 保持 FP32/BF16/FP16 输入 dtype，并在 kernel 的 UB 中转为 FP32 计算。state 可为 FP32 或 BF16；推荐 FP32 状态以降低小步长多 token 更新中的累计误差。

## 10. 性能设计

目标场景是 decode/MTP recurrent 小步长，主要瓶颈是状态矩阵读写和每 token 的 `S @ k`、`outer(delta,k)`、`S @ q`。
当前实现已使用输入/输出双缓冲，并在 arch35 的 `K=128` 路径采用融合归约和 repeat DataCopy；后续优化仍应以
状态搬运、向量流水和同步开销为重点。

## 11. 测试设计

正式测试工程维护于 `tests/atk/recurrent_kda/`；以下保留既有场景背景，本次实际验收范围和结果以该目录 README 为准：

- BSND raw gate + unsafe gate + beta sigmoid。
- BSND safe gate + `allow_neg_eigval`。
- TND 预计算 log gate + 空 initial_state。
- Kimi/KDA 关键泛化 shape：GVA head 映射、`K=128,V=128` dense raw gate、`K=128,V=256` TND safe gate。
- Kimi K3 TP16：本地 head 数 6、`K=V=128`、BF16 Q/K/V、raw+safe gate、TND packed decode/MTP 1-8；
  覆盖 `[0,1,4,4,5]` 非等长序列、二维 slot 索引、容量化 state pool、非连续槽更新和未命中槽保持不变。
- 空序列覆盖 `[0,1,1,4,4,8]`，确认重复 offset 对应的空序列不读取索引或 state。
- Kimi H96/D128 smoke：`H=H_v=96,K=V=128,safe_gate=True`，运行时生成非等距 `cu_seqlens`，
  覆盖 `cu_seqlens` 长度和值泛化；A2/A3 每段遵循 `<=8` 限制，A5 覆盖跨窗口长度。
- Kimi 完整长上下文 stress target：`T_total=12288,H=H_v=96,K=V=128,safe_gate=True` 作为本地 stress case，
  因当前单 kernel 计算量较大，不纳入默认通过矩阵。
- 非连续 state 覆盖 V-first/K-first、FP32/BF16、原位/非原位、slot pool guard 与未命中槽。
- 负向参数组合：A2/A3 超长序列、`safe_gate` 与 raw gate 组合、内部矩阵非稠密 stride 与外层重叠 stride。

## 12. 验证修复记录

A2/A5 精度验证过程中修复以下问题：

- `scale/lower_bound` 的 aclnn 公开参数为 `double`，AICore attr 为 `float`；L0 加入显式 cast，避免 tiling 读取到错误标量。
- A5 tiling 对 `GetOriginShape()` 返回对象生命周期更敏感；tiling context 改为持有 shape 副本，避免 optional/required shape 指针悬挂。
- 原实现 q/k L2 normalize 从 UB 标量读取 norm 前补充 V->S 同步；本次 A5 改为 V 侧广播/相除，A2/A3 保持原同步路径。

## 13. 已知限制与演进计划

- 当前仅支持 BF16 QKV。
- Q/K/V 直接非连续访问仅支持 feature 致密、head/token 不重叠的正 stride；BSND `B>1` 还要求
  batch 维可按 `T * tokenStride` 扁平化。已验证的 feature-gap、zero-stride、batch-gap view
  自动回退连续化；token/head 置换或正 stride 重叠 descriptor 明确拒绝。
- 当前 `K/V` 仅支持 `K=128,V=128` 或 `K=128,V=256`。
- A2/A3 当前工作树每段限制为 `<=8`；A5 不再受该限制，仍受 uint32 容量和设备内存约束。
- state 最后两维必须为行主序稠密矩阵；slot/head 维可有间隔但不得重叠。
- 显式 slot 模式要求所有活跃序列的写槽互不冲突，且 slot 值位于 state pool 容量范围内。
- 后续若扩展 A2/A3 长序列或更多 dtype，需要同步更新 op_host 校验、tiling、kernel、测试 case 和 API 文档。
