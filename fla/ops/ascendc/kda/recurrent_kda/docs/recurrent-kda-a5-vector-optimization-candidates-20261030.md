# Recurrent KDA A5 Vector 优化候选

本文记录当前四项有序优化完成后的后续 Vector 优化实验。受保护的主方案文档不修改；每个候选单独构建、精度验证和 msprof 性能验证，只有超过采样波动的改进才进入下一项。

## 当前流水证据

完整局部 simulator 使用正式 kernel、B=2、H=HV=1、T=8、K=V=128、BF16 State、V-first、X=8、单 AIV：

- 优化前 38.338 μs，当前 36.525 μs；out 和 State 逐位一致。
- `RV_VCVT_F2F`：10336→8320；`RV_VDUP`：4128→2080，说明 v 批量转换已经生效。
- `RV_VMUL` 和 `RV_VCADD` 数量不变，主要数学计算量没有减少。
- `CollectOutputBlock` 仍产生1536次 `RV_VINTLV` 和512次 `RV_VSELR`。
- `RV_SMEM_BAR`：64→80；当前 v scratch 增加了 Vector Store→Load 生命周期和同步。
- Scalar 地址指令仍较多，优化后 `LD_XD_XN_IMM`、`ST_XD_XN_IMM`、`ADD_IMM`、`SHL` 均未下降。

当前完整流水图：`output/rkda-vfirst-ordered-optimizations-20261029/ordered-optimizations-pipeline.png`。

## 候选一：FP32 v 绕过 scratch

当前 BF16/FP16/FP32 v 都经过“加载到 FP32 寄存器→写入 UB scratch→从 UB 广播读取”。FP32 v 已经是计算 dtype，不需要 Cast 和 scratch 往返。

实施方式：仅在 `inType=float` 的编译期路径中直接把 `value + t*D` 传给 `UpdateRows4`；BF16/FP16 保留批量转换路径。预期减少 FP32 路径的 scratch Store、scratch Load 及对应 `VEC_STORE→VEC_LOAD` 屏障，不改变数学顺序和 State 舍入点。

验收：典型20例中 V-first FP32，重点观察 issue680 ID7/ID32；确认 `RV_VSTI`、`RV_SMEM_BAR` 和 Vector 时间变化。若没有超过 CV 的收益，回退该候选。

实测结果：Stage5 通过72对象审计和54例对齐精度。20个典型例相对 Stage4 的几何均值为0.999401，改善仅0.06%，低于采样波动；ID7 V-first FP32 为368.232→367.316μs，但 ID0 V-first FP32 反而上升0.66%。该候选放弃并回退。

## 候选二：减少 out 收集的 Interleave/Gather

`CollectOutputBlock` 对每个四行块执行3次 Interleave、Move 和 Gather。目标是让归约结果直接进入最终 lane，或将多个四行块的重排延后到每个64行半区末尾。

实施约束：继续保留两个 FP32 out 累加寄存器，继续在完整 token 后转换、完整 tile 后批量搬出；不能退回逐块 UB Store。优先尝试固定 lane mask/索引，避免扩大活跃 RegTensor 数量。

验收：重点比较 `RV_VINTLV`、`RV_VSELR`、Vector 时间和寄存器资源。若引入换入换出、`SMEM_BAR` 或精度差异，回退。

状态：候选一之后评估。当前 `Interleave/Move/Gather` 是四个标量归约结果进入单一向量累加器的既有 lane 组织；在没有确认可用的 lane 插入 API 前不直接删除 `Gather`。先保留 Stage2 的 State Store→out 收集顺序，避免为了减少一条指令破坏输出顺序。

## 候选三：使 State 行循环更适合 Hardware Loop

当前 V-first 两个行循环使用 `uint16_t row` 但步长为4，并且 `#pragma unroll 1`。尝试改为 `block=0..15`、步长1，再由 `row=block*4` 或递增指针取得四行地址，观察编译器是否生成更高效的 Hardware Loop 和地址寄存器。

实施约束：不手工引入 AddrReg，保持 `base + index * constant_stride` 形式；不盲目展开，先核对 RegTensor 是否接近32个上限。

验收：比较 `RV_VLOOPv2`、Scalar 地址指令、Vector 时间、寄存器/栈资源及所有布局精度。若硬件循环未增加或寄存器开销上升，回退。

实测结果：Stage6 通过72对象审计和54例对齐精度。相对 Stage4 的典型20例几何均值为0.999943，耗时变化约+0.006%，`VLOOPv2`、Scalar 地址指令和 Vector 时间没有形成可重复改善，属于采样波动范围。该候选放弃并回退；原因是编译器没有把这种整数块循环转成更短的硬件循环，且 `row=block*4` 仍保留了地址乘法。

## 候选四：调整 v scratch 的 UB bank group

3510 UB 的 bank group 冲突与地址低位有关。当前 v scratch 从 `eventOffset+256` 开始，与 event/output 地址物理上较近。只改变 scratch 起始偏移到另一个32B bank-group位置，例如 `eventOffset+288`，不改变容量和生命周期，使用 msprof `ResourceConflictRatio` 验证是否减少冲突。

实施约束：scratch 仍必须位于 event 区域之后的预留尾部，不能覆盖 event[0]/event[8]，不能与输入、out、参数或 DMA 源重叠。连续访问的 V-first State 不直接套用 K-first 的 padding。

验收：比较 ResourceConflictRatio、Vector 时间、`RV_SMEM_BAR` 和8组原生内存检查。若地址变化不影响冲突率或性能，回退。

实测结果：Stage7 通过72对象审计和54例对齐精度。典型20例几何均值为1.001016，相对 Stage4 劣化约0.10%；V-first FP32 的 ID7/ID35 虽分别下降0.27%/0.19%，但 V-first BF16 的 ID3/ID7/ID35 上升0.50%/0.47%/0.60%，没有稳定收益。单次 simulator 为36.56μs，Stage4 为36.525μs。该候选放弃并回退，说明当前主要瓶颈不是这个 scratch 起始地址的 bank-group 相位。

## 候选五：扩大连续搬运或 UB 融合

当前 q/k/v/g/beta 已按 tile 批量搬入，out 已按完整 tile 批量搬出，中间 State/out 不经过 GM。只有在 msprof 显示 MTE2/MTE3 或 UB 往返仍是关键路径时，才考虑重新组织跨 head/task 的连续搬运。

状态：候选四未保留后完成评估，不修改代码。q/k/v/g/beta 已按 tile 批量搬入，out 已整 tile 批量搬出；已有 simulator 中 MTE2/MTE3 只占较小部分，主要时间在 Vector。若直接扩大跨任务连续搬运，需要改变 task 边界和 State 生命周期，当前证据不足以支持这个风险。候选五因此作为“无需实施”的候选关闭。

## 固定实验口径

A5_1 物理卡7，正常标准 run 包，msprof `Task Duration(us)`，每例3次预热、5次样本。每个候选先跑20个典型组合，再根据结果决定是否扩展到 issue680 全144组。精度沿用132个原有用例和54个对齐用例；涉及 scratch 或同步时补充8组原生 sanitizer/trace。

每轮结果记录在本文件对应候选下，并保留远程目录 `<run-root>/rkda-vfirst-ordered-optimizations-20261029` 的原始 CSV、manifest、日志和构建审计。

## 最终回退确认

候选三、四的临时代码均已回退；候选一的 X=8 V-first 固定描述符特化在后续完整验证中保留。Stage8 回退基线曾在 A5_1 完成72对象审计和54例对齐精度，典型20例 msprof 几何均值相对 Stage4 为1.000719。

## 按流水优先级的后续实验

### X=8 V-first 固定描述符路径

在 Stage8 基线上增加了 X=8 完整 tile 的固定窗口特化，仅作用于 V-first；尾 tile、其他窗口和 K-first 保持通用路径。修正了参数槽低位不能压入 launch word 的问题，固定路径最终仍使用捕获的参数指针。

Stage16 结果：72对象审计、132例原始精度和54例对齐精度全部通过。相对 Stage8 的典型20例几何均值为 **0.995293**；V-first FP32 约改善1.2%～1.4%，V-first BF16 约改善0.3%～0.7%，K-first基本不变。simulator 中 Vector 包络由34.842μs降至约34.26μs，启动和 tile 边界空窗缩短。该优化保留。

### 两个 input slot 地址上下文预计算

预计算两个 slot 的 UB 地址并在 Tile 中选择，精度通过，但典型几何均值相对 Stage11 只有 **0.999364**，Scalar 指令数量没有下降，收益低于波动，已回退。

### Sum128 原乘积寄存器复用

将 `Sum128` 的加法结果写回原乘积寄存器，精度通过；`RV_VMUL/RV_VCADD` 数量和 simulator 时延不变，典型几何均值相对 Stage11 为 **0.999333**，已回退。

### 批量加载 BF16 State 后再统一衰减

尝试先加载四行的八个 State 片段，再统一执行衰减和 key 乘法。精度通过，但 BF16 V-first 典型 case 劣化约0.8%～1.0%，几何均值为 **1.000393**，已回退。

### 直接 Gather 替代 Move+Gather

54例对齐精度通过，典型性能表面上改善约15%～20%，但完整132例中 norm_signal 的 case 43、87、131 输出匹配率仅6.25%。原因是直接 Gather 覆盖了 output 累加器，丢失了此前四行块积累的值。该实现错误，已回退，继续保留 `Move+Gather`。

最终 Stage16 检查：72对象审计、132例原始精度、54例对齐精度、74例公共接口和9例确定性检查全部通过。当前保留的只有 X=8 V-first 固定描述符特化。
