# Recurrent KDA A5 流水优化分析

## 采集口径

本次使用 A5_1、当前保留的 Stage4 实现、正式 kernel、B=2、H=HV=1、T=8、K=V=128、BF16 State、V-first、X=8、单 AIV，执行 `msprof op simulator`，采集 `PipeUtilization` 和 `ResourceConflictRatio`。

Stage8 是候选实验全部回退后的重新构建版本，流水结果为：

- 算子时长：约 36.53μs；Vector 区间约 34.84μs。
- Vector 指令区间中的空窗：启动前 4.541μs，两个 tile 之间 0.525μs，另有约 0.05μs 的局部间隙。
- MTE2 的输入、State 和辅助参数搬入在约 8.11μs 前完成，之后没有持续等待 MTE2 的长空窗。
- MTE3 的两个 out 搬出分别约 0.51μs 和 0.86μs；首个搬出与后续 Vector 启动重叠，最后一次形成尾部开销。
- Vector 主要指令计数：`RV_VMUL` 18528 次、`RV_VCADD` 4128 次、`RV_VLD` 6288 次、`RV_VCVT_F2F` 8320 次、`RV_VADD` 8288 次、`RV_VST` 4128 次、`RV_VDUP` 2080 次、`RV_VINTLV` 1536 次、`RV_VSELR` 512 次。

指令 CSV 中的 running time 是各指令累计执行时间，指令之间存在流水重叠，不能把这些数直接相加。流水空窗和 Vector 包络时间才用于判断端到端收益。

## 主要归因

### 1. 首次 Vector 启动前的 Scalar/描述符准备

从 1.30μs 到 5.872μs 的 4.541μs 空窗内，ScalarLDST 执行了约807条指令，主要是 `MOV_XD_IMM`、`LD_XD_XN_IMM`、`ST_XD_XN_IMM`、`MOVK`、`AND`、`SHL` 和 `ADD`。MTE2 正在搬入数据，但 Vector 只完成了 `MOVEMASK` 和 `PUSH_PB`，尚未进入 State 递推主体。

这说明首个 tile 的启动瓶颈是参数捕获、地址编码和任务描述符构建，不能归因于 State 计算或 GM 搬运。

### 2. Tile 边界的重复描述符重建

第一个 Vector 区间结束后，20.856μs 到 21.381μs 有约 0.525μs 空窗。该区间内 MTE2 只需要发出一个完成标记，前一 tile 的 out 搬出已经在 MTE3 上运行；ScalarLDST 仍执行约338条指令，随后才发出下一次 `PUSH_PB`。

因此，边界空窗主要来自下一次 `Tile` 调用重新计算 captured pointer、bank/slot 编码、参数地址和 valid-count，而不是下一 tile 的输入尚未搬完。

### 3. Vector 的数学计算和归约占主路径

`UpdateRows4` 对每个四行块执行 State 衰减、两次 key 乘积、delta 更新和 q 投影，`RV_VMUL` 达到18528次；四个预测值分别执行 `Sum128`，产生4128次 `RV_VCADD`。这两类指令是 Vector 主计算的核心，不能通过简单减少循环次数消除。

### 4. BF16 State 的转换和 UB 往返仍有固定成本

`RV_VLD`、`RV_VCVT_F2F`、`RV_VST` 分别为6288、8320、4128次。它们对应 State 读入、BF16/FP32 转换、State 回写以及 v 的中间 scratch 生命周期。`RV_SMEM_BAR` 共80次，累计约1.65μs，其中一部分用于保证 VST→VLD 和 VST→Scalar 事件发布顺序。

这部分有优化空间，但受“BF16 State 结果只在 UB 保留 BF16”和跨 token 递推依赖约束，不能简单删除转换或同步。

### 5. out 收集是次要的 Vector 优化点

`CollectOutputBlock` 仍产生1536次 `RV_VINTLV` 和512次 `RV_VSELR`。它的累计执行时间低于 State 读写和主乘法，但会延长四个归约结果的依赖链。直接删除 `Gather` 会改变 lane 顺序，必须先确认等价的 lane 插入/选择指令。

## 优化优先级

1. **优先做 X=8 完整 tile 的启动特化。** 为满窗口单独生成固定描述符路径，预先构造两个 input slot 的 q/k/v/gate/beta/output 基址、parameter slot 和 bank 编码；tile 循环只选择 slot 并写入 valid-count。尾 tile 继续走通用路径。目标是减少首 tile 和 tile 边界的 `MOVK/SHL/AND/ADD/LD_XD_XN_IMM`，并缩短 `PUSH_PB` 前空窗。

2. **把不随 tile 变化的地址编码移出 `Tile`。** 当前 `Tile` 每次进入 Vector scope 都重新捕获 UB 地址和参数指针。可以在 `RunTask` 中构造两个 slot 的轻量上下文，Vector scope 只读取一个上下文指针。必须保留 `LocalMemBar` 和事件顺序，先用指令计数确认 ScalarLDST 下降，再看端到端收益。

3. **尝试减少 `Sum128` 的归约依赖。** 评估是否能将四行预测的部分和以更适合 Vector 的布局合并，减少 `VCADD` 或减少 `Duplicate` 的广播链。只有在 lane 顺序和 FP32 归约顺序可证明等价时才实施，不能通过改变归约顺序换取不可接受的精度变化。

4. **优化 BF16 State 转换的隐藏方式。** 保留 BF16 State 的 UB 结果和必要的 VST→VLD/VST→Scalar 同步，尝试让下一行 State 的加载、转换与当前行的 `VMUL` 更深地交错。当前 `LoadDecayPrediction` 已经做了第一层交错，后续应以 Vector 包络和 `RV_SMEM_BAR` 数量为验收指标。

5. **最后再处理 out 收集。** 先验证固定 lane mask、局部半区累加或等价的选择指令能否减少 `VINTLV/VSELR`；保留完整 tile 后批量搬出 out。不要退回逐块 UB Store，也不要增加活跃 RegTensor 数量导致寄存器压力上升。

## 暂不优先的方向

- 不优先扩大 GM 搬运：MTE2 已在首个 Vector 主体之前完成后续输入准备，MTE3 也能与下一阶段重叠，当前没有持续 MTE 阻塞证据。
- 不再尝试单纯改变 v scratch 起始地址：此前 bank-group 实验精度通过但典型性能劣化约0.10%。
- 不再尝试仅改写 `row += 4` 的循环形式：此前没有形成有效 `VLOOPv2` 收益。

## 下一轮验证

先实现第1项和第2项的合并小实验，只针对 X=8 的满 tile，保留尾 tile 通用路径。使用 issue680 的 V-first BF16/FP32 ID7、ID35 作为首轮 case，比较：

- 首次 Vector 启动空窗和 tile 边界空窗；
- `ScalarLDST` 指令总数，尤其是 `MOVK/SHL/AND/ADD/LD_XD_XN_IMM`；
- Vector 包络、`RV_VMUL/RV_VCADD` 是否变化；
- 54 例对齐精度和典型20例 `msprof` 性能。

只有 Scalar 准备指令和端到端时延同时改善，才扩展到完整 issue680 case。
