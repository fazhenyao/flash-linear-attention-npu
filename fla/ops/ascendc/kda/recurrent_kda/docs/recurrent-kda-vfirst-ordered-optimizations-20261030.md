# Recurrent KDA：按流水瓶颈顺序优化

2026-10-29开始，2026-10-30验收。基于已验证的 K-first UB 行距 padding 版本，按 v 转换、out 收集、State 调度、任务边界四步依次构建和验收。本轮只改 A5 的两个私有 kernel header；接口、CPU 标杆、host tiling、X 选择和 A2/A3 kernel 不变。用户的 `recurrent-kda-a5-regbase-redesign-20261026.md` 未改动，SHA256 为 `78f03d56724f4ff6ba36609f1f53dbc05148825b94ee7d253a7867c8c331d5b4`。

## 采用的结构

### 1. v 批量转换

原先每个 V 行从 BF16 v 广播加载后，先转换一个有效元素，再 Duplicate 到整个 FP32 寄存器。一个 token 重复128次 Cast 和128次 Duplicate。

现在每 token 用两个64元素寄存器完成全部128个 v 的精确 BF16→FP32 转换，写入 UB 中现有预留尾部的512B，再由行更新使用 FP32 广播加载。每 token 的 v Cast 从128次降为2次，v Duplicate 从128次降为0；代价为两个向量 Store、两个批量 Load 和一次 VST→VLD 屏障。未增加 GM 搬运或 UB 分配。

单个 Vector Stage 的 UB 总分配仍为 `E+2048` 字节，`E=eventOffset`。State、q/k/v/g/beta、参数和双 out 槽的地址沿用原实现。尾部局部地址图如下：

| 区间，单位B | dtype/有效量 | 生命周期与复用 |
| --- | --- | --- |
| `[E,E+4)` | uint32，VF 完成 token 数 | Scalar 初始化、VF 发布、Scalar 读取，保留原握手 |
| `[E+4,E+32)` | 无有效数据 | 保留，避免共享事件32B块 |
| `[E+32,E+36)` | uint32，MTE3 完成 token 数 | Scalar 发布，VF 覆盖快照源前读取 |
| `[E+36,E+256)` | 无有效数据 | 保留 |
| `[E+256,E+768)` | FP32 `[128]`，当前 token v | 当前 VF 批量写入并屏障后，由当前 token 全部 V 行消费；行循环完成后下一 token 覆盖 |
| `[E+768,E+2048)` | 无有效数据 | 连续空闲1280B |

同一核的 VF 顺序执行，Scalar/MTE 不访问 v scratch；未来输入仍搬入另一原始输入槽。FP32 v 表示原 BF16 值，既不改变数学顺序，也不改变 State 舍入位置。

### 2. out 收集依赖

四行输出继续先从尚未舍入的 FP32 State 完成归约。将独立的 State Cast/Store 放在 `CollectOutputBlock` 前，让输出寄存器重排链不再先于 State Store 下发。仍用原来的 Interleave、合并与旋转；本轮没有消除归约结果的必要广播，也没有减少该收集算法的指令数量。

两组64行仍在两个 FP32 寄存器收集。完整 token 计算后才将 out 转换并写入 UB；**完整 tile 结束后才批量搬出 out**。没有恢复逐行或逐 token 的 out MTE3。

### 3. State 加载、转换与乘法交错

V-first 对每个连续64元素片段依次执行 Load、BF16→FP32 Cast（如需要）、decay 乘法和 prediction 乘法，再处理后续片段。用 `LoadDecayPrediction` 表达该顺序，交由编译器安排独立指令。四行完整 State 和对应乘积仍使用原寄存器集合，128元素归约及最终加法顺序不变。K-first 保留原有成对 Gather/Scatter 和32B行距 padding。

State 在 UB 仍为接口 dtype：BF16 State 每 token 在寄存器转成 FP32，更新后转回 BF16；out 在 State 舍入前计算。没有引入 FP32 State UB 副本。

### 4. 提前 Dispatch

最终状态路径与快照路径都改为：

```text
等待当前 tile 的 MTE2→V、S→V
Dispatch 当前 tile VF
Scalar 提交另一输入槽的未来 q/k/v/g/beta
若下一任务存在且第三 bank 已空闲，提交其 State/参数预取
若第三 bank 仍属于旧 MTE3，按原事件等待后释放并预取
处理原有快照发布/完成握手，回收旧 out DMA
等待当前 tile VF 完成
批量提交当前 tile out；按原规则提交最终 State 或逐 token 快照
后续覆盖前等待；尾部排空
```

当前 VF 只访问当前输入槽、当前参数槽、两块递推 bank 和核私有 scratch。预取使用另一输入/参数槽，以及状态为 free 的第三 bank。bank 编号依当时状态选择，并非固定 A/B/C。所有 Set/Wait、快照覆盖前等待、EVENT1 消费和最终排空保持原语义；只把未来预取移至当前 VF 启动之后。

## 验收口径与证据

A5_1，物理卡7，Ascend950PR，CANN 9.2.0-beta.2。每轮以标准单算子 `.run` 构建安装，核验 opapi MD5、72个 kernel 的 JSON SHA256 与构建目录对象。独立环境和冻结包区分 baseline/stage1/stage2/stage3/stage4，性能与插桩诊断分离。

精度沿用既有 CPU 标杆、输入生成器、值域及容差。每步先54个对齐精度用例；最终回归132个原有用例、54个对齐用例、74项公开接口检查和9项确定性测试。原生 memcheck/trace 覆盖 FP32/BF16 × V-first/K-first × 最终状态/逐 token 快照，每组4个连续任务，S=9、X=8，触发两次 tile、双槽回绕、第三 bank 预取和 snapshot bank 重用。

性能仅使用正常构建的 `msprof --task-time=on`，取目标 RecurrentKda 的 `Task Duration(us)`。每例3次预热、5次正式样本，实际56个 block；先20个典型组合，再以相同设备和脚本重测优化前后全部 issue680 的36组模型形状 × 两种 State 布局 × 两种 State dtype。不同流水计数器有重叠，不能相加；simulator 时间不作为真机性能验收值。

开发结果与最终流水证据见下文。完整原始数据位于主工作区 `output/rkda-vfirst-ordered-optimizations-20261029`，A5_1 对应 `<run-root>/rkda-vfirst-ordered-optimizations-20261029`。

## 分阶段结果

每一阶段都使用独立标准 run 包，并在下一阶段前完成构建审计和54例对齐精度。

| 阶段 | 改动 | 典型20例几何均值（相对上一阶段） | 结果 |
| --- | --- | ---: | --- |
| Stage1 | v 批量转换 | 0.978404 | 54例精度通过，采用 |
| Stage2 | State Store 先于 out 收集 | 0.996504 | 54例精度通过；K-first BF16 稳定改善，采用 |
| Stage3 | V-first State 加载/转换/乘法交错 | 0.998626 | 54例精度通过；V-first BF16 T≥4 改善，采用 |
| Stage4 | 当前 VF 先 Dispatch，再提交未来输入和 State 预取 | 0.990044 | 54例精度通过，采用 |

最终对比以 Stage4 与同卡、同脚本重测的 UB padding baseline 为准。issue680 的36个模型形状、V/K 两种 State 布局、FP32/BF16 两种 State dtype，共144组，均为 msprof `Task Duration(us)`、3次预热和5次样本：

| 布局 | State | 当前/基线几何均值 | 耗时变化 | 当前更快 |
| --- | --- | ---: | ---: | ---: |
| V-first | FP32 | 0.966966 | −3.30% | 35/36 |
| V-first | BF16 | 0.964912 | −3.51% | 36/36 |
| K-first | FP32 | 0.957509 | −4.25% | 36/36 |
| K-first | BF16 | 0.968163 | −3.18% | 36/36 |
| 全部 | 全部 | 0.964379 | −3.56% | 143/144 |

唯一未快于基线的组合是 V-first FP32 ID32，比例1.00135，低于该组采样波动判断阈值；其余143组均下降。issue680 ID7（B=32、H=16、HV=48、T=8）结果为：V-first FP32 381.524→368.533 μs（−3.41%），V-first BF16 448.401→432.535 μs（−3.54%），K-first FP32 1273.109→1228.891 μs（−3.47%），K-first BF16 833.928→813.768 μs（−2.42%）。

## 指令与流水证据

完整局部 simulator 采集使用相同正式 kernel、B=2、H=HV=1、T=8、K=V=128、BF16 State、V-first、X=8、单 AIV。优化前/后完整结束时间为38.338/36.525 μs。动态指令数量：`RV_VCVT_F2F` 10336→8320，`RV_VDUP` 4128→2080，`RV_VLD` 6256→6288，主要 State/归约指令数量不变；差异说明 v 转换重复被消除，增加的32次 `RV_VSTI` 属于批量 scratch 写入。优化前后 out 和 State 均 bitwise equal。流水图为 `output/rkda-vfirst-ordered-optimizations-20261029/ordered-optimizations-pipeline.png`。

## 完成验证

- 72个 kernel 对象身份、JSON SHA256 与构建目录一致。
- 原有132例精度、对齐54例精度全部 SUCCESS，CPU 标杆精度标志为 true。
- 公开接口74项通过；确定性9项全部 SUCCESS 且确定性结果为 true。
- FP32/BF16 × V-first/K-first × 最终 State/逐 token 快照共8组原生 sanitizer：数值检查通过，mssanitizer 均报告 `No error detected.`；每组4个连续任务、8个 tile，快照场景各36次快照，均完成3次下一任务 State 预取。
- `operator-review` 的 issue 覆盖检查通过（188个 completed issue、21条规则）。静态脚本对现有 host 报错路径报告114条 DFX-MESSAGE 存量问题；本轮仅修改两个 arch35 kernel header，未新增这些 host 问题。

正式原始报告、manifest、msprof CSV、构建包、内存日志和静态检查 JSON 保存在本轮输出目录及 A5_1 对应目录。`docs/validation.md` 的有效内容已收敛到本记录，归档前删除开发期文件。
