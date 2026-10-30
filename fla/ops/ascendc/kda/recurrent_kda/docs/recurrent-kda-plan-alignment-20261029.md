# Recurrent KDA 方案对齐记录（2026-10-29）

实现基线：PR #997，`d23fb9e84fb410fb857d299686b679302a32d794`。
目标方案：主工作区 `recurrent-kda-a5-regbase-redesign-20261026.md`（2026-10-29 修订）；本轮不修改该方案。

本轮范围：未启用 raw gate 时忽略专用参数；按最大有效序列长度选择 X（上限8）；根据 bank 生命周期分配递推和预取角色；K-first R=4 和 token 级寄存器 out 收集；快照路径跨任务首 tile 输入预取；BSND 独立 batch stride。公开 ABI 和 CPU recurrence 公式不变。A2/A3 共用新的模式和 gate 语义，硬件路径仍独立。

Stage 仍为单 AIV Vector Stage；开发参考 `CHUNK_DEPENDENT/V1`，来源 `a14961e9dfe8c8264768d67fb447101873da4ecf`，仅采用依赖、所有权和同步闭环方法。State 保持接口 dtype/layout，out 在完整 token 后两次 Store，tile 完成后批量 MTE3。

## 实现对应关系

| 方案要求 | 本轮实现 |
| --- | --- |
| 无 A_log 时忽略专用参数 | Python 两条入口及 ACLNN 在校验、连续化前丢弃 dt_bias，并归一化 safe_gate/lower_bound；共用 tiling 跳过相应 shape/dtype/format 检查 |
| 按 Lmax 选择 X，最大8 | 定长读取 S；A5 packed 读取 INT32/INT64 cu_seqlens 的相邻差分最大值，检查首项、单调性及 capacity；选不超过 clamp(Lmax,1,8) 且满足 UB 的最大候选；全空 X=1 |
| 动态 bank 生命周期 | `FREE/RECURRENCE/PREFETCHED/MTE3_SOURCE` 记录所有权；`BeginBanks/FreeBank/ReleaseStateDma` 分配角色，下一任务初态 bank 不按任务号固定轮换 |
| K-first 与 V-first 都是 R=4 | 四个 V 行按同一数学公式更新；K-first FP32 Gather/Scatter，BF16 以相邻 V 行打包成32位访问；UB 不改变布局或增加 FP32 State 副本 |
| 输出收集及搬出 | 两个64元素 FP32 寄存器收集完整 token，两个 Cast/Store 写 UB；`CopyOutputTile` 仅在 tile 完成后批量提交 MTE3 |
| 跨任务预取 | 两条调度路径均在当前最后 tile 预取下一任务首 tile 输入；空闲 bank 预取下一任务 State 和参数；仍被旧末态 MTE3 占用的 bank 在回收后才复用 |
| BSND 独立 batch stride | 私有 tiling 新增 query/key/valueBatchStride，A5 与通用 kernel 分别按各自 stride 定位 batch；公开 ABI 不变，host/kernel 必须配套重新构建 |

## 变长调用链的代价与兼容性

公开 ABI 仍接受 device Tensor。手写 ACLNN 的 `.ValueDepend(OPTIONAL)` 和 `TilingInputsDataDependency`
不会自动把该 Tensor 的内容复制到 host，因此 A5 `MaterializeCuSeqlens` 在第一阶段调用
`aclrtSynchronizeDevice`，复制包含 stride 的元数据 storage span，按 view stride 收集到 executor
拥有的 host Tensor，再由框架向 kernel 提供本次边界。使用 `GetData()` 定位逻辑 view，支持非零 offset。

设备 Tensor 的值不参与原有 L2 descriptor 缓存键。若复用缓存，shape 相同而 cu_seqlens 改变时会返回
旧 executor，并继续使用旧 host 边界及 X。本轮 A5 packed 路径跳过该缓存查询并 `AbandonCache`；
固定长度及 A2/A3 保留缓存路径。元数据的值仍参与底层 tiling 的值依赖。

上述同步、D2H、host 扫描、executor 重建及 host 常量下发均属于调用链成本，不能用 kernel
Task Duration 代替。packed 模式当前不支持在 ACLGraph 捕获中调用；改变边界必须重新创建 executor，
不能复用旧 executor。固定长度不增加这次同步或复制。原有“host 不读取、任意改变 cu_seqlens 后图回放”
的描述不再适用于 A5 packed 路径。

## 验证口径

已按标准 ascend950 单算子 run 包构建并安装独立 venv；设备为 A5_ci Ascend950PR物理卡5，CANN 9.2.0-beta.2。CPU recurrence、值域和 mixed_tolerance_bm 标准不变。完整逐项结果和36组性能表见 [ATK README](../../../../../../tests/atk/recurrent_kda/README.md#方案对齐验证2026-10-29)。

`atk_recurrent_kda_alignment.json` 的精度映射：0–23 为两种 State dtype/layout × 快照开关；
24–35 为 Lmax=1/2/4/8（物理 capacity 大于有效总长，辅助输入 FP16）；36–38 为全空；
39–44 为两种布局的独立 batch stride；45–50 为无 A_log 时忽略参数；51–53 为定长32。
每个结构3个固定种子。另在公开入口脚本回归相同容量、变化边界以及非连续元数据，防止 L2 缓存问题复发。


## 本轮结果与边界

- 正式 CPU 双输出精度132/132、新增对齐精度54/54、公开入口74项及 mutation/grad 通过。
- 原生8变体生命周期 trace、2个实际启用 sanitizer 的 memcheck、16项元数据值依赖检查通过；确定性9例各3轮通过。
- 36个 issue680 case 使用 msprof，每例3次预热、5个有效样本；对已提交版的耗时比几何平均1.000729（+0.073%），范围−2.51%～+6.28%。最差 ID12，不能声明逐例无回退。
- packed host API 在静止队列的小样本检查中约69μs，fixed约28μs；packed 的全局同步可能等待其他任务，不能以该结果推断并发开销。kernel Task Duration 不包含这部分成本。
- A2/A3 仅完成共用 kernel 的 dav-c220-vec FP32/BF16 交叉编译，未完成设备运行验证。packed ACLGraph 捕获限制见上文。
- 方案原文 SHA256 保持 `78f03d56724f4ff6ba36609f1f53dbc05148825b94ee7d253a7867c8c331d5b4`。

## 静态检视待办

全算子机械检视仍报告114项 DFX-MESSAGE，未达到零告警：原有 generic OP_LOGE/CHECK_RET/CHECK_COND 缺少结构化参数报告。
相对提交基线，归一化目录类误报后新增净4项：元数据复制/同步/分配失败的3个通用 guard，以及 tiling 参数错误的净1项。
运行时错误包含状态码并返回失败，但未全部接入结构化 reporter；后续应按错误码规范分别补充实际参数、期望值和原因。
本轮修复两个原有 README 断链；未改保护方案内的原始引用。静态检查结果不作为功能正确性的替代证据。
