# PR #997 元数据全设备同步修复

## 范围与设计

检视意见 discussion_r4430769129 指出：原版本 `ecc69ea6` 在读取设备 CU、indices/accepted 前显式等待整个设备，导致无关 stream 参与 value-dependent host tiling 的等待。本轮只修改算子私有 op_api，公开 ABI、设备 kernel、最大请求长度/X 选择、CPU 标杆、输入值域和精度阈值不变；冻结方案不修改。

GetWorkspaceSize 没有执行 stream 参数，不能从该接口可靠推断任意原始 ACLNN 调用的生产者 stream。既有契约要求直接 ACLNN 调用方保证元数据生产者就绪；Python stable/ctypes 已在入口等待 PyTorch 当前 stream，保留此正确性依赖。跨 stream 生产者须先向当前 stream 建立依赖，不以设备全局同步隐式补足用户依赖。

最终保留同步 `aclrtMemcpy` D2H，移除 CU 及 State 元数据读取前的两个显式全设备同步。算子局部 `MetadataTransfer` 只封装复制及 opbase 日志，不新建 stream、event 或 pinned staging；host placement 元数据继续直接读取。executor 最终仍持有自己的元数据快照，既有值校验和 value-dependent cache 策略保持，不能为了缓存命中跳过可变值校验。

该结构消除显式全设备等待，仍有当前生产者、D2H 和 host 校验的等待。CU 快照需先完成才能推导活跃 indices/accepted 切片；本轮不减少合法性检查，也不宣称完全异步 enqueue。暂不扩展 CPU Tensor 的 Python descriptor 支持，不改变其他算子的 runtime。生产者完成与 D2H 提交由 host 按序完成，复制返回后才访问目标内存。runtime 内部复制是否等待其他 stream 必须由隔离实测确认，不能由删除 API 的源码推断。

## 排除的候选

先实测独立 ACL stream + pinned staging + 异步 D2H：A5 上无关 stream 的完成 event 在5次调用返回时均未完成，隔离成立；但每次调用创建/销毁 stream 和分配/释放 pinned memory，使 ID0 单调用端到端从121.44升至526.18 μs，8次连续调用从793.85升至4911.15 μs。该候选未采用，原始构建和测量保留，最终代码不含这套资源分配。

## 验证计划

- 两平台仅重建受影响 cust_opapi，核对实际安装库及72个设备对象不变，保存旧库作 `ecc69ea6` 对照。
- A5/A2 运行20项元数据就绪、74项公开契约、59正常/49负向三入口回归；原始 ACLNN 测试显式履行元数据就绪契约，不能借删除同步碰巧通过。
- A5 选择 issue680 ID0、ID7、ID35，same-card old/new，每组3预热、5正式，msprof 记录目标 kernel；另外独立记录 host enqueue、同步完成的端到端及连续 decode/两个 head stream 的 makespan。
- 多流专项在无关 stream 上排入持续工作，检查该 stream 的完成 event 在当前调用返回时是否仍未完成；与原实现对照。只以实际运行证据判断跨流隔离，不能因源码删除一个 API 就认定隔离。
- 保留所有样本、共享设备状态、构建库 hash、输入规格及原始 JSON；不使用 host 时间替代 msprof kernel 性能口径。

## 结果

两平台最终 cust_opapi 重建安装并与构建库逐字节匹配；18个原 host/kernel 源文件中仅本轮 op_api 修改，72个设备对象分别与各平台旧对照一致。未重跑前轮188/234项完整矩阵或 sanitizer；本轮实际执行：

| 检查 | A5_1 / Ascend950PR / CANN 9.2.0-beta.2 | A2 / 910B3 / CANN 9.3.0 weekly |
| --- | --- | --- |
| 框架元数据就绪 | 20项通过 | 20项通过 |
| 公开契约、mutation/grad | 74项通过 | 74项通过 |
| stable/ctypes/原始 ACLNN | 59正常、49负向通过，正常项比较 CPU out/State | 59正常、49负向通过，正常项比较 CPU out/State |
| 原始 ACLNN EZ 码及拒绝后恢复 | 16负向、4条 CPU 健康路径通过 | 16负向、4条 CPU 健康路径通过 |

原始 ACLNN 回归显式等待生产者 stream；Python 元数据专项不手工预同步，验证 wrapper 自身的依赖。报错、快照、空请求、元数据变更和非连续 view 均保留原标准，CPU 标杆和混合容差未改。

### A5 四路径 kernel 计时

旧对照为本轮修改前 `ecc69ea6`，不是远程 main。同一物理卡5串行 old→new，issue680 ID0/7/35 × 四路径，每组3预热、5正式。`msprof` 的两份 op_summary 各192行×48列；目标各96条（12组×8），正式各60条，均为 device5、56 block。逐项输入 spec/options/shape/stride/dtype 相同，所有样本保留。

| ID | 路径 | 修改前 μs | 修改后 μs | 变化 |
| --- | --- | --- | --- | --- |
| 0 | V-first FP32 | 38.134 | 37.366 | -2.01% |
| 7 | V-first FP32 | 289.309 | 288.798 | -0.18% |
| 35 | V-first FP32 | 1665.297 | 1665.101 | -0.01% |
| 0 | V-first BF16 | 41.541 | 41.118 | -1.02% |
| 7 | V-first BF16 | 342.835 | 343.017 | +0.05% |
| 35 | V-first BF16 | 1993.380 | 1993.059 | -0.02% |
| 0 | K-first FP32 | 40.869 | 41.096 | +0.56% |
| 7 | K-first FP32 | 333.801 | 332.698 | -0.33% |
| 35 | K-first FP32 | 1922.391 | 1922.050 | -0.02% |
| 0 | K-first BF16 | 39.624 | 39.616 | -0.02% |
| 7 | K-first BF16 | 331.435 | 330.958 | -0.14% |
| 35 | K-first BF16 | 1922.821 | 1921.914 | -0.05% |

12组耗时比几何均值 0.997326；最大组内 CV 2.59%。设备代码未改，这些小差异按计时波动解释，不认定 host 修改提高了计算性能。未因此重测全部144组最新 main 对照。

### 调用链、多流与连续调用

与 kernel profiling 分轮运行：V-first BF16，ID0/7/35，当前 stream 单调用、8次连续原位调用、两个独立 head stream/host thread 各8次调用。每组3预热、5正式，使用 `perf_counter_ns` 记录 enqueue 及当前 stream 完成的端到端时间；其口径不替代 msprof。第一轮 old→new 全样本保留；随后对 head host 线程也进行3次预热，new→old 反序控制，避免线程首次初始化混入计时。两轮的现象和最终控制均记录，不合并成更大样本。

下表为第二轮已预热 host 线程的5样本均值，单位 μs；decode8 和双 head 是一组调用的总时间，非单 kernel。设备允许共享，不承诺每个小差异具有统计显著性。

| ID | 模式 | 前 enqueue | 后 enqueue | 前端到端 | 后端到端 | 端到端变化 |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | single | 72.66 | 67.10 | 129.28 | 127.06 | -1.71% |
| 0 | decode8 | 821.28 | 834.53 | 877.43 | 894.41 | +1.94% |
| 0 | two_heads | 1299.10 | 1366.67 | 1360.07 | 1439.04 | +5.81% |
| 7 | single | 89.97 | 84.40 | 459.45 | 447.74 | -2.55% |
| 7 | decode8 | 3071.80 | 3076.88 | 3430.31 | 3440.88 | +0.31% |
| 7 | two_heads | 5377.27 | 4983.53 | 6038.72 | 5579.53 | -7.60% |
| 35 | single | 104.11 | 97.82 | 2148.14 | 2145.91 | -0.10% |
| 35 | decode8 | 14835.71 | 14900.41 | 16865.42 | 16945.25 | +0.47% |
| 35 | two_heads | 29203.86 | 28082.85 | 32811.72 | 31969.36 | -2.57% |

小负载并发仍受 host 调度与同步 D2H 影响：ID0 的双 head 控制均值约增加5.81%，不能声称所有场景吞吐都提高。单调用和连续调用的少量上下波动不能掩盖或替代下面的跨流隔离证据。

隔离专项在无关 stream 排入128次4096×4096 FP16 matmul，记录该 stream 的完成 event；当前 stream 没有等待此 event。调用返回时查询 event，查询后才显式排空后台工作：

| 对照 | 5次调用返回时无关 stream 已完成 | enqueue 范围 |
| --- | --- | --- |
| 修改前 | 5/5 | 47.207–57.724 ms |
| 修改后 | 0/5 | 0.431–1.827 ms |

第一轮同样为修改前5/5完成、修改后0/5完成；结合两次反序结果，确认当前 A5/CANN 组合不再被算子中的显式全设备同步拖到无关工作结束。该证据说明等待范围缩小，不证明其他 CANN/SoC 的 runtime 内部复制永不串行；A2 本轮完成正确性/报错回归，未做相同多流性能量化。

## 复核、归档与限制

公开签名与已发布 tag/原 PR base/latest main 的 AST 比较通过，issue corpus 188项/21规则通过；原有9项框架内部诊断、12项性能命令误识别及冻结方案2个本地链接不因本轮发生变化。新增复制失败沿用真实 opbase 日志、保留 runtime errno，不新增伪造参数错误。独立 stream 候选已因开销拒绝，最终无其创建/释放失败路径。

私有证据标识 `output/rkda-metadata-stream-20261031/`：两个压缩包均核验 CRC、逐文件 SHA manifest 和最终源码 hash。它们不是公开下载地址。

| 工件 | SHA256 |
| --- | --- |
| a5-evidence.zip | `f3ee7b2636127501ac7e647a126256192245dc1f52c93236ddecdc68e968a182` |
| a2-evidence.zip | `cfac06039da38e97b666a1dd4fddc412ab1ebb7227ae4615794b3f51ed3a8e56` |

本轮只修复显式全设备等待，不取消 host 值依赖/缓存禁用，不添加 CPU metadata 的 Python descriptor 接口，不把 host 时间写成 kernel 时间。A3、旧 CANN、正式当前 head CI、内部 runtime 故障注入未执行；共享设备和 host 线程抖动限制保留。
