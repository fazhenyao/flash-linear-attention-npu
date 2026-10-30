# K-first 最大相对劣化用例的 A5_1 局部流水

2026-10-29。在 A5_1 使用 `msprof op simulator` 得到两版局部指令流水。两版均保持原模型尺寸、原始56个 block 和已验证的生产 kernel 对象；未修改算子或方案文档。每版只启动一次目标 kernel，采集窗口为3分钟，均超时停止并成功解析局部数据，**不是完整 kernel 流水**。

## 用例与版本

- issue680 ID7：B=32、H=16、HV=48、T=8、K=V=128；FP32 K-first State，QKV/out 为 BF16；packed TND，QKV 为共享存储的 fused view。
- QK norm、raw gate、beta sigmoid 开启；原位更新并返回最终 State；无 indices、无逐 token 快照。
- 远程基线为本轮对比开始时最新主线 `d77c37382aa8cf53e25005615c8c3f761fc137d2` 对应实现；当前版为 PR #997 的 `d23fb9e8` 加方案对齐改动。版本核验沿用测试 README 的72例对比记录。
- 此前 A5_ci 真机 msprof Task Duration：远程990.0466 μs、当前4669.5036 μs，当前/远程=4.716448。该数据用于选择最差用例；本轮仿真时间不能替代此正式计时，也不是 A5_1 真机性能对比。

| 项目 | 远程基线 | 当前实现 |
| --- | --- | --- |
| kernel SHA256 | `4d84b38135adc8887557f424a2d09fbe86e1cd3a5e0f3a7fbcbf81886f534a3f` | `52e4af4f890b650830017b96e9c1ea33097f37ef88ff01fa00b76c478bb8ca7c` |
| blockDim | 56 | 56 |
| 原处理器生成的 vStep | 64 | 128 |
| 当前版 X | 不适用 | 8 |
| tiling 字节数 | 256 | 288 |

## 采集方法与限制

环境为 A5_1 `/home/fazhenyao/runs/rkda-a5-1-perf-20261028-1/env.sh`，CANN 9.2.0-beta.2，msopprof 26.2.0-f7e04b75a861916137c51cb9242a2ba123b6d94a。

真机物理卡4的公开入口和直接对象入口 PipeTimeline 采集均未得到有效 trace，停留在目标 kernel 启动阶段。设备日志曾出现 `prof_aicpu_grp` 查询失败、`Failed init profiling(-1)`；尚未确认这是唯一原因。本轮没有重置设备或更改共享环境，失败采集均已停止。

改用 `Ascend950PR_957c` simulator，清除真实设备可见性变量，在任务独立目录设置仿真 runtime。输入由原生成器在 CPU 上以同一 seed 生成；原生成器本身先在 CPU 生成随机值。保存 fused QKV 的完整共享存储及偏移，核验全部输入 shape/stride/dtype、属性与原性能 manifest 相同。

分别编译两版原有 `RecurrentKdaTilingProcessor` 的诊断调用程序，查询目标平台56个 AIV、253952 B UB，通过原 `ProcessShapes` / `ProcessUb` 生成 tiling；未手工缩小工作量。直接用 `aclrtLaunchKernel` 加载原生产 `.o`。两版未插桩的真机调用正常完成，最终 State 均有限，两版最大绝对差为 `9.313226e-8`；这只用于排除明显的启动参数问题，不代替完整精度验收。

```bash
msprof op simulator --soc-version=Ascend950PR_957c \
  --kernel-name=RecurrentKda --launch-count=1 --core-id=0 --timeout=3 \
  --aic-metrics=PipeUtilization,ResourceConflictRatio \
  --output="$root/$label-simulator" \
  "$python" "$root/rkda-kfirst-native-launch.py" "$label"
```

`--core-id=0` 仅限制结果解析，未改变56个 block 的调度。导出了 core0 的两个 AIV，配图展示 veccore0。两版都在3分钟超时后停止，应用状态139，工具解析成功；工具的 `All task success` 只表示采集解析流程成功，不能解释为 kernel 完整完成。

## 局部观察

veccore0 的已导出区间：远程25.036 μs、当前19.418 μs。这是超时前记录长度，不是两版 kernel 耗时，不能比较大小判断性能。当前版尚未覆盖首 token 全部 State 行，不能据此分析完整 tile、任务交接、稳态吞吐或尾部排空。两版生产二进制缺少 debug_line，暂无源码行映射。

| 已记录指令 | 远程延迟中位数 | 当前延迟中位数 | 当前最长 |
| --- | --- | --- | --- |
| RV_VGATHER2 | 本片段未记录 | 258.8 ns | 267.3 ns |
| RV_VSCATTER | 本片段未记录 | 563.3 ns | 935.2 ns |
| RV_VLD | 5.5 ns | 6.7 ns | 315.2 ns |
| RV_VST | 7.3 ns | 本片段未记录 | — |

上述统计来自不同局部执行阶段，读写指令的数据、组织及访问模式不同，不能视为等工作量微基准或把延迟比直接当作整体性能比。指令会交叠，持续时间不可相加。

当前片段持续存在长延迟 Gather/Scatter，且与 MTE2 搬入交叠。源码也显示当前 FP32 K-first State 使用跨128个元素步长的离散读写；基线先做布局转换后执行连续递推。下一步应优先验证离散 State 读写的资源压力、与搬入的竞争及依赖，而非仅减少 Scalar 指令。**尚不能据本次流水确认 UB bank 冲突，也不能量化它解释了4.716倍回退中的多少。**

## 结果位置

- A5_1：`/home/fazhenyao/runs/rkda-kfirst-worst-pipeline-20261029/`。
- 本机：[中文流水图](/D:/PROJECT/flash-linear-attention-npu/output/rkda-kfirst-pipeline-a5-1/kfirst-pipeline.png)、[原始证据包](/D:/PROJECT/flash-linear-attention-npu/output/rkda-kfirst-pipeline-a5-1/pipeline-evidence.tar.gz)、[统计 JSON](/D:/PROJECT/flash-linear-attention-npu/output/rkda-kfirst-pipeline-a5-1/pipeline-summary.json)。
- 本机目录含两版 `trace.json`、两个 AIV 的指令 CSV、日志、tiling、启动脚本和文件哈希索引。CSV 各7列，远程3639行、当前1886行；CSV 的行是聚合指令位置，不能当作动态执行次数。
