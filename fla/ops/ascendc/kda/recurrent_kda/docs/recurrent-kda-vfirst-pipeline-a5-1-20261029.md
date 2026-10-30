# V-first 最大回退用例的 A5_1 流水

2026-10-29，按用户要求采集远程主线和当前实现的流水。算子实现、同步、tile 完成后批量搬出 out 的行为及受保护的方案文档均未修改。

## 用例与产物

从本轮144组对比结果中，按 V-first 的“当前/远程”耗时比选取最大值：issue680 ID7，B=32、H=16、HV=48、T=8、K=V=128，BF16 V-first State。packed TND、fused QKV view；辅助输入为 FP32，QK norm、raw gate、beta sigmoid 开启，原位并返回最终 State，无 indices。

远程版本锁定与性能对比相同的 `8c4b3691e92c6ac3685742eebe3314618c9bf53b`，相关475个源文件与冻结的标准构建匹配。当前版为已通过验证的方案对齐及 UB padding 优化版。本轮直接加载两版已安装的正式 kernel `.o`，没有重编译或插桩 device kernel；仅编译调用原 host processor 的 tiling 诊断程序。

| 项目 | 远程主线 | 当前实现 |
| --- | --- | --- |
| kernel 文件 | `RecurrentKda_0df854f6a4ff151bc791bef274e55eed.o` | 同名 |
| SHA256 | `b78ab0b0bf7e13a0292eafa7f8fe174a0cab966e4e2a8616f144ad43ce8da5b2` | `e22bc381b766627a61ea38104cfd87c6dfe0380760fe36badcf3d3657fa5700e` |
| 原模型 block 数 | 56 | 56 |
| vStep | 128 | 128 |
| 当前版 X | 不适用 | 8 |
| tiling 字节数 | 256 | 288 |

环境为 A5_1、CANN 9.2.0-beta.2、msopprof 26.2、Ascend950PR_957c simulator。原生真机冒烟使用物理卡7/逻辑卡0；仿真在独立 shell 清除 `ASCEND_RT_VISIBLE_DEVICES`，通过任务目录的 `libruntime.so -> libruntime_camodel.so` 加载仿真 runtime，未修改共享 CANN 安装。

## 原模型流水

使用 CPU generator 按性能 manifest 的同一 seed 重建输入，保留共享 QKV storage、offset 和 stride。全部 tensor 的 shape/stride/dtype 与性能 manifest 逐项相同。通过两版原始 `RecurrentKdaTilingProcessor::ProcessShapes/ProcessUb`，查询56 AIV、253952B UB 生成 tiling，随后用 `aclrtLaunchKernel` 加载原生产对象。

两版未插桩的原模型调用都达到 `TARGET_COMPLETE` 和 `HOST_FINALIZED`，out 与 State 均有限；两版 out/State 最大绝对差分别为 `3.051758e-5`、`7.324219e-4`。这里只检查入口与参数是否存在明显错误，不取代已经完成的 CPU 精度验收，也不要求两版 BF16 递推逐位相同。

```bash
msprof op simulator --soc-version=Ascend950PR_957c \
  --kernel-name=RecurrentKda --launch-count=1 --core-id=0 --timeout=3 \
  --aic-metrics=PipeUtilization,ResourceConflictRatio \
  --output="$root/$label-simulator" \
  "$python" "$root/native-launch.py" "$label"
```

两版原模型均在3分钟限时后停止并解析成功，未达到 `TARGET_COMPLETE/HOST_FINALIZED`；工具退出0不能解释为完整执行成功。core0.veccore0 导出的相对时间末端：远程17.614μs、当前18.185μs。它们是截断记录长度，不能比较其大小作为性能结果。图中共同展示前17.614μs，同时保留原始 trace。解析还包含 core0.veccore1；主图只展示 veccore0。

[原模型局部流水 PNG](assets/vfirst-pipeline.png) · [SVG](assets/vfirst-pipeline.svg)

图中分别展开 Scalar、MTE2、Vector 算术与控制、Vector 读 UB、Cast、写 UB、同步指令和 MTE3。读写包含输入、State、out 与临时数据，未按 UB 地址进一步区分。当前原模型片段没有记录 MTE3 搬出，不代表最终不搬出；该片段尚未完成首个8-token tile。空白也不能直接理解为所有流水都空闲。正式对象缺少源码行映射，本轮仅提供指令级流水。

## 完整递推诊断

为观察完整8-token递推、任务交接和最后搬出，补采相同正式对象的小规模诊断：B=2、H=HV=1、T=8、K=V=128、BF16 V-first State、单 AIV，当前 X=8，其余融合开关相同。仍由原 host processor 生成一致的 tiling，GM shape/stride 随小规模输入合法缩小，QKV fused token stride 为400元素。该图用于局部计算流程诊断，不能代表原模型56个 block 的性能或并发搬运。

两版正常设备调用和 simulator 调用均完整达到 `TARGET_COMPLETE/HOST_FINALIZED`，工具退出0。仿真包含2个连续任务、16个 token，并覆盖尾部排空。完整记录跨度为远程36.424μs、当前38.338μs；此时间只描述诊断程序，不能代替原模型计时。

```bash
msprof op simulator --soc-version=Ascend950PR_957c \
  --kernel-name=RecurrentKda --launch-count=1 --core-id=0 --timeout=2 \
  --aic-metrics=PipeUtilization,ResourceConflictRatio \
  --output="$root/$label-diagnostic" \
  "$python" "$root/diagnostic-launch.py" "$label"
```

[完整局部诊断 PNG](assets/vfirst-diagnostic-pipeline.png) · [SVG](assets/vfirst-diagnostic-pipeline.svg)

完整工作量下，动态指令统计如下。它们包含输入、递推和 out 处理，不能全部计为 State 操作；指令执行存在重叠，不把次数乘以延迟后相加作为总耗时。

| 指令 | 远程次数 | 当前次数 | 远程延迟中位数 ns | 当前延迟中位数 ns |
| --- | --- | --- | --- | --- |
| RV_VLD | 13348 | 6256 | 5.45 | 6.06 |
| RV_VST | 13318 | 4128 | 7.88 | 5.45 |
| RV_VCVT_F2F | 4736 | 10336 | 4.24 | 4.24 |
| RV_VMUL | 18496 | 18528 | 4.85 | 4.85 |
| RV_VCADD | 4128 | 4128 | 13.33 | 13.33 |

当前版读写指令显著减少，但 Cast 次数增加5600次（约2.18倍），乘法、归约次数基本相同。结合源码，当前版每个 token 以 BF16 State 在 UB 中保留结果，下一 token 重新读取后在寄存器中转 FP32；图中持续分布的 Cast 值得优先分析。该统计尚未按地址和调用点拆解，因此不能认定新增5600次全部来自 State，或量化其单独解释了多少回退。

## 真机证据与结论边界

图中附带的真机统计直接来自上一轮同卡性能对比的原始 CSV，每例去除3次预热后取5样本均值；本轮没有再重复大量计时。

| 指标 μs | 远程主线 | 当前实现 |
| --- | --- | --- |
| Task Duration | 401.9718 | 448.1864 |
| aiv_vec_time | 364.4656 | 432.4412 |
| aiv_scalar_time | 109.3810 | 16.3102 |
| aiv_mte2_time | 26.7602 | 15.6818 |
| aiv_mte3_time | 70.6456 | 6.9784 |

原模型耗时增加46.215μs（11.50%），Vector 时间增加67.976μs；Scalar 和搬运时间下降。计数器存在重叠，不能直接相加，但证据支持把后续定位重点放在 Vector 指令、Cast、输出收集和依赖链。当前证据不能将全部11.50%回退归因于某一类指令，也不能用两版原模型局部 trace 的长度比较性能。

## 结果位置

- 本机：`D:/PROJECT/flash-linear-attention-npu/output/rkda-vfirst-worst-pipeline-20261029/`。
- A5_1：`<run-root>/rkda-vfirst-worst-pipeline-20261029/`，保留完整 OPPROF 目录、正常设备输出、原始输入 storage、tiling 和日志。
- 本机证据包 `pipeline-evidence.tar.gz` 包含四轮导出 trace、两个 AIV 的原模型解析结果、指令 CSV、图、脚本、日志、绑定和 SHA256 索引；大体积正常调用输出 NPY 和输入 storage 留在服务器，未加入包。
- 初始 host 诊断链接及 metadata 过滤曾失败，均发生在正式采集之前；修正后四轮采集使用相同正式 `.o`。未重试此前已知失败的真机 PipeTimeline 路径。
