# K-first BF16 连续V方向验证

2026-10-30，基线为当前已验证kfp29。尝试kbf30，将K-first FP32连续V方向的两遍计算推广到BF16 State。公开接口、CPU标杆、值域和混合容差保持不变；受保护原方案不修改。

## 设计与验证计划

K-first BF16的UB行距为144个BF16元素（128个有效元素加32B间隙）。每个FP32寄存器承载64个连续V元素；两遍均从当前bank读BF16并在寄存器转换为FP32。第一遍顺序累计prediction，计算delta后第二遍衰减并更新State，同时用未舍入的FP32更新值累计out，最后Cast为BF16写另一bank。不得从已写回BF16的结果计算当前token的out。

State的UB读取从32KiB/token增加为64KiB/token，BF16→FP32转换量加倍；可省去成对Gather/Scatter、解包/打包位操作、逐行归约和四行out收集。GM流量、UB容量、bank生命周期、预取、必要同步和整tile计算完成后批量搬出out保持不变。K-first路径不再使用vFloat scratch，移除其专用写读与屏障；V-first仍保留原路径。推广通过State模板复用计算流程，FP32也重新回归。

先标准run构建安装、libcust_opapi校验、72对象审计、132原有+54对齐精度，再同卡msprof Task Duration计时。性能覆盖issue680全部36种形状的K-first BF16，另取ID7的K-first FP32及V-first两种State dtype作为控制；每组3预热+5正式样本。最终完成74公开接口、9确定性和当前Pipeline原生内存探针。收益不足则恢复kfp29，候选及结果保留在本记录。

## 实验结果：保留kbf30

本轮基线为已验证的kfp29，其K-first FP32已经采用连续V方向计算，K-first BF16仍为四行成对Gather/Scatter。以下是kbf30与kfp29的同卡重测，不是相对远程main或历史跨卡数据。

性能在A5_ci物理卡5/逻辑0采集；构建、精度、契约、确定性、原生内存探针及诊断启动在A5_1物理卡6/逻辑0执行。CANN 9.2.0-beta.2、Python 3.11.17、torch 2.9.0+cpu、torch_npu 2.9.0.post6；设备Ascend950PR。机器共享，采集脚本保留逐例前后npu-smi占用快照。

msprof Task Duration，每例3次预热、5次正式样本。目标为issue680全部36种形状的K-first BF16 State：B32/64/72、H16、HV32/48/128、T1/2/4/8、K=V=128；packed TND、fused QKV view、QKV/out BF16、gate/beta FP32，QK norm、raw gate和beta sigmoid开启，原位最终State、无indices，X按最大请求长度取1/2/4/8。另取ID7的V-first FP32/BF16和K-first FP32三组控制，同一构造、seed及API参数。

K-first BF16耗时比几何均值为**0.497276（-50.27%）**，36/36组正式均值更低；单组改善22.59%～53.87%，最大组内CV=1.17%，无CV>3%的组，未替换样本。

| T | 组数 | kbf30/kfp29几何均值 | 耗时变化 |
| --- | --- | --- | --- |
| 1 | 9 | 0.572883 | -42.71% |
| 2 | 9 | 0.487637 | -51.24% |
| 4 | 9 | 0.471967 | -52.80% |
| 8 | 9 | 0.463784 | -53.62% |

三个ID7控制组：

| 布局/State | kfp29（μs） | kbf30（μs） | 耗时变化 |
| --- | --- | --- | --- |
| V-first fp32 | 333.865 | 332.058 | -0.54% |
| V-first bf16 | 379.997 | 379.993 | -0.00% |
| K-first fp32 | 341.978 | 341.619 | -0.11% |

正式两版通过一个串行shell命令依次运行，前版进程完成后才启动后版；原始目标kernel时间戳间隔为19.821秒，无交叠。每版CSV为624行×48列，其中312条目标RecurrentKda、39组×8次；按Task Start Time映射manifest，每组剔除前3次，仅后5次取均值，设备5与56个block均校验。

初次`kbf-base29-full`/`kbf30-full`在版本切换处有约1.076秒采集交叠，属于采集调度错误，整对排除；原始样本和汇总保留，不混入最终串行结果。未针对个别case替换样本。本轮未重新测其它布局全部模型性能，不能把三个控制组推论为完整性能回归。

## 精度与工程验证

- 标准run构建安装、libcust_opapi加载/构建MD5一致。通用校验脚本未定位kernel目录，独立审计补全72个对象与JSON/build SHA核对。
- 132个原有精度、54个对齐精度全部通过；沿用原CPU标杆、输入值域和混合容差。
- 74项公开接口契约、9项确定性（每项3次）全部通过。
- K-first BF16最终State和逐token快照两组原生mssanitizer均`No error detected.`，out/State探针对比通过。每组4连续任务、T9、X8，覆盖跨tile、bank复用和下一任务State预取；对应追踪版分别完成49/157条事件，均`HOST_FINALIZED`。入口为使用当前生产Pipeline头文件编译的`state_pipeline_probe`（-O3、sanitizer开启），不是正式RecurrentKda包的mssanitizer报告。
- K-first FP32小规模正常启动的out、State与kfp29逐位一致；三个控制性能见上表。
- 当前18个实现/调用源文件审计一致，跨A5_1/A5_ci的两版本共156项so/o SHA256一致；下载证据逐文件核对大小和SHA256。
- 静态issue覆盖188项通过；仍有当轮114项host DFX-MESSAGE发现，两个修改的kernel头文件无发现。A2/A3未上板。受保护原方案SHA256未变。

## 完整小规模流水与指令

正式标准包原生对象，Ascend950PR_957c simulator，单AIV，B2/H=HV1/T8/K=V128、K-first BF16 State、X8，两任务共16 token，两版均`HOST_FINALIZED`。完整trace事件59585→57613，净减少1972（3.31%）；完整仿真区间56.941→30.027μs，仅作诊断，不作模型真机耗时。启动manifest已将spec/options的state_v_first设为false，并注明实际State dtype。

| 动态指令 | kfp29 | kbf30 |
| --- | --- | --- |
| RV_VMUL | 8320 | 4256 |
| RV_VMULA | 8192 | 12288 |
| RV_VCADD | 4128 | 32 |
| RV_VDUP | 2080 | 32 |
| RV_VGATHER2 | 2048 | 0 |
| RV_VSCATTER | 2048 | 0 |
| RV_VINTLV | 1536 | 0 |
| RV_VSELR | 512 | 8192 |
| RV_VADDS | 80 | 4176 |
| RV_VCVT_F2F | 8320 | 12416 |
| RV_SMEM_BAR | 80 | 64 |

收益来自State连续读写、prediction/out改为沿K向量累计，消除离散State Gather/Scatter、逐行归约与out收集依赖。两遍读取使BF16转换量上升、寄存器内系数Gather增加，因此总指令事件仅下降3.31%；不能把真机约50%的耗时下降直接等同于指令减少比例。剩余VCADD为q/k归一化。K-first不再写读vFloat scratch，专用VST→VLD屏障每token减少1条；必要的State/out同步、覆盖前等待与尾部排空均保留。

State UB有效读量32→64KiB/token、写量仍32KiB/token，UB容量和GM State流量不增加。所有State bank中只保存BF16结果；Load后在寄存器转FP32，两遍预测/递推和out累计均FP32。第二遍先用未舍入State累计out，再Cast_RINT为BF16写UB。两bank递推、可用第三bank预取及完整tile后批量搬出out保持原规则。

## 可复核证据和命令

本机`output/rkda-kfirst-bf16-register-20261030/{a5-1,a5-ci}`保存正式样本、占用、CSV、完整小规模trace、构建/精度/契约/确定性/内存日志和源/对象哈希；两个zip自带逐文件SHA256索引。基线29和最终kbf30头文件快照在本地output根目录。服务器沿用`rkda-layout-register-optimizations-20261030`，分别位于A5_1的`<run-root>/`和A5_ci的`<run-root>/`。

```bash
# A5_1
bash build-round.sh kbf30
bash kbf-accuracy.sh kbf30 atk_recurrent_kda
bash kbf-accuracy.sh kbf30 atk_recurrent_kda_alignment
bash kbf-final-checks.sh kbf30
python kbf-verify-final.py kbf30
bash kbf-simulator.sh kfp29 bf16
bash kbf-simulator.sh kbf30 bf16
bash kbf30-build-memory.sh
bash kbf30-run-memory.sh
# A5_ci：同一命令严格串行，物理卡5
bash kbf-perf.sh kfp29 39 kbf-base29-serial && \
bash kbf-perf.sh kbf30 39 kbf30-serial && \
python compare.py kbf-base29-serial kbf30-serial
```

## 36组目标性能

单位μs，5个正式样本均值；变化=(kbf30/kfp29−1)×100%。

| ID | B | HV | T | kfp29 | kbf30 | 耗时变化 |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 32 | 32 | 1 | 74.456 | 40.382 | -45.76% |
| 1 | 32 | 32 | 2 | 135.246 | 68.203 | -49.57% |
| 2 | 32 | 32 | 4 | 257.098 | 123.452 | -51.98% |
| 3 | 32 | 32 | 8 | 501.771 | 234.598 | -53.25% |
| 4 | 32 | 48 | 1 | 106.928 | 56.023 | -47.61% |
| 5 | 32 | 48 | 2 | 196.542 | 96.783 | -50.76% |
| 6 | 32 | 48 | 4 | 376.835 | 178.862 | -52.54% |
| 7 | 32 | 48 | 8 | 737.222 | 342.829 | -53.50% |
| 8 | 32 | 128 | 1 | 272.521 | 142.949 | -47.55% |
| 9 | 32 | 128 | 2 | 509.602 | 245.264 | -51.87% |
| 10 | 32 | 128 | 4 | 984.778 | 462.400 | -53.05% |
| 11 | 32 | 128 | 8 | 1935.627 | 895.206 | -53.75% |
| 12 | 64 | 32 | 1 | 141.435 | 75.079 | -46.92% |
| 13 | 64 | 32 | 2 | 259.060 | 127.324 | -50.85% |
| 14 | 64 | 32 | 4 | 497.216 | 235.588 | -52.62% |
| 15 | 64 | 32 | 8 | 973.079 | 452.978 | -53.45% |
| 16 | 64 | 48 | 1 | 206.253 | 107.297 | -47.98% |
| 17 | 64 | 48 | 2 | 382.254 | 185.861 | -51.38% |
| 18 | 64 | 48 | 4 | 735.584 | 346.665 | -52.87% |
| 19 | 64 | 48 | 8 | 1443.029 | 668.561 | -53.67% |
| 20 | 64 | 128 | 1 | 537.949 | 408.958 | -23.98% |
| 21 | 64 | 128 | 2 | 1009.645 | 485.241 | -51.94% |
| 22 | 64 | 128 | 4 | 1953.373 | 914.199 | -53.20% |
| 23 | 64 | 128 | 8 | 3841.913 | 1773.217 | -53.85% |
| 24 | 72 | 32 | 1 | 158.899 | 83.286 | -47.59% |
| 25 | 72 | 32 | 2 | 293.438 | 143.578 | -51.07% |
| 26 | 72 | 32 | 4 | 563.406 | 266.225 | -52.75% |
| 27 | 72 | 32 | 8 | 1102.923 | 512.336 | -53.55% |
| 28 | 72 | 48 | 1 | 230.085 | 120.440 | -47.65% |
| 29 | 72 | 48 | 2 | 429.300 | 207.861 | -51.58% |
| 30 | 72 | 48 | 4 | 827.525 | 389.299 | -52.96% |
| 31 | 72 | 48 | 8 | 1624.934 | 752.052 | -53.72% |
| 32 | 72 | 128 | 1 | 602.385 | 466.328 | -22.59% |
| 33 | 72 | 128 | 2 | 1132.409 | 542.910 | -52.06% |
| 34 | 72 | 128 | 4 | 2192.482 | 1024.757 | -53.26% |
| 35 | 72 | 128 | 8 | 4313.081 | 1989.751 | -53.87% |
