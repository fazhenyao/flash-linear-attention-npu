# K-first FP32 寄存器流程优化

2026-10-30，以已验证layout26为基线。保持接口、CPU标杆、值域和混合容差，受保护原始方案不修改。只实验A5 K-first FP32寄存器计算，不改变UB布局、State bank生命周期、必要同步、预取和整tile计算后批量搬出out。

## 候选与验证计划

前轮全面融合点积使K-first FP32典型耗时增加1.18%；更早单独融合State更新约增加0.6%。本轮不直接重复全面融合，先区分prediction和out的关键依赖。

1. kfp27：仅融合out点积，prediction和State更新保留原流程。每个四行块减少4条Vector算术，每token静态减128条；FMA改变舍入，需要原CPU标准验证。Scatter可与out归约重叠，是否获益以编译指令和真机为准。
2. 若第一项无收益，单独评估prediction融合；再评估四行块内两行一组的计算，缩短State/乘积寄存器生命周期。后一项保持四行out收集及外层循环，不把源代码变短当作指令减少。

每候选标准run构建安装、动态库与72对象审计，132原有+54对齐精度通过后采性能。性能使用A5_ci同卡msprof Task Duration，3预热+5正式样本；典型issue680 ID0/1/2/3/7/11/23/35的K-first FP32覆盖短长序列和较大HV，并保留3组其它布局/类型控制。最终采用候选覆盖全部36组K-first FP32。样本、设备占用、二进制身份和完整小规模simulator动态计数归档，不混用真机与仿真耗时。

## 实验结果

kfp27通过132+54精度，但典型8组K-first FP32全部回退，几何均值+0.53%，放弃。完整16token FP32 simulator指令61762→59714，正好少2048条（128/token），区间93.012→93.256μs；减指令不代表更快。

kfp28只将四行块分成两对、保留原算术及四行out收集。精度通过，典型全部回退、几何均值+2.01%；动态事件62420，较基线反而多658条，放弃。

### kfp29：连续V方向的两遍计算

以原K-first UB物理布局为准，寄存器lane对应64个连续V元素。第一遍按K顺序累积全部128个V的prediction：提前计算两个64列的`alpha*k`，每K位置广播对应系数，以连续State Load和FMA累计。计算`delta=beta*(v-prediction)`后，第二遍再读原State，逐K进行衰减、FMA更新和out累计，并连续写另一State bank。两遍始终读原bank，避免读到本token更新后的数据；out完整保存在两个FP32寄存器，仍在token末Cast/Store、tile末MTE3。

该路径不新增UB或GM workspace；两遍使State的UB读取量加倍，但消除State Gather/Scatter、128行prediction/out归约及四行out收集，delta也由逐行广播改为两个向量计算。系数从原q/k/alpha寄存器按lane Gather广播。顺序累加、alpha*k重排及FMA会改变舍入，原CPU标杆与阈值是采用门禁。其余布局/dtype仍走原四行流程。以下通过实际动态指令和真机计时验证，不仅按连续访存推断收益。


## 最终结果：采用kfp29

A5_ci物理卡1/逻辑0，Ascend950PR、CANN 9.2.0-beta.2、Python 3.11.17、torch 2.9.0+cpu/torch_npu 2.9.0.post6。相对本轮开始的layout26同卡重测；不是相对远程main或历史跨卡数据。msprof Task Duration，每例3预热+5正式样本。全部56个block，按Task Start Time映射目标记录，39组/版本、312条RecurrentKda/版本；原CSV为624行×48列，非目标算子不计入性能。

目标为issue680全部36种形状的K-first FP32 State：B32/64/72、H16、HV32/48/128、T1/2/4/8、K=V=128；packed TND、fused QKV view、QKV/out BF16、gate/beta FP32，QK norm/raw gate/beta sigmoid开启、原位最终State、无indices，X按最大请求长度选择1/2/4/8。另保留ID7的V-first FP32/BF16、K-first BF16三组控制。与基线使用同一输入构造、seed、API参数、预热和统计方式，机器共享，保留逐例前后占用快照。

36组目标耗时比几何均值**0.407131（-59.29%）**，36/36正式均值更低，最大组内CV=1.80%，无CV>3%的组，没有替换样本。短序列大HV组只有约1.8%～4.9%的收益，不能将几何均值解释为每例改善59%。

| T | 组数 | kfp29/layout26几何均值 | 耗时变化 |
| --- | --- | --- | --- |
| 1 | 9 | 0.758691 | -24.13% |
| 2 | 9 | 0.456827 | -54.32% |
| 4 | 9 | 0.287643 | -71.24% |
| 8 | 9 | 0.275592 | -72.44% |

控制组ID7变化：V-first FP32 +0.61%，V-first BF16 -0.15%，K-first BF16 +0.09%。只对这3组采集了控制性能，不能声称完成其它布局所有模型性能回归；这几条通路的算术流程保留原实现。

## 完整小规模指令与正确性

A5_1，正式标准包的原生对象，Ascend950PR_957c simulator，单AIV、B2/H=HV=1/T8/K=V=128、K-first FP32 State、X8，两任务共16 token，两版都达到HOST_FINALIZED。总trace指令事件61762→45336，净减少16426（26.60%）；完整仿真区间93.012→31.187μs，仅用于诊断。

| 动态指令 | layout26 | kfp29 |
| --- | --- | --- |
| RV_VMUL | 18528 | 4256 |
| RV_VADD | 8288 | 96 |
| RV_VMULA | 0 | 12288 |
| RV_VCADD | 4128 | 32 |
| RV_VDUP | 2080 | 32 |
| RV_VGATHER2 | 4096 | 0 |
| RV_VSCATTER | 4096 | 0 |
| RV_VINTLV | 1536 | 0 |
| RV_VSELR | 512 | 8192 |
| RV_VADDS | 80 | 4176 |
| RV_SADD | 5280 | 224 |
| RV_SADDI | 3680 | 96 |
| RV_SMEM_BAR | 80 | 64 |

State Gather/Scatter改为连续VLD/VST；新增寄存器内系数Gather不能与原UB离散Gather混为同一访存。prediction/out的逐行归约消除，剩余VCADD来自q/k归一化。必要State/out同步及预取闭环保留；K-first FP32不再写/读vFloat scratch，因此每token少一条该scratch专用VST→VLD屏障，SMEM_BAR减少16条。State的UB读取量由64KiB/token增为128KiB/token，写入仍64KiB/token；GM State搬入搬出和UB容量不增加。

诊断输入由前轮小规模存储复用，启动脚本将State BF16位模式扩成FP32、设置stateVFirst=0并选择FP32正式对象；binding内原始inputs描述仍保留源数据BF16/V-first元信息，实际类型与布局以转换代码、tiling和对象dtype为准，不把原描述当作本轮输入类型证据。

标准run构建安装、libcust_opapi校验、72个对象与JSON/build SHA审计通过；当前18个实现/调用源文件核对通过；layout26及3个候选共312项so/o跨A5_1与A5_ci SHA256一致。kfp27/28/29各通过132原有+54对齐精度；最终kfp29通过74项公开接口、9项确定性（每项3次）。沿用原CPU标杆、值域和混合容差，顺序累计及FMA不要求与基线逐位一致。

另用当前生产Pipeline头文件编译-O3、启用sanitizer的原生探针，在A5_1卡7对K-first FP32最终State和逐token快照各执行一组mssanitizer memcheck，均No error detected，out/State探针对比通过。每组4个连续任务、序列长9、X8，覆盖跨tile、bank复用与下一任务预取；对应追踪版分别完成49/157条事件记录，均HOST_FINALIZED。探针入口为state_pipeline_probe，不冒充正式RecurrentKda对象的mssanitizer报告。

静态issue覆盖188项通过；全算子仍有当轮114项host DFX-MESSAGE发现，两个修改的kernel头文件无发现。A2/A3未上板；此次只修改arch35。原始受保护方案SHA256保持不变；正式design.md通过本补充描述当前K-first FP32特殊流程。

## 可复核证据与命令

本机`output/rkda-kfirst-fp32-register-20261030/{a5-1,a5-ci}`含原始正式样本、manifest、占用、CSV、trace、构建/安装/精度/确定性/内存日志、二进制与源码哈希。各zip自带逐文件SHA256索引，下载后逐项校验。服务器仍使用`rkda-layout-register-optimizations-20261030`run目录，所有本轮文件以kfp前缀区分，原先记录没有替换。公共构建/采集/诊断脚本见该目录及前轮归档。

```bash
bash build-round.sh kfp29
bash kfp-accuracy.sh kfp29 atk_recurrent_kda
bash kfp-accuracy.sh kfp29 atk_recurrent_kda_alignment
bash kfp-final-checks.sh kfp29
python kfp-verify-final.py kfp29
bash kfp-simulator.sh kfp29 fp32
bash kfp29-build-memory.sh
bash kfp29-run-memory.sh
# A5_ci物理卡1
bash kfp-perf.sh layout26 39 kfp-base26-full kfp-full-selection.json
bash kfp-perf.sh kfp29 39 kfp29-full kfp-full-selection.json
python compare.py kfp-base26-full kfp29-full
```

## 逐例目标性能

单位μs，仅正式5样本均值，变化=(kfp29/layout26−1)×100%。

| ID | B | HV | T | layout26 | kfp29 | 耗时变化 |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 32 | 32 | 1 | 117.238 | 41.471 | -64.63% |
| 1 | 32 | 32 | 2 | 219.616 | 68.352 | -68.88% |
| 2 | 32 | 32 | 4 | 425.326 | 123.307 | -71.01% |
| 3 | 32 | 32 | 8 | 837.039 | 233.344 | -72.12% |
| 4 | 32 | 48 | 1 | 170.121 | 86.145 | -49.36% |
| 5 | 32 | 48 | 2 | 322.015 | 103.085 | -67.99% |
| 6 | 32 | 48 | 4 | 624.690 | 182.221 | -70.83% |
| 7 | 32 | 48 | 8 | 1230.450 | 343.627 | -72.07% |
| 8 | 32 | 128 | 1 | 467.989 | 445.160 | -4.88% |
| 9 | 32 | 128 | 2 | 838.969 | 457.866 | -45.43% |
| 10 | 32 | 128 | 4 | 1639.256 | 467.942 | -71.45% |
| 11 | 32 | 128 | 8 | 3239.911 | 887.444 | -72.61% |
| 12 | 64 | 32 | 1 | 225.333 | 162.897 | -27.71% |
| 13 | 64 | 32 | 2 | 425.336 | 173.199 | -59.28% |
| 14 | 64 | 32 | 4 | 826.419 | 236.647 | -71.36% |
| 15 | 64 | 32 | 8 | 1626.979 | 450.818 | -72.29% |
| 16 | 64 | 48 | 1 | 345.774 | 319.925 | -7.48% |
| 17 | 64 | 48 | 2 | 627.924 | 331.493 | -47.21% |
| 18 | 64 | 48 | 4 | 1222.681 | 348.437 | -71.50% |
| 19 | 64 | 48 | 8 | 2411.845 | 663.575 | -72.49% |
| 20 | 64 | 128 | 1 | 915.664 | 898.882 | -1.83% |
| 21 | 64 | 128 | 2 | 1656.287 | 920.637 | -44.42% |
| 22 | 64 | 128 | 4 | 3245.424 | 940.594 | -71.02% |
| 23 | 64 | 128 | 8 | 6424.369 | 1751.891 | -72.73% |
| 24 | 72 | 32 | 1 | 254.212 | 205.685 | -19.09% |
| 25 | 72 | 32 | 2 | 481.924 | 216.140 | -55.15% |
| 26 | 72 | 32 | 4 | 935.343 | 267.728 | -71.38% |
| 27 | 72 | 32 | 8 | 1844.252 | 509.012 | -72.40% |
| 28 | 72 | 48 | 1 | 393.046 | 368.153 | -6.33% |
| 29 | 72 | 48 | 2 | 705.351 | 378.823 | -46.29% |
| 30 | 72 | 48 | 4 | 1376.419 | 392.457 | -71.49% |
| 31 | 72 | 48 | 8 | 2717.359 | 747.057 | -72.51% |
| 32 | 72 | 128 | 1 | 1032.572 | 1014.320 | -1.77% |
| 33 | 72 | 128 | 2 | 1857.554 | 1028.463 | -44.63% |
| 34 | 72 | 128 | 4 | 3642.101 | 1053.607 | -71.07% |
| 35 | 72 | 128 | 8 | 7210.711 | 1965.797 | -72.74% |
