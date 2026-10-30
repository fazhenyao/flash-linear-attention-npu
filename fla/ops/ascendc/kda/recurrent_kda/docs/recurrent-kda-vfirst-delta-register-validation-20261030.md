# V-first四行delta寄存器验证

2026-10-30，方案规则V2，基线为已验收的kbf30。受保护原方案不修改；此次仅实验arch35的V-first FP32/BF16寄存器计算，不修改接口、CPU标杆、值域或精度标准。既有K-first两遍计算保留。

## 候选vf31

当前V-first每四行分别归约prediction、广播prediction、从vFloat scratch广播v，再进行四次向量减法；vFloat由每token两次VST和专用VST→VLD屏障产生。候选将四个prediction的有效lane经三次Interleave收集到前四lane，与当前v寄存器前四lane一次Sub，随后分别广播四个delta供原State FMA更新。v寄存器每块按四lane循环移动，处理16块覆盖64行；两个64元素半区独立消费。prediction归约树、Sub和State FMA的操作数不变，预期out/State逐位一致。

理论每块原4次prediction广播+4次v广播+4次Sub共12条，替换为3次Interleave+1次Sub+4次delta广播+1次v轮换共9条；每token32块约净减96条，另省vFloat的两次VST及专用屏障。需要额外3个寄存器索引常量，并延长v寄存器存活时间，实际收益必须检查编译结果与真机性能，不能仅按静态计数判定。必要State/out同步不变。

仍为一个Vector Stage、原任务映射、原128×128 State布局与bank生命周期；UB和GM空间、流量规划不变，vFloat尾部预留不再使用但不回收。两个递推bank及第三bank预取、全tile后批量out搬出保留。BF16 State仅在Load后/Store前于寄存器Cast，out仍使用未舍入FP32 State。

数值风险：只合并独立lane的减法、移动和广播，归约及乘加顺序不变；四lane mask排除其余无效归约位置，v的轮换覆盖0～63且最后回到初始顺序，不引入高风险算术。有限输入和既有归一化/gate保护沿用原实现。

## 验证计划

标准run构建安装和72对象身份核对；132原有+54对齐精度后，先在A5_ci同卡串行比较issue680 ID0/2/3/7/20/35的V-first FP32/BF16（12组），每例3预热+5正式样本，唯一口径msprof RecurrentKda Task Duration。目标为收益超过采样波动且典型case无不可接受回退；不满足则恢复kbf30。若有稳定收益，再覆盖全部36模型形状×两State dtype，补充K-first控制、74公开契约、9确定性和原生内存检查。正常启动逐位对照及小规模完整simulator用于核对实际指令变化，不替代模型真机计时。

开发参考沿用CHUNK_DEPENDENT V1（a14961e9），只复用跨tile状态依赖和同步闭环检视；本候选不引入Cube、workspace或新Stage。

## 结果：两个候选均不采用，恢复kbf30

vf31四行prediction打包后共同计算delta，vf32取消prediction打包、保留独立prediction广播/Sub，仅直接从寄存器读取v。两版均通过精度，但未得到性能收益。当前本地两份生产头文件和A5_1构建源均恢复到本轮开始的kbf30，隔离venv也通过原标准run重新安装kbf30；18源文件与78个已安装so/o身份核对通过。候选源码快照、封装包和结果仅留作实验对照。

采用ascend-ssh和msprof-op-profiling的方法：构建/精度及原生启动在A5_1物理卡6/逻辑0，正式性能在A5_ci物理卡5/逻辑0；CANN 9.2.0-beta.2、Python 3.11.17、torch 2.9.0+cpu/torch_npu 2.9.0.post6。机器共享，逐例前后占用记录保留。每轮重新采基线，基线进程完成后才启动候选，同卡串行，无采集交叠。基线是当前kbf30，不是远程main。

msprof Task Duration：issue680 ID0/2/3/7/20/35×V-first FP32/BF16，共12个组合；保留原模型shape和输入构造，H16、K=V128、packed TND、fused QKV view，QKV/out BF16、gate/beta FP32，QK norm/raw gate/beta sigmoid开启、原位最终State、无indices，X按序列长度选择1/2/4/8。每例3预热+5正式样本，不替换样本。

| 候选 | State类型 | 候选/基线几何均值 | 耗时变化 | 均值更低组数 | 最大组内CV | 判定 |
| --- | --- | --- | --- | --- | --- | --- |
| vf31 | V-fp32 | 1.224961 | +22.50% | 1/6 | 0.72% | 不采用 |
| vf31 | V-bf16 | 1.207999 | +20.80% | 0/6 | 1.11% | 不采用 |
| vf32 | V-fp32 | 1.007203 | +0.72% | 2/6 | 2.06% | 不采用 |
| vf32 | V-bf16 | 1.009709 | +0.97% | 1/6 | 0.87% | 不采用 |

vf31的大幅回退超出组内波动。vf32短例ID0有约1.5%～2%的降低，但多数长序列组增加约1%～2%，未获得可采用的整体收益；不能仅选择短例报告加速，也不将接近波动范围的差异宣称为稳定的指令级回退。两项均在典型实验阶段淘汰，不扩展到完整72组合，不重复74契约/9确定性/内存探针；最终保留的是上一轮已验收、且本轮按身份恢复的kbf30。此次只排除两项候选，不代表V-first不存在其它优化空间。

每份原始CSV为192行×48列，96条目标RecurrentKda，设备5、56block；按Task Start Time与manifest对应，每组前3条预热不计入5样本均值。四份CSV和两个独立汇总均保存。

## 精度和指令证据

两候选各自完成标准run构建安装、libcust_opapi MD5一致、72对象JSON/build审计，132原有+54对齐精度全部通过，未修改CPU标杆、值域或混合容差。小规模正常原生启动中，FP32/BF16两种State的out和State均与kbf30逐位一致。每轮18实现/调用源文件核对通过，基线+候选156项so/o跨A5_1与A5_ci SHA256一致。下载归档逐文件校验大小与SHA256。

正式包原生对象的Ascend950PR_957c simulator，单AIV、B2/H=HV1/T8/K=V128、V-first BF16 State、X8，两任务16token，三版完整执行到HOST_FINALIZED；仿真只作诊断，不代替模型真机计时。

| 版本 | 完整事件数 | 相对基线事件变化 | 仿真区间（μs） |
| --- | --- | --- | --- |
| kbf30-v | 50192 | +0.00% | 32.422 |
| vf31-v | 48403 | -3.56% | 40.311 |
| vf32-v | 50451 | +0.52% | 32.848 |

| 动态指令 | kbf30 | vf31 | vf32 |
| --- | --- | --- | --- |
| RV_VCADD | 4128 | 4128 | 4128 |
| RV_VSUB | 2048 | 512 | 2048 |
| RV_VDUP | 2080 | 544 | 2592 |
| RV_VINTLV | 1536 | 3072 | 1536 |
| RV_VSELR | 512 | 2560 | 2560 |
| RV_VLD | 6288 | 4240 | 4240 |
| RV_VSTI | 32 | 0 | 0 |
| RV_SMEM_BAR | 80 | 64 | 64 |

vf31指令事件净减约3.56%，但新增的prediction Interleave使VINTLV加倍，寄存器Gather的VSELR增至五倍。原来各行可独立完成的delta，现在都必须等四行prediction归约及三次Interleave后才能广播；这条依赖链是有代码依据的主要嫌疑，完整仿真也从32.422增到40.311μs。尚未通过逐周期停顿分解将全部回退归因到单条指令，不能只按“寄存器数量”解释。

vf32取消额外Interleave，仿真回到32.848μs，但v从UB广播改为寄存器Duplicate/Gather和轮换，VLD减少2048的同时VSELR增加2048、VDUP增加512；省去32条scratch VSTI及16条专用屏障后，总事件仍增加约0.52%。因此删除UB临时写读没有得到净关键路径收益，当前scratch方案优于这次寄存器替换。必要State/out同步均未删除；基线的128行prediction/out归约合计VCADD4128在两候选中不变。

K-first的连续V两遍算法通过沿K累计同时产生64个V结果，消除了逐行归约；V-first在K连续布局上直接照搬需要离散访问或转置。这次两项实验保持原连续State读写，未能消除其逐行归约，不能期待复制K-first的约50%收益。

静态issue覆盖188项通过；全算子仍有114项既有host DFX-MESSAGE发现，两个实验kernel头文件无发现。A2/A3未上板。正式design.md和受保护原方案均未因淘汰候选改变，受保护SHA256保持原值。通用安装校验未找到kernel目录，恢复后额外逐项核对78个已安装so/o，未用MD5通过替代kernel身份检查。

## 复核路径与命令

本机`output/rkda-vfirst-delta-register-20261030/{a5-1,a5-ci}`保存全部原始报告、正式样本、占用、CSV、完整小规模trace及哈希；两个zip均自带SHA256索引。根目录保存baseline30/vf31/vf32两份kernel头文件快照及候选封装tgz。服务器沿用两机`runs/rkda-layout-register-optimizations-20261030`，本轮文件以vf前缀区分。恢复安装和源/对象核对见`vf-restored-*`。

```bash
# A5_1：每候选先构建并通过精度
bash build-round.sh vf31
bash vf-accuracy.sh
bash build-round.sh vf32
bash vf32-accuracy.sh
# A5_ci：每条命令执行完成后再启动下一条
bash vf-perf.sh kbf30 12 vf-base30-typical vf-typical-selection.json
bash vf-perf.sh vf31 12 vf31-typical vf-typical-selection.json
python compare.py vf-base30-typical vf31-typical
bash vf-perf.sh kbf30 12 vf32-base30-typical vf-typical-selection.json
bash vf-perf.sh vf32 12 vf32-typical vf-typical-selection.json
python compare.py vf32-base30-typical vf32-typical
# A5_1：完整小规模诊断
bash vf-simulator.sh kbf30 bf16
bash vf-simulator.sh vf31 bf16
bash vf-simulator.sh vf32 bf16
# 先恢复两份生产头文件，再恢复隔离venv
bash vf-restore.sh
```

## vf31逐例性能

单位μs，正式5样本均值，各轮单独重测基线；变化=候选/该轮kbf30基线−1。

| ID | B | HV | T | State | kbf30 | vf31 | 耗时变化 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 32 | 32 | 1 | fp32 | 40.865 | 48.117 | +17.75% |
| 2 | 32 | 32 | 4 | fp32 | 119.633 | 154.288 | +28.97% |
| 3 | 32 | 32 | 8 | fp32 | 226.889 | 295.402 | +30.20% |
| 7 | 32 | 48 | 8 | fp32 | 331.329 | 432.469 | +30.53% |
| 20 | 64 | 128 | 1 | fp32 | 764.476 | 764.364 | -0.01% |
| 35 | 72 | 128 | 8 | fp32 | 1919.621 | 2513.579 | +30.94% |
| 0 | 32 | 32 | 1 | bf16 | 44.735 | 52.258 | +16.82% |
| 2 | 32 | 32 | 4 | bf16 | 136.230 | 171.727 | +26.06% |
| 3 | 32 | 32 | 8 | bf16 | 259.944 | 331.206 | +27.41% |
| 7 | 32 | 48 | 8 | bf16 | 379.764 | 485.462 | +27.83% |
| 20 | 64 | 128 | 1 | bf16 | 399.670 | 404.533 | +1.22% |
| 35 | 72 | 128 | 8 | bf16 | 2210.599 | 2829.659 | +28.00% |

## vf32逐例性能

单位μs，正式5样本均值，各轮单独重测基线；变化=候选/该轮kbf30基线−1。

| ID | B | HV | T | State | kbf30 | vf32 | 耗时变化 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 32 | 32 | 1 | fp32 | 42.278 | 41.661 | -1.46% |
| 2 | 32 | 32 | 4 | fp32 | 119.646 | 122.095 | +2.05% |
| 3 | 32 | 32 | 8 | fp32 | 226.295 | 229.608 | +1.46% |
| 7 | 32 | 48 | 8 | fp32 | 331.138 | 335.019 | +1.17% |
| 20 | 64 | 128 | 1 | fp32 | 763.510 | 763.189 | -0.04% |
| 35 | 72 | 128 | 8 | fp32 | 1919.084 | 1941.718 | +1.18% |
| 0 | 32 | 32 | 1 | bf16 | 44.704 | 43.812 | -2.00% |
| 2 | 32 | 32 | 4 | bf16 | 135.849 | 137.938 | +1.54% |
| 3 | 32 | 32 | 8 | bf16 | 259.489 | 263.676 | +1.61% |
| 7 | 32 | 48 | 8 | bf16 | 379.755 | 386.100 | +1.67% |
| 20 | 64 | 128 | 1 | bf16 | 398.133 | 403.836 | +1.43% |
| 35 | 72 | 128 | 8 | bf16 | 2210.415 | 2246.240 | +1.62% |
