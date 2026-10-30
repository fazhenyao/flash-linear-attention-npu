# K-first State 离散读写归因

2026-10-29，A5_1。结论：FP32 K-first State 的512 B lane步长造成单次 Gather/Scatter 内部的硬件 subbank 冲突，是已复现的主要瓶颈。下一任务预取、MTE2并发以及寄存器溢出均不是复现此问题的必要条件。结论依据为生产二进制的小用例、相同访存诊断二进制的行距对照和硬件地址映射，而非仅比较聚合流水时间。

本轮只添加诊断程序和记录，未修改生产实现或用户方案。全部9次诊断启动均完整结束、同步及退出成功，输出检查通过；没有用超时截断数据计算本轮统计。它们是缩小用例和独立访存实验，不替代issue680原模型的真机性能验收。

## 1. 地址归因

当前 FP32 K-first 分支在 `recurrent_kda_regbase.h` 的 `UpdateRows4` 中按逻辑 V 行处理：

```text
addr(lane) = State软件bank基址 + 4 * (v + 128 * (k_half * 64 + lane))
lane = 0..63
```

同一条 Gather/Scatter 内相邻 lane 的地址差为512 B。一个四行块发出8条UB Gather和8条Scatter；每个128行 token 共256条Gather和256条Scatter。这里分析的是UB读指令 `RV_VGATHER2`，不是out寄存器收集所用的寄存器重排Gather。

[Ascend C 3510 bank说明](https://asc.gitcode.com/guide/%E7%AE%97%E5%AD%90%E5%AE%9E%E8%B7%B5%E5%8F%82%E8%80%83/SIMD%E7%AE%97%E5%AD%90%E6%80%A7%E8%83%BD%E4%BC%98%E5%8C%96/%E5%86%85%E5%AD%98%E8%AE%BF%E9%97%AE/%E9%81%BF%E5%85%8DUB%E7%9A%84bank%E5%86%B2%E7%AA%81/avoid_bank_conflict_npu_arch_3510.html)给出低位交织布局：BG由地址位[7:5]选择，组内bank由位8选择，subbank由位[4:3]选择；离散访问同一subbank的不同地址会冲突。该资料来自Huawei asc-devkit master生成的预览文档；实际实验环境为CANN 9.2.0-beta.2的Ascend950PR_957c模型。

```text
BG       = (addr >> 5) & 7
bank     = (addr >> 8) & 1
subbank  = (addr >> 3) & 3
```

增加512 B只改变更高地址位，三个选择值均不变。由此，64个lane落入同一硬件bank、同一subbank的不同地址，需要排队服务。两个K半区相隔32768 B，也没有改变这些选择位。

代码中的三个State“bank”是三块软件缓冲，基址相隔65536 B；它们不是三个硬件bank。65536也是512的整数倍，所以切换软件bank没有打散对应位置的硬件bank/subbank映射。仅整体偏移某个软件bank，不能消除单条指令内64个lane的冲突；增加R也不改变这一点。

## 2. 同一生产二进制的隔离实验

生产 `.o` SHA256为 `52e4af4f890b650830017b96e9c1ea33097f37ef88ff01fa00b76c478bb8ca7c`，与前次模型采集完全一致。手工构造自洽诊断tiling：单AIV、H=HV=1、T=1、X=1、B=1或2，K=V=128；融合开关沿用原case，FP32 State，有限常量输入。脚本记录全部tiling字段及改动，不冒充原始56-block模型。

| 诊断 | Gather中位数 | Scatter中位数 | 首token State行计算区间 | 区间内MTE2 |
| --- | --- | --- | --- | --- |
| 单任务K-first，无下一任务 | 258.8 ns | 386.1 ns | 20.260 μs | 无 |
| 两任务K-first，保留下一任务预取 | 258.8 ns | 386.1 ns | 20.260 μs | 无，预取已先完成 |

单任务K-first中最后一笔MTE2在约3.368 μs结束，首条State Gather在4.662 μs开始，故不能把该区间的长延迟归因于并发搬入。两任务小用例的预取也在State行计算开始前结束，不能用它声称已测到并发竞争。

同一二进制的单任务V-first完整仿真区间为7.259 μs，K-first为26.124 μs；V-first的VLD/VST中位数为5.45/7.88 ns。这个小用例的完整耗时包含入口、搬运和排空，不能与上述State行计算区间混用，也不用于替换正式计时。

三个生产对象诊断均检查全部out及State；out相对标量预期至多允许1个BF16 ULP，FP32 State绝对误差要求小于1e-6，本次最大误差为0。

## 3. 只改变物理行距

独立 `state_pitch_probe` 复现四个V行、两个K半区的访存：共32次64-lane Gather、32次Scatter及32次加1，读写各2048个FP32元素。相同二进制、相同输入缓冲、相同UB基址和指令数，只改变运行时物理行距；访问的具体元素随索引改变。初始MTE2完成后才计算，VF期间没有搬入。

| 物理行距（FP32元素） | lane字节步长 | Gather中位数 | Scatter中位数 | 首Gather至末Scatter结束 |
| --- | --- | --- | --- | --- |
| 128 | 512 B | 258.8 ns | 294.5 ns | 2.49394 μs |
| 129 | 516 B | 22.4 ns | 21.8 ns | 0.16970 μs |
| 132 | 528 B | 38.2 ns | 40.0 ns | 0.32182 μs |
| 136 | 544 B | 69.7 ns | 76.4 ns | 0.63212 μs |

129行距的读写区间降低93.20%，约为原先的1/14.70；136约为原先的1/3.95。它们是局部读写改善，不是整算子加速比。所有变体输入/输出元素数相同，改善不来自减少计算或数据量。编译报告为9个VReg、1个MaskReg、VF栈0 B，且行距变体使用同一个可执行文件，排除了“改变行距后寄存器分配不同”的解释。

对基址0、V下标0、64个lane，128行距把64个请求集中到BG0/bank0/subbank0；129行距将它们分散到8个BG的bank0、共32个subbank位置，每处2个请求。实际延迟下降与冲突缓解吻合。

行距实验二进制SHA256：`d39fe6baee686b2fdcd0724cb79d3fd308e45e19d259d0d1e37f78df7b0e0729`，归档为 `pitch-probe-v1`；源码为 `pitch-probe-v1.cpp`。

## 4. 显式并发MTE2对照

另一个诊断二进制保留相同行距128的VF，通过运行时开关控制是否向独立scratch搬入4次64 KiB数据。VF与scratch没有数据依赖，全部搬运在释放UB前等待完成。trace确认4笔MTE2区间与Gather/Scatter区间重叠。

| MTE2干扰 | Gather中位数 | Scatter中位数 | 读写区间 |
| --- | --- | --- | --- |
| 关闭 | 258.8 ns | 294.5 ns | 2.49394 μs |
| 开启 | 258.8 ns | 294.5 ns | 2.49394 μs |

该模型下未测到VF区间增加，额外DMA使完整诊断kernel的区间由6.64增至9.28 μs。结合无MTE2时同样存在长延迟，可以排除“MTE2是该瓶颈必要来源”的解释。**不能据此断言所有真实负载下MTE2完全没有竞争**，也不能据模型行为声称已量化真机冲突计数。

此组使用相同二进制SHA256 `16c6948af140bd5e2e1fa509c17a2e558e6c8261aba5754eb12b237a645ae87a`，VF仍为9个VReg、栈0 B。它与行距实验分别保存，不把两组不同编译产物当作同一二进制。

## 5. 修复方向及接入限制

优先评估保留GM接口布局、在UB中给K-first State增加物理padding，使Gather/Scatter的lane落点分散。只移动软件bank基址、减少Scalar、调整预取时机或增加R，不能消除已证明的单指令内部冲突。

129是本轮访存诊断中最快的行距，但不是可直接替换的生产参数：它需要同步修改State加载/写回、三块软件bank的大小和基址、两K半区偏移、Gather/Scatter索引、快照与FP32/BF16路径。不能只把columns乘数改为129。

现有2D DataCopyPad路径在UB一侧按32 B描述行间隔；本机CANN `asc/impl/basic_api/dav_3510/kernel_operator_data_copy_impl.h` 的步长计算也乘以 `BYTE_32_ALIGN` 后对齐。129个FP32元素是516 B，不能直接用原路径表达每行4 B间隔，需要额外重排或其它受支持搬运方式；其代价尚未计入上述收益。[DataCopyPad参数说明](https://www.hiascend.com/doc_center/source/zh/canncommercial/900/API/ascendcopapi/atlasascendc_api_07_0265.html)。

136个FP32元素是544 B，保留32 B行对齐，适合作为首先接入验证的候选。单State缓冲由65536增至69632 B，三块增加12288 B；按当前X=8资源公式估算，峰值由224832增至237120 B，仍低于253952 B。该估算不是完整生命周期和搬运实现验收。BF16的32 B对齐行距和成对读写必须另行设计，不能直接套用FP32的136。

候选集成后仍须验证真实模型精度、输出布局、indices快照、bank复用，以及msprof Task Duration；本轮没有宣称修复已集成或原case已获得14.7倍加速。历史基线“先转置后连续递推”也能避开此类访问，但不属于本轮修改范围。

## 6. 复现与证据

环境沿用前次A5_1独立venv及CANN 9.2.0-beta.2。生产对象诊断使用 `msprof op simulator --soc-version=Ascend950PR_957c --kernel-name=RecurrentKda --launch-count=1 --core-id=0 --timeout=1 --aic-metrics=PipeUtilization,ResourceConflictRatio`；独立访存诊断将kernel名替换为 `state_pitch_probe`。每个变体只启动一次，全部完整结束，无重复200样本采集。

- 远程目录：`<run-root>/rkda-kfirst-state-access-20261029/`。
- [中文图](assets/state-access-diagnosis.png)。
- 完整证据包（仅本地归档，未发布：`state-access-evidence.tar.gz`，SHA256 `11dcc8b656b8fe08a0f1bab506941991145eaf1cc453d0147e601457d07ee711`）：诊断源码、两个二进制、全部trace与CSV、命令、输出检查、编译报告和SHA256索引。
- 统计JSON（仅本地归档，未发布：`access-summary.json`，SHA256 `dc2a70e9b7ed5b3c3a098313c69b4c8e7ccaea549089478694395ead5e564723`）：直接解析trace事件计算，不把聚合CSV行数当作执行次数；`prefetch-k`只取首任务的256条Gather和Scatter作对照。
