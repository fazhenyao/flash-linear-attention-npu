# RecurrentKda ATK 工程

本目录提供 `recurrent_kda` 的 ATK 单算子工程，包含 `executor_recurrent_kda.py`、`gen_recurrent_kda.py`、`recurrent_kda.yaml`、`atk_recurrent_kda.json`。

## 输入约束

- `layout` 支持 `BSND/TND`；`BSND` 下 `q/k=[B,T,H,K]`，`v/out=[B,T,HV,V]`，`g=[B,T,HV,K]`，`beta=[B,T,HV]`。
- `TND` 下 `q/k=[T,H,K]`，`v/out=[T,HV,V]`，`g=[T,HV,K]`，`beta=[T,HV]`。
- `initial_state` 是 state pool；`state_v_first=true` 时为 `[state_capacity,HV,V,K]`，否则为 `[state_capacity,HV,K,V]`。
- `cu_seqlens` 可选；为空时 BSND 按 batch 行划分，TND 视为一条序列。显式传入时为同设备 `INT32/INT64`，shape `[seq_num+1]`，首项0、末项<=capacity、单调不减。
- `q/k/v/out` 仅支持 `BFLOAT16`；`g/beta` 支持 `FLOAT/BFLOAT16/FLOAT16`，`A_log/dt_bias` 支持 `FLOAT`。
- `HV % H == 0`，`H/HV <= 256`；当前支持 `K=128,V=128` 或 `K=128,V=256`。
- A5 每段长度不再受8限制；A2/A3 当前工作树仍限制为8，对应扩展由参考 PR #917 提供。显式 `ssm_state_indices` 可使 `state_capacity > seq_num`。
- 当前 ATK 用例遵循上述约束，并通过 `case_spec` 固定具体取值；扩展用例时应继续满足这些限制。

## 标杆来源

CPU recurrence 对齐 [PR #917](https://github.com/flashserve/flash-linear-attention-npu/pull/917)
提交 `082b0c704407d0b38909fd4b0f58703b8ab52873`。同一原始 dtype 量化输入提升 FP64 后
计算，最终 golden 按 ATK 规则转 FP32；L2 使用 `x/sqrt(sum(x*x)+1e-6)`。
标准沿用 `mixed_tolerance_bm`。完整 state pool 参与比较，仅排除 API 未定义的 out padding tail。

CPU 标杆、输入构造、run_cpu、run_npu 和 FunctionApi 均在本目录的 `executor_recurrent_kda.py` 中实现；公共文件只提供基础工具函数。

## SOC 支持

YAML 元信息覆盖 `ascend910b`、`ascend910_93` 和 `ascend950`，可配合统一脚本的 `-soc=ascend910b|ascend910_93|ascend950` 使用。
本次新增的长序列正式包是 A5 验收矩阵，不能据此声称 A2/A3 验收通过。

## 默认用例

- BF16 用例：`{"dtype": "bf16", "B": 2, "T": 2, "H": 2, "HV": 4, "K": 128, "V": 128, "layout": "BSND", "state_v_first": true, "op": "recurrent_kda", "case_id": 0, "seed": 20260817, "route": "ascendc", "soc": "ascend910b"}`

## 执行方式

```bash
bash tests/atk/run_test_cpu.sh -op=recurrent_kda -npu_device_id=6
bash tests/atk/run_test_cpu.sh -op=recurrent_kda -npu_device_id=6 -scope=accuracy
bash tests/atk/run_test_cpu.sh -op=recurrent_kda -npu_device_id=6 -scope=performance
bash tests/atk/run_test_cpu.sh -op=recurrent_kda -npu_device_id=6 -scope=determinism
bash tests/atk/run_test_cpu.sh -op=recurrent_kda -npu_device_id=6 -scope=mssanitizer
bash tests/atk/run_test_cpu.sh -op=recurrent_kda -scope=gen_cases
```

`gen_cases` 默认传入 `-dt 100 -en 0`。所有新增工程的 marker dtype 都保留两路生成入口，生成器会把不支持 FP16 的算子改回合法 BF16 用例。

## Issue 680 A5 验证矩阵

公开入口默认使用安装态 Stable-ABI 后端；不强制切换 ctypes，不过滤有定义的 state 槽。
`python tests/atk/recurrent_kda/gen_recurrent_kda.py` 可重建三份正式包；精度候选、模型性能、
人工选择的内存/确定性用例分别维护，`gen_cases` 只生成精度候选。

| 包 | 数量与来源 | 覆盖 |
| --- | --- | --- |
| `atk_recurrent_kda.json` | 42结构×3种子=126 | 原有小 shape；8融合开关组合；长序列、布局、dtype、state、slot及边界 |
| `atk_recurrent_kda_perf.json` | issue 的36组模型 shape | B=32/64/72，HK=16，HV=32/48/128，T=1/2/4/8，K=V=128 |
| `atk_recurrent_kda_mss.json` | 人工选择8组，全部 key=0 | short/window/tail/transpose/slots/packed/fallback/V256 |

精度覆盖每段长度1/2/4/8/9/15/16/17/31/32/33/65/129/257/1024，HV/HK=1/2/3/16；
全零/小范数/强信号；safe/unsafe gate、大 raw gate、beta负特征值开关；融合 QKV storage
offset、head gap、feature stride=2回退；非连续 state、两种布局/dtype、原位/非原位；
无 metadata、空初始 state、空序列/padding；跨窗口重复 slot、accepted token 与未命中槽。
算子只有 key=0，host 不新增 key；8个 MSS case 覆盖本次窗口与现存缓冲分支。

性能属性固定为三融合开关开启、FP32 V-first state、原位更新、融合 storage QKV 切片。
用户确认本轮目标为不劣于库上已有实现；以变更前 `67dc99d7` 同 shape、属性为基线。
不将其他未指定属性或原 issue 的0.8倍原生目标作为本轮达标要求。

## 输入校准与公开入口回归

随机输入沿用 QKV normal(0,0.05)、state normal(0,0.02)，raw gate normal(0,0.3)、
raw beta normal(0,0.6)。小输出可能使全零结果被 BF16 atol 接受，故额外固定
Q/K单坐标=1、V=1、beta=0.5、log gate=0 的 `norm_signal`；保留原随机值域和容差。
独立解析检查覆盖 Q/K=0和2^-12、T=1/9/17，直接检查 epsilon 和每token递推。

```bash
python tests/atk/recurrent_kda/scripts/check_reference_calibration.py --artifacts /tmp/kda-calibration
python tests/atk/recurrent_kda/scripts/check_public_contract.py --device 6
ct viz /tmp/kda-calibration/correct_0.npy /tmp/kda-calibration/golden_0.npy --out_dir /tmp/kda-viz --name correct_0 --spatial
ct viz /tmp/kda-calibration/wrong_0.npy /tmp/kda-calibration/golden_0.npy --out_dir /tmp/kda-viz --name wrong_0 --spatial
# 对 output_1/state 同样执行；不能用 out 代表 state。
```

CPU 校准42结构同精度与FP64逐输出比较通过；固定PR917源码的37个非padding结构逐输出
完全一致（该参考不接受padding capacity，其余5个由本仓接口和独立recurrence验收）。
单坐标解析一致，故意置零out和错误state
均被相同标准拒绝。CT Tool v0.9.1：正确BF16 out贴近y=x，最大绝对误差约1.94e-4；
全零out明显分离，最大误差约0.0884；错误state最大误差约1。图表阈值只用于诊断。
公开入口另检查不返回final state、空初始state、输入不变性、storage、版本计数、grad契约及非法参数。

正式执行不设置 case 范围：

```bash
bash tests/atk/run_test_cpu.sh -op=recurrent_kda -soc=ascend950 -scope=gen_cases
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
bash tests/atk/run_test_cpu.sh -op=recurrent_kda -soc=ascend950 -npu_device_id=6 -scope=all
```

`all` 包含精度/NaN（GM初始化）、ATK performance_device、50轮确定性和mssanitizer memcheck。
性能另按预分配输入的脚本采修改前后 msprof Task Duration，区分目标kernel与辅助kernel：

```bash
msprof --output=/tmp/kda-prof --task-time=on \
  python tests/atk/recurrent_kda/scripts/profile_model_cases.py --device 6 --warmup 10 --samples 20 --manifest /tmp/kda-manifest.json
```

常规采集按36组各30次 `RecurrentKda` task 对齐shape，舍弃前10次，统计后20次均值；
最终交替复测统一预热30次、采样100次，逐组统计后100次均值。
增量内存为输入驻留后的allocator峰值增量，包含输出、workspace和调用内连续化。
最终验证范围、环境、通过结果及用户接受的性能例外归档如下。

## A5_ci 当前验收结果（2026-10-20）

对应基线 `67dc99d7` 上本次工作树的向量归一化+寄存器稳定softplus版本，
未采用实验性的单行Scalar归一化。功能验证通过；原始严格阈值29/36不劣、7/36未达标。
用户于2026-10-20明确确认“接受差异，先交付功能”，接受下表7个性能差异作为本轮例外；
本轮按该例外完成验收，原始逐case阈值仍未全部达到。

环境：A5_ci Ascend950PR卡6，CANN 9.2.0-beta.2，Python 3.11.17，
PyTorch 2.9.0+cpu / torch_npu 2.9.0.post6，ATK 26.9.8；独立venv与源码，
`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`，安装态公开Stable-ABI入口。

被测源码SHA256：

| 文件 | SHA256 |
| --- | --- |
| `op_kernel/arch35/recurrent_kda.h` | `380e74d45a9b31fd08d35c98cb84af0b82a39d02978916ef8f051b2b78b9f46a` |
| `op_host/recurrent_kda_tiling_processor.h` | `4d6df07233b9a63fd9b450f1e1b599fc0f1ca15831a5ee5b3b4c5f71a2c774b2` |
| `torch_custom/fla_npu/fla_npu/ops/ascendc/_stable.py` | `0d79f61a80969d553a036de9411fa390b2990d7b1f8b127518ff07bd9bf66a48` |
| `executor_recurrent_kda.py` | `8852b9b04a8796d0fd09669d51e80cb8379b27c985846178cc47cabe6aa187a1` |

三份JSON语义SHA256（`sha256(json.dumps(json.load(file),sort_keys=True).encode())`）：
accuracy `b6bd764fbcb6a235fd8106e6dfe471f311ef1c99cf31dba13de249e167b2f809`；
perf `d4e9749143f1254257b86f96ab35d58e319a2716a55714b130fc3240ad25ba30`；
mss `39e16928aa594b3a8d4ba09f690714b802e4b5010c192fbc45fba4586eacbaac`。

| 动作 | 结果 | 报告或证据 |
| --- | --- | --- |
| `gen_cases`，无range | 100个精度候选生成成功；正式包42结构×3种子 | `fla-680-a5-gen-final.log` |
| FP64参考校准 | 42结构通过，37个与固定PR917完全一致；独立解析和故意错误拒绝通过 | `fla-680-a5-calibration-final.log` |
| `all`精度/NaN初始化 | 126/126，失败0 | `atk_recurrent_kda_reports_2026-10-20-19-56-33.xlsx` |
| `all` performance_device | 36/36执行成功；性能达标以独立msprof为准 | `fla-680-a5-atk-v5.log` |
| `all`确定性，50轮 | 8/8，失败0 | `atk_recurrent_kda_mss_reports_2026-10-20-20-00-43.xlsx` |
| `all` memcheck | 8/8，失败0，目标kernel无错误 | `atk_recurrent_kda_mss_reports_2026-10-20-20-01-04.xlsx` |
| 公开接口、mutation/grad | 42项通过，含无cu/空initial_state/V256 | `fla-680-a5-gatevf-public42.log` |
| 安装核验 | 72个kernel .o、opapi和Python wrapper匹配构建源码 | `fla-680-a5-gatevf-md5.log` |

交付前已恢复上述源码哈希对应的向量版本，重新安装保存的同一构建产物
`/home/fazhenyao123/runs/fla-680-gatevf.run`；未保留Scalar实验版本作为安装结果。
该run包SHA256为 `dd89ae1a7363aa744f8ad873281da5ebaafeb072d408acb54dac814e4431b03d`。
`fla-680-a5-delivery-md5.log`再次核对全部72个kernel、opapi和wrapper通过，
新Python进程重跑42项公开接口通过（`fla-680-a5-delivery-public.log`）。

新增覆盖跨窗口长序列、epsilon、稳定gate、全部融合开关、state复制与slot完整pool。
原有V256和FP16辅助输入支持保留；正式矩阵覆盖issue要求的BF16/FP32辅助输入，
不能据此新增FP16验收结论。A2/A3 kernel未改动，本次没有A2/A3设备验收。

ATK_OUTPUT_ROOT只重定向本轮accuracy/perf；确定性和memcheck实际写入算子目录的
`atk_output/`。统一脚本会对后两项显示“未找到报告”；上表已使用
`tests/atk/common/check_atk_result.py` 对实际目录分别复检，不能据“跳过”判通过。

### TilingKey与分支

| key | 条件 | 精度case id | MSS case id | SoC | 实际证据 |
| --- | --- | --- | --- | --- | --- |
| 0（唯一） | 全部合法输入，dtype由编译变体决定 | 普通0、边界1-41；每42项换种子重复 | 0-7 | A5/ascend950 | `fla-680-gatevf-ab-1-new-tiling.log`：RecurrentKda launch `tilingKey: 0,numBlocks: 56`；MSS执行 `RecurrentKda_06bba6860f70bfd6aa54df1d4ed1928e_0` |

MSS 0短序列、1跨窗、2尾窗、3 K-first/间隔state、4 speculative/repeated slot、
5 packed/空序列、6 feature-gap回退、7 BF16/V256；属性组合见生成器。
B64/HV32/T1的新旧运行日志均为UB=253952B、vStep=64、state/out队列深度2，
本次没有改变该模型的分核与队列选择。

### 36组性能逐项比较

均为Ascend950/BF16 QKV、FP32 g/beta及state，HK=16、K=V=128，属性固定如上。
基线 `67dc99d7`，目标 `new_us <= old_us`；单位us，ratio为新/旧，越小越好。
同卡交替安装新旧版本，预分配输入，预热30次后采样100次；每版本4680个目标task，
只统计RecurrentKda设备时间，辅助算子与输入生成不计入目标kernel时间。

| case id | B / HV / T | 基线us | 本次us | 比值 | 结论 |
| --- | --- | --- | --- | --- | --- |
| 0 | 32 / 32 / 1 | 59.9578 | 57.8751 | 0.965264 | 不劣 |
| 1 | 32 / 32 / 2 | 95.2449 | 79.8444 | 0.838307 | 不劣 |
| 2 | 32 / 32 / 4 | 159.8954 | 128.1739 | 0.801611 | 不劣 |
| 3 | 32 / 32 / 8 | 288.5218 | 219.8480 | 0.761981 | 不劣 |
| 4 | 32 / 48 / 1 | 100.0874 | 95.6775 | 0.955939 | 不劣 |
| 5 | 32 / 48 / 2 | 138.9297 | 125.0125 | 0.899825 | 不劣 |
| 6 | 32 / 48 / 4 | 236.8486 | 191.5319 | 0.808668 | 不劣 |
| 7 | 32 / 48 / 8 | 430.3986 | 327.3650 | 0.760609 | 不劣 |
| 8 | 32 / 128 / 1 | 452.0251 | 451.8628 | 0.999641 | 不劣 |
| 9 | 32 / 128 / 2 | 467.1132 | 464.8114 | 0.995072 | 不劣 |
| 10 | 32 / 128 / 4 | 668.8143 | 535.2314 | 0.800269 | 不劣 |
| 11 | 32 / 128 / 8 | 1137.9879 | 865.4143 | 0.760478 | 不劣 |
| 12 | 64 / 32 / 1 | 173.2434 | 167.1979 | 0.965104 | 不劣 |
| 13 | 64 / 32 / 2 | 199.1451 | 179.8826 | 0.903274 | 不劣 |
| 14 | 64 / 32 / 4 | 317.4018 | 256.7382 | 0.808874 | 不劣 |
| 15 | 64 / 32 / 8 | 573.3985 | 435.8081 | 0.760044 | 不劣 |
| 16 | 64 / 48 / 1 | 325.8903 | 326.0334 | 1.000439 | 未达标 |
| 17 | 64 / 48 / 2 | 337.4034 | 337.4280 | 1.000073 | 未达标 |
| 18 | 64 / 48 / 4 | 495.9449 | 397.4722 | 0.801444 | 不劣 |
| 19 | 64 / 48 / 8 | 853.4129 | 647.6490 | 0.758893 | 不劣 |
| 20 | 64 / 128 / 1 | 907.1381 | 906.1128 | 0.998870 | 不劣 |
| 21 | 64 / 128 / 2 | 927.4451 | 927.4904 | 1.000049 | 未达标 |
| 22 | 64 / 128 / 4 | 1324.9615 | 1052.6453 | 0.794472 | 不劣 |
| 23 | 64 / 128 / 8 | 2249.0926 | 1700.6941 | 0.756169 | 不劣 |
| 24 | 72 / 32 / 1 | 207.0574 | 211.1663 | 1.019844 | 未达标 |
| 25 | 72 / 32 / 2 | 233.2828 | 219.5204 | 0.941005 | 不劣 |
| 26 | 72 / 32 / 4 | 364.3337 | 291.9659 | 0.801369 | 不劣 |
| 27 | 72 / 32 / 8 | 651.4040 | 494.3825 | 0.758949 | 不劣 |
| 28 | 72 / 48 / 1 | 376.6066 | 377.9932 | 1.003682 | 未达标 |
| 29 | 72 / 48 / 2 | 388.9009 | 386.1567 | 0.992944 | 不劣 |
| 30 | 72 / 48 / 4 | 561.4603 | 449.4191 | 0.800447 | 不劣 |
| 31 | 72 / 48 / 8 | 959.7260 | 727.9848 | 0.758534 | 不劣 |
| 32 | 72 / 128 / 1 | 1019.8165 | 1020.6973 | 1.000864 | 未达标 |
| 33 | 72 / 128 / 2 | 1041.6621 | 1044.9204 | 1.003128 | 未达标 |
| 34 | 72 / 128 / 4 | 1485.8383 | 1180.4448 | 0.794464 | 不劣 |
| 35 | 72 / 128 / 8 | 2521.3429 | 1907.9097 | 0.756704 | 不劣 |

T=4耗时为基线的约79.4%～80.9%，T=8为75.6%～76.2%；主要收益来自无slot时
只写最终state与批归一化。未达标case id为16/17/21/24/28/32/33；
其中24（B72/HV32/T1）慢1.9844%，其余慢0.0049%～0.3682%。
平均比值0.875758不替代逐case结论。用户接受的例外只对应上述7组、
当前属性和本次数据，不扩展为其他输入或后续变更的性能豁免。

36组原位调用在输入驻留后的allocator额外峰值最大54,526,464B（52.0005MiB），
低于issue的200MB上限，计入输出、系统workspace及调用内连续化。
该数字不能用于非原位的大state pool，后者还需复制整个初始state。

证据保留于A5_ci：源码 `/home/fazhenyao123/src/fla-680-a5-port-20261020`，
日志/报告根 `/home/fazhenyao123/runs/`。精度/perf在`fla-680-a5-atk-v5/`，
确定性/内存报告在源码的 `tests/atk/recurrent_kda/atk_output/`；
性能根`fla-680-gatevf-ab-1-{old,new}-prof/`，各自manifest固定shape、属性、采样数，
`op_summary*.csv`保存逐task时间，`fla-680-gatevf-ab-results.json`保存均值/中位数/标准差。
