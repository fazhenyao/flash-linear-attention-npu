# Recurrent KDA A2 上板测试记录（2026-10-31）

首轮历史结论：**未通过**。当时 PR #997 提交 `be0ecad8a739de5b205d1a229058d58d2407a89d` 在 A2 上完成构建安装，但 8 个冒烟用例均触发设备地址异常；另一个普通 T=2 用例也复现失败。没有有效的 out/State 精度通过结果，不具备继续性能采集或声明 A2 验收通过的条件。本轮未修改算子实现和冻结的 A5 方案。

## 环境与版本

- 设备：A2 / 910B3，物理卡 4，进程内逻辑卡 0；开始和结束的设备快照均健康，未观察到其他 NPU 进程。
- 系统：aarch64；CANN `9.3.0~weekly.20261021.01`，驱动 `26.1.1`。
- Python 3.11.17，torch `2.9.1+cpu`，torch_npu `2.9.1`，正式尝试使用 ATK `26.9.8`。
- 当前提交通过 `git archive` 同步，使用独立 venv 构建和安装单算子 wheel。wheel 的 `+unknown` 版本后缀来自 archive 不含 `.git`，被测来源由上述提交及私有证据固定。
- `libcust_opapi.so` MD5 为 `77a12b82ba591c52fb421d16fd2fa999`；全部 72 个安装 kernel 对象与本次构建逐一匹配，另保存 78 个已安装 so/o 的 SHA256。

构建使用 `FLA_NPU_SOC=ascend910b`、`FLA_NPU_OPS=recurrent_kda`、`FLA_NPU_BUILD_ARGS=-j8`，执行 `python scripts/build_wheel.py --no-progress`，随后安装 wheel 并执行 `python scripts/verify_libcust_opapi_md5.py`。

首次构建因 bisheng 自动选择不完整的 GCC 12 目录而找不到 `cstdint`；在本次进程设置 GCC 11 的 `CPLUS_INCLUDE_PATH` 后标准构建通过。首次运行未预配置自定义 OPP，误选系统旧 tiling，且共享 ATK 26.9.24 缺少 `mixed_tolerance_bm`；分别预配置当前 wheel 的 OPP、在独立 venv 引入既有 ATK 26.9.8 后重试。共享环境未修改，失败日志均保留。这些环境失败与下述正式设备异常分开记录。

## 实际执行与结果

冒烟 shape 为 `B=2,T=9,H=1,HV=2,K=128`，种子 `20261031`，开启 q/k L2、raw gate 和 beta sigmoid，原位 State 更新。所有用例均设 `soc=ascend910b`；沿用现有 CPU 标杆对 A2 的 FP32 跨 token State 规则及原 `mixed_tolerance_bm`，没有修改数学公式、输入值域或阈值。

| State 路径 | V=128 | V=256 |
| --- | --- | --- |
| V-first FP32 | 设备异常 | 设备异常 |
| V-first BF16 | 设备异常 | 设备异常 |
| K-first FP32 | 设备异常 | 设备异常 |
| K-first BF16 | 设备异常 | 设备异常 |

八例均报 `507035 / vector core exception`。设备扩展信息包含 `The GM address accessed by scalar exceeds 48 bits`，其中 V=128 V-first FP32 和 V=256 V-first BF16 还明确报告 `The address for the VEC instruction to read/write UB is out of bounds`。首次失败后，其余七例分别在新进程中执行，排除单个异常 context 继续污染后续用例的解释。

另外，对原有精度集合第 0 例单独复现：`B=2,T=2,H=2,HV=4,K=V=128`、V-first FP32 State、种子 `20260817`，同样出现 UB 越界与 Scalar 地址异常。这说明问题并非仅由 T=9 跨窗口触发。

ATK 在异常路径仍可能返回进程退出码 0，且未生成完整精度报告；本轮依据任务错误日志和设备异常判断失败，未将退出码当作通过证据。预备的 234 例完整精度集、公开接口、确定性、sanitizer 和性能均未继续执行，不能把未执行项计为失败或通过。

复现入口为 ATK 的 NPU/CPU 节点链，加载当前 `executor_recurrent_kda.py`，执行 `--task accuracy --bm_device cpu -sp --save_report_json --save_data output`。分例执行通过 `-s i -e i+1` 选择用例，精确用例 JSON 和 shell 命令保留于私有证据。

## 已确认的 UB 预算不一致

[kernel 的 InitLocalBuffers](../op_kernel/recurrent_kda.h) 固定使用 `PACKED_INPUT_BUFFER_NUM=2` 分配 packed q/k/v/g 输入。每槽字节数为：

```text
X × (4K + 2V + 4K)，其中 generic kernel 的 X=8
K=V=128：8 × (512 + 256 + 512) = 10240 B
```

而 [host 的 CalcFixedUbBytes](../op_host/recurrent_kda_tiling_processor.h) 使用 `windowBufferNum = isA5 ? 2 : 1`，在 A2 分支只计入一槽，少算 **10240 B（10 KiB）**。beta 队列有自己的单槽预算，不能将整个 fixed 区统一翻倍来代替逐区核算。

T=2 复现的实际 tiling 日志为 `ubCalSize=196352`、`ubRestBytes=84928`、`vStep=64`、`stateOutBufferNum=1`、`attnOutBufferNum=2`。按 kernel 的分配顺序核算：

| 分配 | 字节数 |
| --- | ---: |
| 两槽 packed 输入 | 20480 |
| beta 输入 | 128 |
| 两槽 FP32 State 输入 | 65536 |
| 一槽 FP32 State 输出 | 32768 |
| 两槽 out | 256 |
| tmpBuff | 84928 |
| scalarBuf | 64 |
| 合计 | **204160** |
| host 报告的可用 UB | **196352** |

分配总量比可用 UB 多 **7808 B**；即使考虑 host 的 beta 对齐富余和 2 KiB guard，也不足以吸收少算的一槽。按同一顺序，gate 工作区结束于偏移 `201728`，scalarBuf 位于 tmpBuff 之后，均超出可用 UB。这一静态不一致与本轮设备异常一致。

上述预算缺陷已经从代码及实际 tiling 数值确认；尚未通过修正后重测证明它是所有异常的唯一原因。后续应先按 A2 kernel 的真实双槽输入、单槽 beta、State/out 队列、计算区及对齐重新核算预算，再重新构建并从同一冒烟集合开始验证。A5 采用独立预算路径，本记录不推断其性能变化；A3 尚未上板。

## PR 引入来源核查

GitHub API 核对 PR #997 的 base 为 `a7c608f2e3f828779ceeec1fcf155a6e37ede83c`，head 为本次被测 `be0ecad8a739de5b205d1a229058d58d2407a89d`，本地 merge-base 与该 base 一致。对照两端代码及逐提交 diff，确认上述 UB 预算回退由 PR 内的提交 `2e29351f135ba3a57eb26a34345d52ab668fc7ef` 引入：

1. base 的 `CalcFixedUbBytes` 为 A2/A3 使用 `rawBufferNum=2`，并按实际 NV 计算单槽 beta；该提交将 packed 输入预算改为单槽。后续 `d23fb9e8` 添加 `windowBufferNum=isA5?2:1`，A2/A3 仍是一槽。
2. base 的 `FinalizeVStepFromUb` 在 A2/A3 分支校验 `tmpBytes <= ubRestBytes`，并只分配具名计算区需要的 `tmpBytes`。`2e29351f` 删除了这段校验和收缩逻辑；当前 A2/A3 使用按少算的 fixed 区推导出的全部剩余字节。
3. kernel 的 `PACKED_INPUT_BUFFER_NUM=2` 和对应 `InitBuffer` 来自已合入的 `d04a1e3c`（PR #917），base 中已存在。预算改变时，这一实际分配没有改变。

按 T=2 复现的同一硬件容量和 shape，对 base 与 head 的 host 公式做资源核算（这是代码算术对照，并非 base 上板结果）：

| 项目 | PR base | 当前 head |
| --- | ---: | ---: |
| vStep | 48 | 64 |
| State 输出队列深度 | 2 | 1 |
| out 输出队列深度 | 2 | 2 |
| tmpBuff 字节 | 66048 | 84928 |
| kernel 总分配字节 | 185216 | 204160 |
| 可用 UB 字节 | 196352 | 196352 |

因此可以确认：当前发现的 UB 预算不一致是本 PR 引入的 A2/A3 回退；编译器标准头文件和 ATK 比较器缺失是另外的环境问题。尚未进行 base 上板对照或修复后回归，不能由此声称已完成所有设备异常的动态因果验证。本轮只记录归因，不修改实现。

## 证据与范围

私有证据保存在本次任务的 `output/rkda-a2-test-20261031/`，含构建安装日志、两次环境失败日志、八例设备错误、T=2 的详细 tiling 日志、用例来源与 JSON、复现脚本、设备快照及安装对象身份。`evidence.zip` 收录 79 个文件和逐文件 SHA256 清单，下载后 CRC 与全部 hash 校验通过；包 SHA256：

```text
0ed25e49050a7b8f4dbb0bd3bae652bfd2fc11803548090eeb6b590b1f66545d
```

该目录不是公共下载工件。首轮只记录失败与归因，未修改实现或推送 PR；后续修复验证在下节另记。

## 2026-10-31 修复、rebase 与复验

先 rebase 到远程 main `34851b76c9986aeb9dd9826a4ca57c3ba0c205f1`，原6个提交等价重放后 head 为 `c7af2aaa`。停止旧源码构建，从完整 rebased 源码重新构建 wheel，安装对象与 Python 包身份核验通过。主线更新未改变 recurrent_kda 实现及 CPU 标杆。

修复 generic host 的双槽 packed 输入、按实际 NV 计算 beta/具名 FP32 窗口区，恢复 `tmpBytes <= ubRestBytes` 校验并只分配所需 tmpBytes。A5 的独立资源规划函数逐字不变；另行采用的 X=1 两遍读取仅属于 A5，不影响 A2 generic kernel。

| 检查 | A2 910B3 物理卡4、CANN 9.3.0 weekly 实际结果 |
| --- | --- |
| 原失败8例：V=128/256、两布局、两种 State dtype，T=9 | 8/8 SUCCESS，out/State 精度均 true |
| 原132例、54对齐、36契约及12项V=256回归 | 234/234 SUCCESS，out/State 精度均 true |
| 17项确定性，每项3轮 | 17/17 SUCCESS，一致性字段均通过 |
| 公开契约及三入口 review 脚本 | 未全部通过：A2 无 indices 的 packed 输入未拒绝非法 cu_seqlens 起点 |

原 Vector/Scalar 地址异常在上述成功集合中不再出现，支持 UB 预算修复的动态归因。A2 无 indices 时 `MaterializeCuSeqlens` 沿用原路径、不做设备 cu 值的 host 快照校验，因此两个扩展脚本在 `cu_start` 处失败。该分支不在本轮预算修复范围，PR base 也没有新增的 host cu 值物化；保留失败记录，不修改用例或用进程返回0覆盖失败，不宣称所有公开入口已验收通过。

未改变 CPU 标杆、输入值域和阈值，未测 A2 性能、A3、sanitizer 和全仓 ST。私有证据归档标识为 `output/rkda-a2-ub-fix-20261031/`，包含原始 ATK 报告、失败脚本、构建安装、rebase 影响核对及 tiling 日志。后续 host 日志统一另记。

## 后续检视：CU 校验补齐

按 operator-review 的有效意见，移除 `MaterializeCuSeqlens` 对 A2/A3 无 indices 场景的跳过条件；只要存在 CU 就保存 host 快照并禁用旧 executor 缓存。该修复增加这些 packed 调用必要的同步/D2H；A5 的原 CU 值依赖路径不变，固定长度且无 indices 不增加此成本。未修改 generic kernel 算术或 CPU 标杆。

重新构建并安装 `cust_opapi`，核对与构建库 SHA256 一致。最终74项公开契约（含同 descriptor 修改 CU）、59正常/49负向三入口回归、16项原始 ACLNN 报错及拒绝后四路径合法调用全部通过；再次运行234项双输出精度及17项各3轮确定性，全部通过。因此上节 cu_start 失败属于保留的中间记录，不代表最终版本结果。插桩内存检查及最新 main 的 A5 性能见[最终检视验收记录](recurrent-kda-pr-review-acceptance-20261031.md)。

## 最终补齐验收

所有平台 CU 值校验/缓存修复后，A2 最终234项双输出 CPU 精度、17项各50轮确定性、17项真实插桩 memcheck 均通过。memcheck 独立标准构建72个 sanitizer 设备对象，17次目标 kernel 启动/完成均报告无错误；没有将正常 wheel 当作插桩包。

随后补齐 stable/ctypes 的框架元数据生产者依赖，最终 wrapper 的20项默认/独立 stream 就绪专项、74项公开契约及59正常/49负向三入口回归通过。最终构建、安装库身份与未覆盖范围见[最终检视验收](recurrent-kda-pr-review-acceptance-20261031.md)。上述结果取代本记录早期 cu_start 失败的中间结论，原失败日志继续保留。
