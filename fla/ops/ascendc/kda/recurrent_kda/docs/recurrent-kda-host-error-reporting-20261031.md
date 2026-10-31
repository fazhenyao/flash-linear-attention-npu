# RecurrentKda host 报错接口统一

## 范围与结论

在 rebase 后 PR head `c7af2aaa` 和 A2 UB 预算修复基础上，扫描整个 `op_host/`，包含 ACLNN、L0、tiling 和 InferShape。原实现未统一：公开空指针/dtype 校验使用 opdev 包装宏；36处 `CHECK_RET/CHECK_COND` 混合了条件判断、PLOG 和泛化 ErrorManager 上报；另有 Cube 内部报错封装。

本轮已将本算子直接编写的错误出口统一到实际 CANN `op_common/log/log.h` 接口，保留公开 errno、输入校验条件及顺序。没有修改公共 ABI、CPU 标杆、值域、容差、tiling 数值或设备计算。X=1 两遍读取为另一个独立修改，见[复验记录](recurrent-kda-x1-two-pass-20261031.md)。冻结方案未修改。

| 根因/层次 | 采用接口 | ErrorManager/返回值 |
| --- | --- | --- |
| 必需 Tensor、workspaceSize、executor 为空 | `OP_LOGE_WITH_INVALID_INPUT` | EZ0004；Tensor 仍161002，workspaceSize/executor 仍161001 |
| Tensor dtype 不支持 | `OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON` | EZ0020，包含参数名、实际 dtype 和支持集合；仍161002 |
| shape、stride、gate、metadata 等参数关系错误 | 既有 `ReportInvalid` 或 `OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON` | EZ0037；仍161002 |
| 必需 tiling/InferShape 输入形状或描述为空 | `OP_LOGE_WITH_INVALID_INPUT` | EZ0004，使用稳定输入名；仍 GRAPH_FAILED |
| Runtime、Contiguous、ViewCopy、executor/view/host Tensor 构造失败的外层检查 | `OP_CHECK_IF` 配合 `OP_LOGE_LIBOPAPI_REPORT` | 仅补 PLOG，保留原 errno 映射；不额外覆盖下层首错 |
| framework context、输出描述、平台、attrs、compiled info、workspace 或序列化内部不变量 | `log.h` 的 `OP_LOGE` | 保留内部诊断，仍 GRAPH_FAILED；不伪装成用户 Tensor 错误 |

对 `CheckNotNull/CheckDtypeValid/CheckShape/CheckQkvMaterializationSafety` 已经上报的公开参数错误，删除外层重复条件日志，直接返回原 errno。L0 launcher 失败也只用 opbase PLOG 补充原 ret。公共框架 `L0/L2_DFX`、allocator、Runtime 和 `CommonOpExecutorRun` 的内部实现不在本算子源码整改范围，沿用其自身错误处理。

## 实际头文件和宏定义证据

依据两台服务器受影响 target 的真实 `flags.make`、编译 include 路径及实际头文件，而非仅看宏名：

- A2：`${CANN_ROOT}/aarch64-linux/include/op_common/log/log.h`。
- A5：`${CANN_ROOT}/x86_64-linux/include/op_common/log/log.h`。
- `OP_LOGE_WITH_INVALID_INPUT` 展开到 EZ0004，字段为 `param_name,op_name`；dtype reason 宏展开到 EZ0020，字段为 `param_name,op_name,incorrect_dtype,reason`；一般参数 reason 宏展开到 EZ0037。
- `OP_LOGE_LIBOPAPI_REPORT` 在同一 `log.h` 第106行定义，只写 PLOG；它也是该头文件公开语义宏和 `OP_LOGE_WITHOUT_REPORT` 的实际日志叶子。ACLNN 文件同时包含 `opdev/op_log.h`，后者重新定义同名 `OP_LOGE_WITHOUT_REPORT`，因此本轮显式调用 opbase 日志叶子，避免依赖 include 顺序猜测同名宏来源。
- 当前 CANN `opdev/op_log.h` 的 `CHECK_RET` 只写 PLOG，并不新建 ErrorManager 首错；`CHECK_COND` 则通过 errno 包装生成泛化上报。不能把二者一概描述为重复上报。
- `log.h` 的通用 `OP_LOGE` 展开到内部诊断 EZ9999。仅在上述 framework 不变量路径保留；没有把 Runtime 返回错误再次改报为泛化内部码。

用户给出的 GitCode 网页在检索工具中不可访问，以上采用实际安装头文件、构建和负向运行输出作为版本证据。真实运行验证了 EZ0004/EZ0020/EZ0037 的模板字段替换；未自行拼接正式码文本或调用未注册的新码。

## 编译与测试

A2 910B3 物理卡4、A5_1 Ascend950PR 物理卡5，均使用任务私有 Python 环境。先构建 `ophost_transformer_opapi_obj`、`ophost_transformer_tiling_obj`、`ophost_transformer_infer_obj`，两平台全部通过；随后标准完整 wheel 构建安装及 Python 包身份核验通过。最后清理重复日志和统一剩余 Cube 入口后，仅重建受影响 `optiling/cust_opapi` target，将准确对应的库安装至私有 wheel 的 OPP，核对构建/安装库 SHA256 相同，再执行最终测试。最终安装包不与稍早生成的 wheel 文件混同。

最小重建命令：

```bash
cmake --build build --target optiling --parallel 8
cmake --build build --target cust_opapi --parallel 8
```

复用现有 `_runtime` 的 ACL descriptor 构造和 ATK 输入/CPU 比较器，定向调用原始 ACLNN，绕过 Python 参数预检。16项负向覆盖8个必需 Tensor、workspaceSize/executor、dtype、head ratio、State stride、cu 起点、负 slot、accepted=0。A2 的 cu 用例显式带 indices，覆盖实际执行 host 元数据校验的路径。

| 平台 | 16项原始 ACLNN 负向 | 错误码及参数字段 | 首错唯一性 | 拒绝后四路径合法调用 |
| --- | --- | --- | --- | --- |
| A2 | 全部通过 | 原 errno，EZ0004/EZ0020/EZ0037，参数名及具体原因 | 每例目标码只出现一次，无额外 EZ9999/EZ9903/EZ1001 | 两布局、两 dtype 的 out/State CPU 精度通过 |
| A5_1 | 全部通过 | 同上 | 同上 | 同上 |

A5 另通过74项公开契约及 mutation/grad；最终三入口回归为59项正常、49项负向，通过新增的具体 EZ 码及禁止泛化覆盖断言。一次较早完整运行曾未拒绝 stable 的 accepted=0；孤立的原包/当前包各6次检查都拒绝，最终两次完整回归均通过，保留原失败日志，不据此声明已证明偶发问题根因或消失。

A2 UB 修复的8例冒烟、234例精度及17例各3轮确定性见[A2记录](recurrent-kda-a2-test-20261031.md)。首次日志整改时，A2 无 indices 的非法 cu_seqlens 尚未被公开入口拒绝，扩展回归失败未被归为“日志宏已修复所以通过”。随后按 operator-review 的独立功能意见补齐该平台的 CU 值校验及缓存处理，两平台重新通过74项公开契约和59正常/49负向回归，见[最终检视验收记录](recurrent-kda-pr-review-acceptance-20261031.md)。A3 未上板。

## 候选台账与静态复核

首次修改报错源码前冻结整个 op_host 的9个源文件及59个治理候选，台账保留原 hash 和稳定 ID。逐项处置：44 FIXED、4 REMOVED、11 RETAINED，0 PENDING/BLOCKED。12处必需输入空指针检查合并到一个具名循环；固定台账对每个原 shape/descriptor 检查显式映射，未通过缩小扫描范围避开候选。

```bash
python check_candidate_dispositions.py check --ledger candidate-dispositions.json
```

实际退出码0，输出 `Candidate disposition reconciliation: PASS`。这只证明处置结构与复扫对账，不证明未执行的故障注入。残留11项为上述 framework 内部不变量；全部使用真实 opbase `OP_LOGE`，保留理由在台账和本报告中对应。

仓内 operator-review 的 issue 覆盖为188项 completed issue、21条规则，校验通过。机械静态检查仍返回非零：9项 tiling 通用 `OP_LOGE` 命中，均为 framework 内部诊断；另2项 InferShape 内部诊断由错误码扫描器识别。没有宣称零发现。最终错误码审计与逐候选台账保存在同一私有证据目录。

## 验证边界与关闭结论

后续检视补齐 stable/ctypes 的 PyTorch 生产者就绪依赖：存在 CU/indices 时，在进入 ACLNN host 快照前等待当前框架 stream；直接 ACLNN 由调用方保证元数据已就绪。两平台各20项不手工预同步的默认/独立 stream 专项，以及最终74项公开契约、59正常/49负向回归通过。此举修复缺失的显式依赖，不将此前一次 accepted_zero 漏拒绝的未稳定复现机制认定为已经证明的唯一根因。详见[最终检视验收记录](recurrent-kda-pr-review-acceptance-20261031.md)。

报错接口统一及公开参数代表性验证已完成。**完整错误码治理是否可关闭：否**。尚未对 Runtime/allocator 失败、损坏的 gert context、输出描述缺失、平台/attrs 缺失及序列化容量失败执行故障注入，因此没有动态证明这些内部路径的首错唯一性，也未证明所有 Runtime/allocator 调用都在下层正式首报。不能用公共参数的16项通过替代这些证据；没有新增不准确正式码来使审计器归零。

host 报错修改不改变设备代码，无需据此重复 msprof；X=1 改动使用独立复验记录。私有任务证据标识 `output/rkda-host-log-20261031/` 包含冻结源文件、59项处置台账、实际头文件/flags、构建安装库身份、负向 JSON、完整/失败回归日志和静态审计；该目录不是公共下载地址。
