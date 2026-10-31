# PR #997 检视、整改与双平台验收

## 版本与范围

使用远程 main `34851b76c9986aeb9dd9826a4ca57c3ba0c205f1` 的 [operator-review skill](https://github.com/flashserve/flash-linear-attention-npu/tree/34851b76c9986aeb9dd9826a4ca57c3ba0c205f1/docs/agents/skills/operator-review)。PR 原 head 为 `be0ecad8`；rebase 后六个提交等价重放至 `c7af2aaa`，本记录覆盖其后的 A2 预算、host 报错整改及 X=1 两遍读取。冻结的用户方案未修改。

检视以 PR 相对最新 main 的差异为主，包含共用 host、A2/A3 generic 路径、A5 四条计算路径，以及 stable/ctypes/直接 ACLNN。CPU 标杆、值域、混合容差和公开参数列表不变；明确的 layout/gate 行为变化保留用户已确认的语义，不以签名一致冒充行为完全兼容。

## 有效意见与整改

### 1. A2/A3 资源预算与实际双槽分配不一致

【review】

问题分类：兼容性问题

精度级别：lv3

细粒度根因：内存容量/超限

位置：`op_host/recurrent_kda_tiling_processor.h:801` 的 `CalcFixedUbBytes`、`:812` 的 `CalcWorkingUbBytes` 及 `:923` 的 tmpBytes 检查。

问题：PR 的共用 host 只预算一槽 packed 输入，而 generic kernel 使用两槽；同时缺少 tmpBytes 容量约束。（额外发现）

可能导致的结果：A2 K=V=128、普通 T=2 的实际分配204160 B超过可用196352 B；T=9、两布局、两 State dtype 及 V=256 冒烟发生 Vector/Scalar 地址异常。

建议的修改方法：按 generic kernel 的实际两槽、NV 和具名 scratch 分配恢复预算及容量校验，保留 A5 独立资源规划。

处置：已修复。原失败8例通过；最终234项双输出精度及17项各3轮确定性通过。详见 [A2 原始失败及修复记录](recurrent-kda-a2-test-20261031.md)。A3共用该资源修复，但没有 A3 上板结果。

### 2. A2 packed 无 indices 时未校验 CU 实际值

【review】

问题分类：编码红线问题

精度级别：lv1

细粒度根因：shape/校验缺失

位置：`op_host/op_api/aclnn_recurrent_kda.cpp:655` 的 `MaterializeCuSeqlens` 入口和 `:991` 的 `valueDependent` 条件。

问题：用平台及 indices 条件提前跳过 CU 校验；无 indices 的 A2 packed 调用同时允许复用与 CU 值无关的缓存。（额外发现）

可能导致的结果：非法 CU 起点、逆序和越容量输入不能在 host 按契约拒绝；相同 descriptor 修改 CU 后可能继续使用旧执行计划。A2 的公开及三入口回归实际停在 cu_start。

建议的修改方法：所有平台只要传入 CU，就读取并校验实际值、持有快照并禁用旧 executor 缓存。

处置：已修复。A2/A5 最终均通过74项公开契约（含同 descriptor 修改 CU）、59项正常及49项异常三入口回归。API/design 已说明 A2/A3 无 indices packed 路径新增必要的同步/D2H；固定长度且无 indices 不增加该成本。A5 的原 CU 校验路径不变。

### 3. Host 失败出口未统一使用实际 opbase 接口

【review】

问题分类：cleancode

精度级别：lv1

细粒度根因：空值/null

位置：`op_host/op_api/aclnn_recurrent_kda.cpp:472` 的空指针与 `:488` 的 dtype 校验、`op_host/recurrent_kda_tiling.cpp:263` 的必需输入检查、`op_host/recurrent_kda_infershape.cpp:47` 的输入形状检查及 L0 失败传播。

问题：公共参数检查及错误传播混用 opdev、Cube 和 opbase 封装，部分传播重新产生泛化错误，不能仅靠宏名确认首报归属。

可能导致的结果：空指针、dtype 等失败出口缺少一致的参数模板；上层重复泛化上报可能覆盖下层原因。

建议的修改方法：解析两平台实际 `op_common/log/log.h`，参数错误使用具名结构化宏，下层失败只补 opbase 日志并保留原 errno。

处置：已整改。冻结的59项候选逐项对账通过；两平台各16项原始 ACLNN 负向断言具体 EZ 码、字段及唯一首错，拒绝后四路径 CPU 精度通过。framework 内部故障诊断保留内部码；未用参数错误冒充框架损坏。内部故障注入的未覆盖范围见 [host 报错记录](recurrent-kda-host-error-reporting-20261031.md)。

### 4. 公开记录包含机器绝对路径

【review】

问题分类：cleancode

精度级别：lv1

细粒度根因：责任田补充根因

位置：`recurrent-kda-kfirst-ub-padding-20261029.md:124`、`recurrent-kda-upstream-layout-compare-20261029.md:234`、`recurrent-kda-vfirst-pipeline-a5-1-20261029.md:86` 及本轮 host 报错记录 `:24`。

问题：结果位置使用本机绝对目录或安装绝对目录，公开后不可复现。（额外发现）

可能导致的结果：读者把私有路径误认为可访问工件，并泄露无关的机器目录结构。

建议的修改方法：改用私有证据标识及相对 CANN_ROOT 的版本路径。

处置：已修改。冻结方案原有的用户参考文件链接按用户“不要修改方案文档”的指令保留，作为明确的范围例外。

### 5. Python 生产者与 host 元数据快照缺少显式就绪依赖

【review】

问题分类：编码红线问题

精度级别：lv3

细粒度根因：同步/依赖缺失

位置：Python `_stable.py`、`_aclnn_ctypes.py` 的 recurrent_kda 调用边界，以及 `_recurrent_kda_compat.py` 的算子局部 helper。

问题：GetWorkspaceSize 在 host 读取 CU/indices/accepted，ACL runtime 同步不能保证 PyTorch 队列中的生产者已提交。此前一次 accepted_zero 漏拒绝尚不能仅凭复跑通过关闭。（额外发现）

可能导致的结果：host 校验或快照与框架元数据生产者缺少显式顺序。

建议的修改方法：两个 Python 入口仅在 CU/indices 存在时等待当前框架 stream，保留固定长度无元数据的异步路径；直接 ACLNN 明确调用方就绪前提。

处置：补齐上述依赖。两平台各20项专项覆盖 stable/ctypes、默认/独立 stream、定长带 indices/packed、正常 CPU 双输出及排队生成的非法 accepted/CU；调用前不手工同步。随后各74项公开契约及59正常/49负向回归通过。此前偶发失败的准确触发机制未被稳定复现，未声称由此证明其唯一根因。该修改不改变 host/kernel 二进制，72个计时设备对象保持一致；新增等待的调用链成本不计入 msprof 目标 kernel 结果。

## 四层复核证据

| 规则 | 核对依据与结果 |
| --- | --- |
| FUNC-SYNC / FUNC-DETERMINISM | A5 pipeline 文件头 Stage/事件表；bank 按状态接管，覆盖前等待 MTE3，最终排空；确定性实际结果见下表 |
| FUNC-MEMORY | A5 独立 PlanA5Ub 与实际偏移公式、K-first pitch；A2 两槽和 tmpBytes 已修复；scratch 与事件字不重叠 |
| FUNC-VARLEN / FUNC-SHAPE | CU 首项/单调/容量、空请求及最大有效长度选择 X<=8；公开旧 rank 别名先归一化；CU 值依赖修复见意见2 |
| FUNC-DTYPE-FORMAT / FUNC-PRECISION | 四条 layout/dtype 路径、非连续合法 view、FP16 辅助输入；State 按接口 dtype 在 UB 保留，计算使用 FP32，CPU 标杆及容差不变 |
| FUNC-STATE / FUNC-OPTIONAL-OUTPUT | 两遍读取不改变 token 依赖；原位独立输出 ViewCopy、非原位完整 State pool 初始化、输出别名拒绝及未写槽保留均有回归 |
| FUNC-REGRESSION | 公开74项、三入口59正常/49负向及拒绝后合法调用；补齐 Python 元数据生产者依赖并专项验证，保留此前偶发失败及未稳定复现的限制 |
| DFX-VALIDATION / DFX-MESSAGE | 具名空指针/dtype/参数关系错误及 errno 双向核对；静态剩余9项为 framework 内部诊断，另2项 InferShape 内部诊断由错误码扫描识别 |
| ENG-STRUCTURE / ENG-BUILD / ENG-COMPATIBILITY | 独立算子工程、薄 kernel 入口、arch35 context/vec/pipeline 分层；标准 wheel/run 构建、最终受影响 target 重建及安装库身份核验；A2 与 A5 分开构建 |
| DOC-FORMAT / DOC-CONSTRAINT | API/design/ATK 记录与 CU、X=1 两遍及 A2 资源修复对齐；机器路径清理；冻结方案不修改 |
| TEST-COVERAGE / PERF-REGRESSION | 保留 CPU 标杆/值域/容差；最终精度、异常、确定性、插桩内存检查及最新 main 四路径性能分别记录 |
| CI-WORKFLOW / FEATURE-DELIVERY | 无 workflow/CODEOWNERS 变更；不把本地验收冒充正式 NPU CI 或 Example/ST |

Issue corpus 的188项 COMPLETED issue均映射到21条规则，机器校验通过；当前快照无 `op:recurrent_kda` 专属已完成 issue，仍逐项检查跨算子规则。issue680作为模型性能输入覆盖，不将其状态混入上述计数。

算子静态扫描保留9项 framework 内部报错；PR diff 扫描另有12项 `DOC-SENSITIVE` 误将 性能采集参数的字符片段识别为凭据，人工核对为误匹配，不是真实凭据。另2项断链是冻结方案中用户给出的本地参考链接，按用户指令不修改。Windows 原检查器会在解析这些链接时触发 WinError 123；私有只读适配仅将无效 Windows 路径保留为不存在的链接，未修改 skill 或过滤意见，实际报告仍包含上述23项命中。未宣称机械扫描零发现。

公开签名通过 AST 与 `v26.9.1`、原 PR base `a7c608f2`、最新 main `34851b76` 比较，参数名、顺序、默认值、参数种类相同；发布契约的返回槽位保留 `(out, final_state)`。stable/ctypes/直接 ACLNN 的形状别名、返回 shape、原位与非原位语义均以动态回归核对。layout/gate 的有意行为变更及迁移说明在 [API](api.md) 明确列出。

## 正常构建验收

所有记录均核对实际安装库，未把标准 wheel 文件与稍后替换的 host 库混为同一产物。源码18个 host/kernel 文件的归一化 SHA256 与最终本地源码一致，72个设备对象与正常性能采集包逐项匹配；两平台的库和设备对象分别核对，不跨平台比较二进制。

| 检查 | A5 Ascend950PR / CANN 9.2.0-beta.2 | A2 910B3 / CANN 9.3.0 weekly |
| --- | --- | --- |
| 双输出 CPU 精度 | 188/188（186+2项 X=1 BF16），out/State 字段均 true | 234/234，out/State 字段均 true |
| 确定性 | 最终11项，各50轮，全部通过 | 最终17项，各50轮，全部通过 |
| 公开契约及 mutation/grad | 74项，通过 | 74项，通过 |
| 三入口扩展回归 | 59正常、49负向，通过 | 59正常、49负向，通过 |
| 原始 ACLNN 报错及拒绝后恢复 | 16负向、4条健康路径，通过 | 16负向、4条健康路径，通过 |
| Python 元数据就绪专项 | 20项，通过；随后公开与三入口回归再次通过 | 20项，通过；随后公开与三入口回归再次通过 |
| 插桩 memcheck | 11项，无错误 | 17项，无错误 |

精度/确定性任务退出正常，逐项解析实际 JSON 报告后判定；没有只依据进程返回0。精度覆盖说明及历史失败保留在 [ATK README](../../../../../../tests/atk/recurrent_kda/README.md)。

A5 的内存检查使用单独的 sanitizer 构建，72条生成编译命令均包含 `--op_debug_config=sanitizer`，72个设备对象均不同于正常计时包。原9项及补充的 X=1 BF16 fused/general 两项共11项，通过实际 ATK JSON、11次目标 kernel 启动/完成及 `No error detected` 三者核对。补充两项已进入正式 MSS JSON 和 generator，未为了补测而重新采集全部性能。正常包保留在原私有环境，插桩包通过独立 PACKAGE/OPP 路径加载。
A2 同样使用独立的完整 sanitizer wheel 构建，72条生成命令及72个对象与正常包分别核验；17项实际报告通过，17次目标 kernel 的启动、完成和 `No error detected` 逐项匹配。未将未插桩的正常构建冒充 memcheck。确定性50轮、CPU 精度和 memcheck 为三个独立任务，均保留原始日志；Python 就绪补充只影响调用边界，其20项及公开/三入口回归在最终 wrapper 上执行。

## 最新 main 的分路径性能

issue680 全36形状×四路径共144组，同卡串行 msprof，每组3预热+5正式；均值均低于 main。V-first BF16、K-first BF16、K-first FP32 的36组耗时比几何均值分别为0.761890、0.422191、0.351471。V-first FP32 首轮为0.690407，但 main 有7组 CV>5%；仅复核这7组后最大 CV 为0.74%，对应29组首轮加7组复核的诊断汇总为0.708479。两轮原样本全部保留，不隐去首轮噪声。完整36形状表、7组复核表及统计边界见[性能对照记录](recurrent-kda-upstream-final-compare-20261031.md)。

## 未覆盖范围

A3、整体 Example/ST、旧 CANN 版本和正式 NPU CI 未执行。Runtime/allocator 及损坏 gert context 的故障注入未执行，不能声称完整错误码故障治理已关闭。设备允许共享，目标 kernel 计时不包含 Python/host CU 和索引快照成本。

## 私有证据归档

证据标识为 `output/rkda-final-review-20261031/`，不是公共下载链接。三个压缩包均已逐文件核验 CRC 和 SHA256 manifest：

| 工件 | 文件数 | 压缩包 SHA256 |
| --- | --- | --- |
| perf-evidence.zip | 196 | `98da360d4eeca8f62e906debd302b4bb195cda0c9d98010dd6bf0e4a92e2f0b3` |
| a5-review-evidence.zip | 299 | `8a0533939fa69251b4e7cdf5e0a80e218834b0eb5523acef2b3a1ce8e3f734dc` |
| a2-review-evidence.zip | 40 | `c842669f84a8b6d7e7a40cb7938c645123fe21dd06f799da7e0c6171be1afc51` |

计时 CSV、输入 manifest、库/设备对象身份、正常与插桩报告、失败历史及最终 Python helper 的安装 SHA 分别保留。完整 tensor 输出留在远程任务目录；压缩归档不复制这些大文件。最后新增的 Python 等待与已采集设备对象无关，最终三文件源码/安装归一化 SHA 已匹配。

## 后续检视意见

本记录的全量验收对应 `ecc69ea6`。随后针对 discussion_r4430769129 移除元数据显式全设备同步；增量验收、未采用候选、12组四路径 kernel 对照及多流/连续调用结果见[同步修复](recurrent-kda-metadata-stream-fix-20261031.md)。新一轮未重复完整矩阵和 sanitizer，设备对象身份保持一致；两轮证据分开保留。
