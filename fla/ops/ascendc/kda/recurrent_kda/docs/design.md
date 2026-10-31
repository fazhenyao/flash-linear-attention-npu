# RecurrentKda 设计方案

方案设计规则版本：`V2`

接口依据为 [api.md](api.md)。A5 的完整计算、UB 布局、bank 生命周期、预取与同步方案采用用户确认的[寄存器方案](recurrent-kda-a5-regbase-redesign-20261026.md)，该文件为本轮开始时的原样快照。

实现以 PR #997 的 `d23fb9e8` 为基线，差异、验证计划及结果记录到[方案对齐记录](recurrent-kda-plan-alignment-20261029.md)。本轮不修改用户方案。

A5 单 Vector Stage 按 tile → token 执行。V-first以四个V行的归约结果作为out收集粒度，计算分组不受四行约束；所有X均采用下文的两遍State读取及按State dtype分别选择的交错调度。K-first FP32/BF16采用连续V方向两遍计算。K-first保持GM接口布局，在UB按[行距优化补充](recurrent-kda-kfirst-ub-padding-20261029.md)增加32B行间空隙。State在UB保持接口dtype，Cast在寄存器中完成；两个bank交替递推，另一个可复用bank预取。两个FP32 out寄存器保存完整token，最后两次Cast/Store到UB；tile完成后批量提交MTE3。

定长 X 按 S，变长按各请求最大有效长度选择，候选为1/2/4/8。cu_seqlens 是 host 值依赖；改变元数据须重新 tiling。A5 资源峰值为 `199744+3072*X+2*max(128,32*X)` 字节；FP32 K-first 额外增加12288字节，BF16 K-first 与 V-first 无额外 bank 分配。不增加算法 workspace。

所有平台的 packed 调用都在 host 校验 cu_seqlens 的首项、单调性和容量，保留 executor 自有快照，并禁用旧 executor 缓存。A2/A3 的无 indices packed 路径也执行该校验，因此新增必要的 host 同步/D2H；固定长度且无 indices 不增加此开销。A5 还用快照计算最大有效长度及 X。
Python stable/ctypes 入口在 CU/indices 存在时先等待当前 PyTorch NPU stream，补足框架生产者到 ACLNN host 快照的就绪依赖。该处理局限于 recurrent_kda，不修改公共 runtime；固定长度且无 indices 不等待。直接 ACLNN 的元数据就绪责任由调用方承担，详见 API。

A2/A3 共用序列模式、gate 选择及 inactive raw-gate 参数忽略规则，保留对应硬件计算与资源路径；A5 的计算分组、三 bank 容量及 X 预算不套用于 A2/A3。

A2/A3 的 UB 预算按 generic kernel 的实际分配计算：固定窗口 X=8，两槽 packed q/k/v/g 输入共 `2*X*(8*alignK+2*alignV)` 字节，beta 单槽为 `4*X*NV` 字节，另留64字节 scalarBuf 和2048字节 guard。具名 FP32 窗口工作区为 `X*(4*alignV+12*alignK+4*NV)` 字节；加上每个 V 行的 State、broadTmp 和两个行结果后得到 tmpBuff 所需容量。host 必须先验证 tmpBuff 不超过扣除真实输入及 State/out 队列后的剩余容量，再只分配所需字节。该修复恢复 PR base 的资源计算，不修改 A5 的独立 PlanA5Ub 或冻结方案。验证记录完成后归档到 ATK README。

后续 VF 指令和任务边界调度采用[有序优化补充](recurrent-kda-vfirst-ordered-optimizations-20261030.md)：每 token 将128个 v 批量转成 FP32。V-first复用 `[eventOffset+256,eventOffset+768)` 的512B尾部预留区：所有X均保存下文第一遍形成的delta。VST→VLD 后由行循环消费，下一个 token 才覆盖；K-first直接消费两个FP32 v寄存器。事件字占用前36B，与 scratch 不重叠；UB 总峰值公式不变。V-first四行 State Store 先于 out 收集，按64元素片段交错 State 加载/转换/乘法。当前 tile 必要同步后先 Dispatch VF，再提交另一槽输入和可用第三 bank 的预取，覆盖前等待与尾部排空保留。State 的 UB dtype 和整 tile out 批量搬出规则不变。

前轮寄存器指令优化见[独立实验补充](recurrent-kda-a5-register-optimizations-20261030.md)：V-first与当时的K-first BF16每token预计算有效beta与k的两个公共乘积，行内通过MulAddDst更新State；当时K-first FP32保留原乘加顺序。该变化只合并或重排FP32寄存器运算，可能改变舍入；沿用原CPU标杆和精度标准验收。当前V-first继续按该方式计算，K-first已替换为下文两遍流程。

前轮[分布局寄存器优化](recurrent-kda-layout-register-optimizations-20261030.md)采用V-first和K-first BF16的prediction/out点积融合：以一条Mul和一条MulAddDst合并两个64列片段，再按原归约树求和。当时K-first BF16按成对Gather、解包/Cast、衰减和预测交错执行，写回前用每VF初始化的`0xFFFF`寄存器掩码保留低半字，替代左右移；该BF16路径现已由连续V方向两遍计算替换，V-first继续保留点积融合。FMA的舍入变化通过原CPU标准验收；V-first FP32短序列性能存在跨轮波动，记录保留完整采集及复测，不能宣称所有用例稳定加速。

当前K-first FP32采用[连续V方向寄存器优化](recurrent-kda-kfirst-fp32-register-optimizations-20261030.md)，BF16采用同一计算流程，见[BF16验证记录](recurrent-kda-kfirst-bf16-register-validation-20261030.md)。每个FP32寄存器覆盖64个连续V元素，两遍均从当前State bank读取；BF16在Load后于寄存器转FP32。第一遍沿K累计`prediction=Σ State*(alpha*k)`；随后两个向量计算`delta=beta*(v-prediction)`。第二遍逐K计算`State_next=alpha*State+delta*k`，先累计`out=Σ State_next*q`，再以接口dtype写另一bank；BF16写回使用Cast_RINT，不从已舍入BF16结果计算当前token的out。q已包含归一化和scale，alpha、beta沿用门控及sigmoid规则。第二遍系数从q/k/alpha寄存器中Gather广播；第一遍的prediction系数按下文ks54方式从UB广播。State通过连续Load/Store访问，省去State Gather/Scatter、逐行归约和四行out收集。FP32顺序累计及FMA的舍入变化按原CPU标准验收。

K-first每token的State UB有效读取量：FP32由64KiB增到128KiB、BF16由32KiB增到64KiB，写入分别仍64KiB/32KiB；GM流量、UB总峰值、State bank所有权和整tile out批量搬出保持不变。K-first两种dtype直接消费两个FP32 v寄存器，省去vFloat scratch写读及其专用VST→VLD屏障，其余必要同步保留。

V-first采用[两遍读取验证](recurrent-kda-vfirst-two-pass-validation-20261030.md)中的v2r34。X=1也采用[两遍读取复验](recurrent-kda-x1-two-pass-20261031.md)中的已验证流程，所有X均使用两遍；公开TilingKey及fused/general分派规则不变。变长输入按最大有效请求长度选择公共X，X>1调用中的短请求和单token尾tile仍执行两遍。

两遍路径第一遍只读当前State bank，计算`prediction[v]=Σ_k State[v,k]*(alpha[k]*k[k])`，按四行收集到两个FP32寄存器，再批量形成`delta=v-prediction`。将delta以两次VST写入原512B scratch，保留专用VST→VLD屏障；delta存活到本token最后一行消费完成，下一token才能覆盖。第二遍重读同一源bank，计算`State_next=alpha*State+delta*(beta*k)`，使用未舍入的FP32更新值计算out，再以接口dtype写另一bank。BF16在寄存器读入后Cast为FP32、写回前Cast_RINT，UB不保存FP32 State影子；out保持整tile完成后批量搬出。

V-first所有X的State有效UB读取量为FP32 128KiB/token、BF16 64KiB/token，写入仍64/32KiB。增加UB读取和BF16转换，以缩短State跨prediction归约的寄存器生命周期，不减少逐行归约。GM流量、UB峰值、bank所有权、预取和必要同步保持原规则。`State*(alpha*k)`相对原`(State*alpha)*k`存在FP32乘法重排舍入变化，按原CPU标杆、值域及混合容差验收。

V-first采用[分路径交错调度记录](recurrent-kda-dual-issue-optimizations-20261030.md)的vuf43：BF16的第一遍将两行Load/Cast、点积交错，再处理后两行；FP32的第二遍以滚动双行更新，在生成低片段out乘积后提前Store并释放低片段State，随后完成高片段点积和写回。两种调度均保留每四个归约结果的原收集方式、512B residual UB传递及同步；不改变FP32算术和归约顺序。BF16第二遍、FP32第一遍保持v2r34两遍流程，四种layout/dtype分别依据实测选择。

K-first两种State dtype采用[四路径独立优化记录](recurrent-kda-four-path-vf-optimizations-20261030.md)的ks54。每token在寄存器计算完整`alpha*k`，两次VST写入既有`[eventOffset+256,eventOffset+768)`的512B尾部预留区，经专用VST→VLD屏障后，第一遍沿K从UB广播系数，替代128次寄存器Gather及128次lane自增。系数在第一遍结束前全部消费，第二遍不再读取此区；下一token在前一token更新和必要屏障之后复用。事件字占前36B，系数区与事件、DMA输入/out和State bank均不重叠；State预取不使用此区。此区在V-first仍按前文保存delta。该扩展不增加UB分配峰值、State影子、workspace或GM流量；K-first第二遍计算、State Cast位置、累计顺序及整tile out搬出规则保持，新增的两次VST和屏障纳入实测。

二轮检视整改保留上述算术与同步策略。A5 纯数据 context 和集中布局常量在 `arch35/recurrent_kda_struct.h`；VF 启动/计算在 `recurrent_kda_vec.h`；任务、预取与写回函数在 `recurrent_kda_pipeline.h`，其文件头给出 Stage、布局和同步表。公开 host 对活跃 indices/accepted 做范围校验并保存 executor 自有快照，空请求和 speculative padding 跳过；此路径禁用旧 executor 缓存。无 indices 不增加 slot/accepted 快照步骤，packed 调用仍执行前文的 CU 校验和快照。State 的逻辑 view 复制由同一 helper 覆盖非原位初始化与原位独立输出；冻结的用户方案保持原样。验证结果归档于[二轮整改记录](recurrent-kda-review-r2-fixes-20261030.md)和[最终双平台检视验收](recurrent-kda-pr-review-acceptance-20261031.md)。
