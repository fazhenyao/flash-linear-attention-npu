# PR #997 检视整改与验证

本轮以提交 `8e818942035fdfb263ea78cd4f0c70dfd310964f` 的六项检视意见为依据，
修复公开入口和交付问题。冻结方案 `recurrent-kda-a5-regbase-redesign-20261026.md`
保持不变，SHA256 为 `78f03d56724f4ff6ba36609f1f53dbc05148825b94ee7d253a7867c8c331d5b4`。

## 整改结果

| 检视项 | 实现与验证 |
| --- | --- |
| 非原位 State 未写槽 | ACLNN 在非原位且要求输出 State 时，将初态的逻辑 view 复制到独立输出，再提交递推。空请求、稀疏 pool 的未写槽保持初值；复制处理外层间隔、非零 offset、单例 view 的较大底层 storage 及 plain format 标签。stable、ctypes、直接 ACLNN 均在真实 NPU 上验证。 |
| 已发布调用兼容 | Python 两入口及 ACLNN 将 rank3/无 cu_seqlens 规整为单 batch，rank4/有 cu_seqlens 展平为 packed；恢复原 out shape。layout 仍被忽略，kernel 仍只根据 cu_seqlens 区分模式。旧形状别名发出弃用提示，保留至 v26.12，最早 v27.1 移除。直接 ACLNN 的旧 rank4 packed 别名要求 out 连续；Python 输出自动分配为连续。与 v26.9.1 的公开函数签名、默认值及返回约定核对一致。 |
| 参数错误闭环 | 公开 shape、stride、维度、可选参数约束采用参数级上报并附实际描述；A5 cu_seqlens 的值错误在 ACLNN 返回 161002，包含 index、previous、actual、capacity。Stable-ABI 异常保留 ACL 最近错误上下文，外层失败不覆盖叶子原因。 |
| 负向测试过宽 | 原契约测试断言具体状态和消息片段；新增三入口负向测试，并在每次拒绝后执行合法输入及 CPU 对比。异常种类本身不作为通过依据。 |
| 公开环境信息 | 可编辑报告中的具体主机、账号和运行根目录改为占位符，保留软件版本、shape、样本数与 hash。 |
| 本机资产链接 | 六个精简 PNG/SVG 纳入相对 assets 目录；未公开的大型原始证据改为纯文本名称及 SHA256，不作为可访问的公共链接。冻结方案中的三个历史链接按用户要求保留。 |

## 本轮真实验证

A5_1，Ascend950PR，物理卡4映射逻辑0；按用户授权共享设备。
CANN 9.2.0-beta.2、Python 3.11.17、torch 2.9.0+cpu、torch_npu 2.9.0.post6、ATK 26.9.8。
使用独立 venv 完整构建 wheel，标准 run 安装；libcust_opapi MD5、22个构建源文件、
三个已安装 Python 文件、Stable-ABI 动态库及78个已安装 so/o 的身份均已核对。

| 项目 | 最终结果 |
| --- | --- |
| 原有精度 | 132项 SUCCESS，CPU 精度字段全部为 true |
| 对齐精度 | 54项 SUCCESS，CPU 精度字段全部为 true |
| 新增正常场景 | 48项通过：三入口 × 两 State 布局 × 两 dtype × 空请求/稀疏 pool/两种旧形状别名 |
| 新增负向检查 | 18项通过：三入口 × head 比例、Q dtype、State stride、cu 起点、顺序、容量；逐次合法调用恢复检查通过 |
| 公开契约 | 74项通过，mutation/grad 通过 |
| 设备对象 | 全部72个 `.o` 与已验收 ks54 逐字节相同 |
| 静态复核 | 最新192项 issue、21条规则覆盖检查通过；公开函数 AST 和 diff 空白检查通过 |

沿用原 CPU 标杆、值域、混合容差，不修改 CPU 计算规则。
正式报告逐行核验，不以 ATK 进程退出码代替精度结果。
前期构建、环境及调试失败日志与最终成功日志一起保留。

静态扫描原始剩余项为36个 DFX-MESSAGE、10个 DOC-SENSITIVE、3个 DOC-BROKEN-LINK，
扫描器仍返回非零，不能表述为零发现。36个 DFX 位置为内部资源/运行时失败、
已结构化叶子校验的传播及内部上下文检查；未将内部故障改报为公开参数错误，
也未宣称已做内部资源失败注入。10个敏感项均为采样时间选项名称中的子串误匹配；
3个断链均在受保护方案中。本轮没有更改扫描规则或增加忽略标记。
历史报告的114项是当轮机械命中数，其中19个位于 PR 新增行，不能统称为 PR 前已存在的114个问题。

## 性能与验证边界

按用户“修改不影响性能可以不重新对比”的要求，本轮不重复 msprof。
计算内核、X 选择、State bank 调度、同步及 tile 完成后批量搬出 out 均未改变，
72个设备对象完全一致。原 issue680 四路径性能报告针对 inplace 调用；本轮该路径不新增 State 复制。
[既有144组性能报告](recurrent-kda-upstream-ks54-compare-20261030.md)继续作为历史证据，
不是整改后重新采集的结果。

非原位 ACLNN/ctypes 为正确保留未写槽新增初态复制；旧 rank4 别名可能物化输入，
cu_seqlens 校验增加 host 检查。这些调用的端到端开销不能由设备对象一致推断为不变，
本轮未采集其性能。A2/A3 上板、全量 Example/ST 和当前 head 正式 NPU CI 未执行；
此前 ks54 的确定性和 Pipeline sanitizer 记录是历史结果，不冒充本轮重跑。

## 复现与归档

构建、安装及测试环境先按[ATK README](../../../../../../tests/atk/recurrent_kda/README.md)配置。
`<run-root>` 为执行者自己的私有工作目录；先加载 CANN 和已安装 vendor 的 `bin/set_env.bash`，
避免 ATK 启动时尚未发现 OPP。

```bash
FLA_NPU_SOC=ascend950 FLA_NPU_OPS=recurrent_kda FLA_NPU_BUILD_ARGS=-j12 \
  python -m pip wheel --no-build-isolation --no-deps . -w dist
python -m pip install --force-reinstall --no-cache-dir --no-deps dist/*.whl
bash build_out/fla_npu_linux-x86_64.run --install
python scripts/verify_libcust_opapi_md5.py \
  --run-package build_out/fla_npu_linux-x86_64.run --python
python tests/atk/recurrent_kda/scripts/check_review_regressions.py --device 0
python tests/atk/recurrent_kda/scripts/check_public_contract.py --device 0
```

132项与54项正式精度命令、原始任务行及报告包含于本机未公开的
`output/rkda-review-fixes-20261030/evidence.zip`，共94文件，CRC 及逐文件 SHA256 均通过。
证据包 SHA256：`9bfa902ab698ac968139c03df9703ac9df348f3594c3d53306767101ff7a7caf`。
该名称仅标识私有证据，不作为公共可点击工件；交付源码、测试和精简图位于仓库内。
