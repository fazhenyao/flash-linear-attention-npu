# RecurrentKda 验证记录索引

本文件承接设计方案中的验证链接。最终 PR #997 的复核依据、整改、覆盖和限制见[双平台检视验收](recurrent-kda-pr-review-acceptance-20261031.md)。

- A5：188项 CPU 双输出精度、11项各50轮确定性、11项插桩 memcheck。
- A2：234项 CPU 双输出精度、17项各50轮确定性、17项插桩 memcheck。
- 两平台各74项公开契约、59项正常/49项负向三入口回归、20项元数据就绪专项通过。
- issue680 全36形状、四路径的最新 main 对比见[性能记录](recurrent-kda-upstream-final-compare-20261031.md)。

原始失败和中间修复历史见[A2记录](recurrent-kda-a2-test-20261031.md)、[host 报错记录](recurrent-kda-host-error-reporting-20261031.md)和[X=1两遍读取记录](recurrent-kda-x1-two-pass-20261031.md)。本地验收不替代正式 NPU CI；A3 和内部故障注入未覆盖。
