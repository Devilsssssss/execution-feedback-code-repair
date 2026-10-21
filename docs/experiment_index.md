# 实验索引与当前状态

这份索引给面试复核使用：先看项目简介，再按“问题、对照、结果、限制”的顺序打开原始协议和摘要。原始模型、逐例轨迹和训练数据体积较大，保留在实验机；Git 只保存可复核的协议、哈希和聚合摘要。

## 已完成的主线

| 阶段 | 要回答的问题 | 状态 | 入口 |
|---|---|---|---|
| 数据与执行审计 | 错误是否可追踪、公开/隐藏测试是否隔离 | 已完成 | `artifacts/train_sft_mixed_controlled_quality.md`、`artifacts/controlled_experiment_preflight.summary.json` |
| AST-only 对照 | Mixed 的提升是否只是错误类型分布变化 | 已完成 | `artifacts/ast_control_supplemented_v3.report.json`、`artifacts/ast_logic_supplement_v1.docker_audit.summary.json` |
| 多 seed 复验 | 结果是否依赖单个训练 seed | 已完成 | `artifacts/controlled_experiment_results_20261018.md` |
| HumanEvalFix 反馈消融 | 结构化反馈是否稳定优于无反馈 | 已完成但为不确定/阴性 | `artifacts/humanevalfix_atomized_full_comparison.json`、`artifacts/humanevalfix_feedback_shuffle_multiseed_summary.json` |

## 当前状态

`feedback-contract-sft-v3` 已完成数据构造、1241/1241 内容审计、服务器训练及全部预注册评测。v3 主集 none→structured repair@1 为 49/143→58/143，但原 Mixed checkpoint 为 72/143→71/143；因此训练目标改变后 v3 的绝对能力显著较低，不能声称整体收益。确认集方向相同但不是盲测，五组置乱仅一组名义显著，整体不支持稳定利用正确反馈。最准确结论是“v3 内部出现有限的反馈条件交互信号，但训练目标变更的整体效果为负面”。误用 157 例全集的早期结果已排除。详见[反馈契约实验](feedback_contract_experiment.md)。

## 面试时的结论边界

1. 可以明确声称：完成了可审计的数据生成、隔离执行、等规模 AST 对照、多 seed 复验和任务级配对分析。
2. 不应声称：执行反馈已经带来稳定收益，或 36 例确认集是盲测集。
3. Code Repair 的预注册实验已全部完成；若继续，优先做独立新确认集、多训练 seed 或 on-policy 反馈依赖数据，不要直接把当前结果包装成 RL 增益。
