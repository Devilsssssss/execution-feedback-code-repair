# 实验索引与当前状态

这份索引给面试复核使用：先看项目简介，再按“问题、对照、结果、限制”的顺序打开原始协议和摘要。原始模型、逐例轨迹和训练数据体积较大，保留在实验机；Git 只保存可复核的协议、哈希和聚合摘要。

## 已完成的主线

| 阶段 | 要回答的问题 | 状态 | 入口 |
|---|---|---|---|
| 数据与执行审计 | 错误是否可追踪、公开/隐藏测试是否隔离 | 已完成 | `artifacts/train_sft_mixed_controlled_quality.md`、`artifacts/controlled_experiment_preflight.summary.json` |
| AST-only 对照 | Mixed 的提升是否只是错误类型分布变化 | 已完成 | `artifacts/ast_control_supplemented_v3.report.json`、`artifacts/ast_logic_supplement_v1.docker_audit.summary.json` |
| 多 seed 复验 | 结果是否依赖单个训练 seed | 已完成 | `artifacts/controlled_experiment_results_20261018.md` |
| HumanEvalFix 反馈消融 | 结构化反馈是否稳定优于无反馈 | 已完成但为不确定/阴性 | `artifacts/humanevalfix_atomized_full_comparison.json`、`artifacts/humanevalfix_feedback_shuffle_multiseed_summary.json` |

## 当前唯一开放实验

`feedback-contract-sft-v3` 已完成数据构造和 1241/1241 内容审计，服务器 2026-10-20 已完成 3 epoch 训练。审计后 143 例主集的 `none` 已完成（49/143 repair@1）；`structured` 评测仍在进行，36 例追溯确认集及 5 个置乱 seed 尚待启动。早期误用 157 例全集的结果已排除。最终解释和对外结论必须等正式评测、确认集与置乱审计完成后更新；协议和运行记录见[反馈契约实验](feedback_contract_experiment.md)。

## 面试时的结论边界

1. 可以明确声称：完成了可审计的数据生成、隔离执行、等规模 AST 对照、多 seed 复验和任务级配对分析。
2. 不应声称：执行反馈已经带来稳定收益，或 36 例确认集是盲测集。
3. 最有价值的后续实验是完成 v3 单变量对照；若仍为阴性，下一步应转向 on-policy 反馈依赖数据或更大模型，而不是直接把结果包装成 RL 增益。
