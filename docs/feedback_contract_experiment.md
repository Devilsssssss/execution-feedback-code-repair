# 反馈契约 SFT 实验

状态：**协议和 v3 训练数据已冻结并通过本地审计；远端训练与 HumanEvalFix 评测尚未完成。**

## 动机

此前结论指出：混合数据模型在 HumanEvalFix 上结构化反馈（71/143）相对无反馈（72/143）无增益，5 组错配反馈虽整体低于正确反馈（均值 64.4 vs 72），但逐次检验不显著。最可能的根因是 SFT 训练目标从不依赖反馈——提示词中的执行反馈与目标答案无因果关系，模型学不到使用反馈的策略。

本实验把训练目标从“直接给参考答案”改为“显式依赖反馈”。每一行 completion 在参考代码前增加一行确定性 `# FEEDBACK_CONTRACT` 注释；注释只包含该 bug 真实公开执行的 `passed`、`total` 和首个失败 `error_type`。prompt、`bug_id`、任务划分和参考代码完全不变，模型输出仍由评测器从首个顶层 Python 声明开始解析。`build_feedback_contract_sft.py` 生成数据，`audit_feedback_contract_sft.py` 检查一一对应、prompt 不变、参考代码不变和契约字段一致。

v1/v2 曾尝试更长的测试文本和 expected/actual 摘要。tokenizer 审计显示少数样本超过训练 `max_length=1536`，这两个版本已作废；v3 收缩为最小失败证据，并通过 1241 条内容审计。

## 对照与评测

| 组 | 训练目标 | seed |
|---|---|---:|
| Mixed baseline | 直接输出参考代码 | 42（已有 checkpoint）|
| feedback-contract-v3 | 反馈契约注释 + 参考代码 | 42 |

两组使用相同 QLoRA 超参数、HumanEvalFix 143 例主集、36 例追溯冻结确认集、`none`/`structured` 条件和 Bubblewrap 执行器。另对实验组运行 5 个无自配对反馈置乱 seed，单轮评测检查输出是否依赖反馈对应关系。

## 预注册判读

- 阳性候选：实验组 `structured - none` 至少增加 5 个百分点且确认集同方向；或至少 3/5 个置乱 seed 相对正确反馈显著下降；
- 阴性：没有上述信号。结论应是当前 SFT 目标不足以教会反馈使用，下一步需要更强的 on-policy 数据或更大模型，而不是直接启动 GRPO；
- 不把一次 seed 写成稳定收益，也不把本实验解释成 RL 证据。

## 本地复现

```powershell
$env:PYTHONPATH = "src"
python scripts/build_feedback_contract_sft.py `
  --input-sft data/processed/mbpp/train_sft_mixed_controlled.jsonl `
  --bugs data/processed/mbpp/train_bugs_mixed_controlled.jsonl `
  --tasks data/processed/mbpp/train_audited_tasks_v2_dedup.jsonl `
  --output data/processed/mbpp/train_sft_feedback_contract_v3.jsonl `
  --summary artifacts/feedback_contract_sft_v3.summary.json

python scripts/audit_feedback_contract_sft.py `
  --original data/processed/mbpp/train_sft_mixed_controlled.jsonl `
  --contract data/processed/mbpp/train_sft_feedback_contract_v3.jsonl `
  --output artifacts/feedback_contract_sft_v3.audit.json
```

训练和评测脚本在 `scripts/run_feedback_contract_evals.sh` 与 `scripts/run_feedback_contract_shuffles.sh`；需要服务器上的锁定 QLoRA 环境和 Bubblewrap。训练数据、adapter 和逐例轨迹不纳入 Git，只提交哈希、协议和汇总结果。
 
