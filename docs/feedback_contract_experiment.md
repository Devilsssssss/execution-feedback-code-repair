# 反馈契约 SFT 实验

状态：**v3 训练与预注册评测全部完成；审计后的主集、确认集及 5 个置乱 seed 均已完成，未发现 runner error。**

## 动机

此前结论指出：混合数据模型在 HumanEvalFix 上结构化反馈（71/143）相对无反馈（72/143）无增益，5 组错配反馈虽整体低于正确反馈（均值 64.4 vs 72），但逐次检验不显著。最可能的根因是 SFT 训练目标从不依赖反馈——提示词中的执行反馈与目标答案无因果关系，模型学不到使用反馈的策略。

本实验把训练目标从“直接给参考答案”改为“显式依赖反馈”。每一行 completion 在参考代码前增加一行确定性 `# FEEDBACK_CONTRACT` 注释；注释只包含该 bug 真实公开执行的 `passed`、`total` 和首个失败 `error_type`。prompt、`bug_id`、任务划分和参考代码完全不变，模型输出仍由评测器从首个顶层 Python 声明开始解析。`build_feedback_contract_sft.py` 生成数据，`audit_feedback_contract_sft.py` 检查一一对应、prompt 不变、参考代码不变和契约字段一致。

v1/v2 曾尝试更长的测试文本和 expected/actual 摘要。tokenizer 审计显示少数样本超过训练 `max_length=1536`，这两个版本已作废；v3 收缩为最小失败证据，并通过 1241 条内容审计。

## 对照与评测

| 组 | 训练目标 | seed |
|---|---|---:|
| Mixed baseline | 直接输出参考代码 | 42（已有 checkpoint）|
| feedback-contract-v3 | 反馈契约注释 + 参考代码 | 42 |

两组使用相同 QLoRA 超参数、HumanEvalFix 审计后的 143 例主集（`audited_tasks.jsonl`/`audited_bugs.jsonl`）、36 例追溯冻结确认集、`none`/`structured` 条件、greedy decoding、单轮最多 256 个新 token 和 Bubblewrap 执行器。另对实验组运行 5 个无自配对反馈置乱 seed，单轮评测检查输出是否依赖反馈对应关系。

## 服务器执行记录与结论

2026-10-20：v3 在服务器完成 3 epoch QLoRA 训练，共 234 optimizer steps，训练 loss 0.291，runtime 1082 秒。adapter、训练日志和原始逐例轨迹保留在实验机，不进入 Git。主集评测固定为审计后的 143 例，确认集 36 例，所有条件均为 greedy、max-new-tokens 256；9 份主/确认/置乱结果均有 0 runner error。

| 模型/条件 | repair@1 | repair@k |
|---|---:|---:|
| Mixed checkpoint，无反馈（none） | 72/143 (50.35%) | 72/143 (50.35%) |
| Mixed checkpoint，结构化反馈 | 71/143 (49.65%) | 71/143 (49.65%) |
| feedback-contract-v3，无反馈（none） | 49/143 (34.27%) | 51/143 (35.66%) |
| feedback-contract-v3，结构化反馈 | 58/143 (40.56%) | 61/143 (42.66%) |
| 确认集，无反馈 | 8/36 (22.22%) | 9/36 (25.00%) |
| 确认集，结构化反馈 | 11/36 (30.56%) | 12/36 (33.33%) |

在 v3 模型内部，主集 repair@1 差值 +6.29pp，任务级精确 McNemar 检验为 14 个 structured-only、5 个 none-only，p=0.0636；repair@k 差值 +6.99pp，15 对 5，p=0.0414。确认集同方向（+8.33pp），但配对检验只有 3 个 structured-only、0 个 none-only，p=0.25；它是开发期间形成的追溯确认集，不是独立盲测，也不应单独作为统计确认。

不过与预先列出的 Mixed checkpoint 做绝对水平比较时，v3 的 repair@1 低 23/143（−16.08pp，none 条件配对 p=0.000066），structured 条件低 13/143（−9.09pp，配对 p=0.0192）。原 Mixed checkpoint 的 structured−none 为 −0.70pp，v3 为 +6.29pp，描述性的差中差为 +6.99pp；这提示训练目标可能改变了模型对反馈条件的响应方向，但 v3 的绝对修复能力退化，且差中差尚未作为独立预注册统计检验，不能把它说成收益。

反馈置乱的 5 个 seed repair@1 为 58、49、52、59、57/143，均值 55/143（38.46%），正确反馈为 58/143（40.56%）。仅 seed 42 的逐 seed 配对检验名义 p<0.05（12 对 3，p=0.0352），其余 4 个均不显著；置乱有时与正确反馈持平或更高，且并非所有置乱结果都更差。未校正的单 seed 显著不能作为稳定依赖性的证据；五 seed 汇总也不支持“模型稳定利用了反馈与 bug 的正确对应关系”。

**判定：训练目标变更的整体效果为负面，内部反馈条件差值则是有希望但证据有限的探索信号。** 预注册的“v3 主集至少 +5pp 且确认集同方向”候选门槛满足；但 v3 绝对水平低于 Mixed checkpoint，主集 repair@1 配对 p=0.0636，确认集样本小且非盲测，置乱敏感性也不稳定。因此项目结论不能叫作正向模型收益。准确表述是：该训练目标让 v3 在结构化反馈条件相对自身无反馈条件呈现两集同向提升，但绝对修复能力下降，且没有证据证明这种提升来自稳定利用正确反馈对应关系。不要把它说成稳健增益或 RL 证据。

执行中发现早期命令曾误用 157 例全集，该次结果已作废且没有进入上述比较；正式脚本固定使用 143 例审计集并将新 token 上限固定为 256。主集/确认集任务级配对摘要、Mixed 对照配对摘要与 5 个置乱映射审计摘要保存在 `artifacts/feedback_contract_v3_*`；原始 generations 和逐例记录不进入 Git。训练行数/哈希、最终 adapter 哈希以及每份评测轨迹的哈希记录在[复现清单](../artifacts/feedback_contract_v3_run_manifest.json)。

## 预注册判读

- 阳性候选：实验组 `structured - none` 至少增加 5 个百分点且确认集同方向；或至少 3/5 个置乱 seed 相对正确反馈显著下降；
- 阴性：没有上述信号。结论应是当前 SFT 目标不足以教会反馈使用，下一步需要更强的 on-policy 数据或更大模型，而不是直接启动 GRPO；
- 不把一次 seed 写成稳定收益，也不把本实验解释成 RL 证据。

本次实际结果满足第一条候选门槛（主集 +6.29pp，确认集同方向），但不满足第二条置乱敏感性门槛（仅 1/5 个 seed 名义显著，且置乱均值仅低 3/143）。因此按协议记录为：**候选门槛阳性、整体收益判定阴性/不确定**。v3 的绝对水平低于 Mixed checkpoint，确认集非盲测，不能将该候选信号升级为稳定收益。

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
 
