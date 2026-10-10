# LoRA 训练前数据质量报告

结论：**warn**

## 数据规模

| 项目 | 数量 |
|---|---:|
| train_tasks | 341 |
| validation_tasks | 43 |
| test_tasks | 42 |
| train_bugs | 1241 |
| train_sft | 1241 |

## 检查结果

| 检查 | 状态 | 重点问题数量 |
|---|---|---:|
| task_integrity | warn | 8 |
| bug_integrity | pass | 0 |
| sft_integrity | pass | 0 |
| split_leakage | pass | 0 |

## Bug 分布

| 类型 | 数量 |
|---|---:|
| boundary_condition | 122 |
| logic_error | 267 |
| model_logic_error | 297 |
| model_runtime_error | 171 |
| model_timeout_error | 1 |
| runtime_error | 383 |

## SFT 长度

提示词近似 token 数按字符数除以 4 估算，最终训练仍应以 tokenizer 实测为准。

| 字段 | p50 | p95 | p99 | 最大值 |
|---|---:|---:|---:|---:|
| approx_prompt_tokens | 231 | 433 | 502 | 1197 |
| approx_completion_tokens | 34 | 98 | 148 | 226 |

Tokenizer 实测：`models/qwen2.5-coder-0.5b-instruct`。训练输入合并后的最大长度为 1482 tokens（纯文本拼接，不含对话模板）。

| 字段 | p50 | p95 | p99 | 最大值 |
|---|---:|---:|---:|---:|
| tokenizer_prompt_tokens | 268 | 517 | 631 | 1428 |
| tokenizer_completion_tokens | 45 | 151 | 198 | 210 |
| tokenizer_combined_tokens | 330 | 595 | 719 | 1482 |
| tokenizer_training_chat_tokens | 382 | 647 | 771 | 1534 |

## 建议

- 训练前保留 train_audited_tasks.jsonl 的参考答案审计结果；不要把 rejected 文件混入 SFT。
- 若 prompt/completion mismatch 或 public feedback 无效，先修复数据生成流程再训练。
- 正式实验固定任务级 train/validation/test 划分，并报告同题不同错误版本只能出现在一个 split。
- 当前 train SFT 样本只有运行时错误和少量逻辑/边界错误时，应做按 bug_type 的分层统计，避免只学会 NameError 修复。
