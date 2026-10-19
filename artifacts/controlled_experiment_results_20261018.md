# 受控 AST-only / Mixed 复验结果

更新日期：2026-10-18

## 状态

精确 AST-only 对照、Mixed seed 43/44、多条件 HumanEvalFix 评测和 36 例追溯冻结确认集均已在远端 RTX 3090 上完成。AST-only 与 Mixed 训练集都固定为 1241 条，AST-only 的归一化配额为 `boundary=122 / logic=564 / runtime=555`。训练使用同一基础模型、4-bit QLoRA、3 epochs、`max_length=1536`；正式评测使用 143 个 HumanEvalFix 审计案例、单轮 greedy 解码、最多 256 个新 token 和 Bubblewrap 隔离执行。

## 正式 HumanEvalFix 结果

| 训练 run | seed | 无反馈 | 结构化反馈 |
|---|---:|---:|---:|
| 精确 AST-only | 42 | 57/143（39.9%） | 55/143（38.5%） |
| Mixed | 43 | 61/143（42.7%） | 64/143（44.8%） |
| Mixed | 44 | 64/143（44.8%） | 65/143（45.5%） |

每个条件的 `repair@1` 与 `repair@k` 相同，因为本轮 `max_repairs=1`。所有运行均无 runner error。

## 配对分析

在同一批 143 个 bug 上进行 exact McNemar 检验：

| 条件 | 对比 | left-only / right-only | `p` |
|---|---|---:|---:|
| 无反馈 | AST42 vs Mixed43 | 11 / 15 | 0.5572 |
| 无反馈 | AST42 vs Mixed44 | 9 / 16 | 0.2295 |
| 无反馈 | Mixed43 vs Mixed44 | 10 / 13 | 0.6776 |
| 结构化反馈 | AST42 vs Mixed43 | 16 / 25 | 0.2110 |
| 结构化反馈 | AST42 vs Mixed44 | 15 / 25 | 0.1539 |
| 结构化反馈 | Mixed43 vs Mixed44 | 12 / 13 | 1.0000 |

因此，精确 AST-only 与新 Mixed run 的点估计方向在部分条件下偏向 Mixed，但逐例差异都没有达到统计显著。Mixed seed 43/44 之间也没有显示出可检测的差异，支持“新的两个 seed 在本协议下处于相近水平”，但不提供大样本训练方差估计。

旧 Mixed seed 42 的 72/143（无反馈）和 71/143（结构化反馈）是历史单 seed 结果。新的 seed 43/44 分别为 61/64 和 64/65，因此不能把 50.3%/49.7% 继续表述为已被多种子复现的主结论。旧 seed42 与新 seed 的差异只作为诊断性比较保留，不作为新的因果检验。

## 追溯冻结确认集

36 例确认集是在 143 例开发集上按 `bug_id` 哈希冻结的，来源数据此前已经参与开发分析，所以不是历史盲测；冻结之后禁止用于数据选择、checkpoint 选择或提示调参。

| 训练 run | seed | 无反馈 | 结构化反馈 |
|---|---:|---:|---:|
| 精确 AST-only | 42 | 12/36（33.3%） | 11/36（30.6%） |
| Mixed | 43 | 12/36（33.3%） | 15/36（41.7%） |
| Mixed | 44 | 13/36（36.1%） | 13/36（36.1%） |

这些结果只能作为冻结后的确认性结果，不能替代真正从未查看过的盲测集。

## 结论边界

1. 精确 AST-only 数据缺口已经补齐并通过公开/隐藏测试和 Docker 复审；同规模、同归一化配额的模型对照已经真正训练和评测完成。
2. Mixed seed 43/44 的复验没有复现旧 seed42 的 72/71 点估计；因此旧的 50.3%/49.7% 应标为历史单 seed 结果。
3. 在当前 143 例配对检验下，没有证据证明精确 AST-only 与 Mixed 的差异稳定显著。可以说“数据分布差异仍是待解释因素”，不能说“已证明模型真实错误带来稳定提升”。
4. 反馈消融的旧结论仍成立：新 run 的无反馈和结构化反馈差异没有形成稳定方向，不能宣称 SFT 已学会可靠利用执行反馈。
5. 36 例确认集已经冻结并完成一次评测，但由于其来源集曾用于开发，它是追溯冻结确认集，不是历史盲测。

## 复核产物

- `artifacts/controlled_experiment_none_paired_20261018.json`
- `artifacts/controlled_experiment_structured_paired_20261018.json`
- `artifacts/controlled_experiment_mixed_seed42_43_44_none_paired_20261018.json`
- `artifacts/controlled_experiment_mixed_seed42_43_44_structured_paired_20261018.json`
- `artifacts/controlled_experiment_20261018_summary_comparison.json`
- `artifacts/humanevalfix_confirmation_v1.summary.json`
- 六个正式评测 summary：`artifacts/humanevalfix_{ast_matched_v3_seed42,mixed_controlled_seed43,mixed_controlled_seed44}_{none,structured}.summary.json`
- 六个确认集逐例结果及 summary：`artifacts/humanevalfix_confirmation_v1_{ast_seed42,mixed_seed43,mixed_seed44}_{none,structured}.*`
