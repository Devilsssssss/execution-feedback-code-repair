# Execution-Feedback Code Repair

一个面向求职作品集的研究型最小项目：评估小型代码模型能否利用公开测试的执行反馈，修复 Python 单函数中的逻辑与运行时错误。

当前版本已经完成数据构建、QLoRA 训练和独立测试集评测：

- 按题目划分训练/评测数据，阻止同题泄漏；
- 用 AST 进行可追踪的受控错误注入；
- 严格分离可反馈的公开测试和只用于最终评分的隐藏测试；
- 统一比较无反馈、原始反馈、结构化反馈和 LoRA 微调四种条件；
- 通过 Docker 或 Linux Bubblewrap 隔离执行模型生成的代码；
- 输出逐样本轨迹和聚合指标，支持按错误类型分析。

最终结论不是“执行反馈一定有效”，而是：**加入经过执行审计的模型真实错误，显著改善了跨数据集修复；当前 SFT 尚未学会稳定利用反馈。** 项目同时保留正结果和负结果，避免只展示最高分。

## 目录

```text
src/repairbench/        核心数据、执行、提示词、模型和评测逻辑
scripts/                一键演示、发布核验、数据准备、审计与训练入口
data/demo/              不依赖下载的最小演示任务
requirements/           开发环境与原实验环境的固定依赖
artifacts/              Git 仅保留小型报告、统计结果与数据清单
tests/                  评测可信性与核心流程测试
```

## 最终实验结论

2026-10-10 的主结论来自 HumanEvalFix Python 外部评测。所有条件使用相同的 143 个审计案例、greedy decoding、单轮最多 256 个新 token；公开断言可形成反馈，隐藏测试不进入提示词。

| 训练数据 | 无反馈 | 结构化反馈 |
|---|---:|---:|
| 仅 AST 合成错误 | 42/143（29.4%） | 41/143（28.7%） |
| AST + 模型生成错误 | **72/143（50.3%）** | **71/143（49.7%）** |

等规模混合数据在两种提示条件下都增加 30 个成功案例：无反馈配对 McNemar `p=2.27e-7`，结构化反馈 `p=6.04e-7`。但混合模型的无反馈 72/143 与结构化反馈 71/143 没有显著差异（`p=1.0`）；最多三轮时也只是 77/143 与 75/143（`p=0.804`）。因此证据支持**训练错误分布改进**，不支持“模型已学会有效使用执行反馈”。

后续增加了 5 个反馈置乱对照（seed 42–46）：每次都将 143 条反馈做无自配对的一一置换。正确反馈为 71/143，5 次置乱分别为 64、69、59、65、65/143，均值 64.4/143（45.0%），范围 59–69。虽然 5 次都低于正确反馈，但逐次配对检验只有 1/5 在未校正条件下达到 `p<0.05`，结果对置乱方式较敏感；而且无反馈仍为 72/143。因此它提示反馈内容会影响输出，但仍不能证明反馈带来稳定净收益。

可直接复核的 Git 内产物：

- `artifacts/experiment_report.md`：完整结论、局限和下一步；
- `artifacts/humanevalfix_atomized_full_comparison.json`：六组单轮配对统计；
- `artifacts/humanevalfix_atomized_k3_feedback_ablation.json`：三轮反馈消融；
- `artifacts/humanevalfix_feedback_shuffle_multiseed_comparison.json`：5 个置乱 seed 的逐例配对统计；
- `artifacts/humanevalfix_feedback_shuffle_multiseed_summary.json`：置乱映射完整性、汇总指标与结论边界；
- `artifacts/on_policy_later_success_summary.json`：后续成功轨迹的数量与类型汇总；
- `artifacts/train_sft_mixed_controlled_manifest.json`：1241 条混合数据的来源、数量与哈希；
- `artifacts/train_sft_mixed_controlled_quality.md`：训练前质量审计。

### 第一阶段内部基线（MBPP）

2026-09-30 在一张 RTX 3090 上完成了 Qwen2.5-Coder-0.5B-Instruct 的第一轮 QLoRA 实验。测试集包含 42 个未参与训练的 MBPP 任务及其 152 个错误版本；四组均使用同一批错误代码、一次修复机会、256 个最大新 token，并只以修复过程中不可见的隐藏测试评分。

| 实验组 | 反馈 | repair@1 | 边界错误 | 逻辑错误 | 运行时错误 |
|---|---|---:|---:|---:|---:|
| 原始模型 | 无反馈 | 58/152（38.16%） | 4/11 | 11/50 | 43/91 |
| 原始模型 | 原始反馈 | 17/152（11.18%） | 2/11 | 3/50 | 12/91 |
| 原始模型 | 结构化反馈 | 32/152（21.05%） | 3/11 | 13/50 | 16/91 |
| QLoRA 模型 | 结构化反馈 | **106/152（69.74%）** | **6/11** | **25/50** | **75/91** |

QLoRA 相比相同结构化反馈下的原始模型提高 48.68 个百分点，相比最强的原始模型条件提高 31.58 个百分点。原始反馈反而明显降低 0.5B 模型的表现；结构化反馈比原始反馈好，但对未微调模型仍不如无反馈。

这组结果只能说明：在训练和测试共享受控 AST 变异机制的 MBPP 内部分布上，QLoRA 提高了修复率。由于当时没有对微调模型做无反馈消融，不能据此断言提升来自“利用反馈”；后续 HumanEvalFix 实验表明，主要收益来自训练错误分布。它也不能直接外推到大型代码仓库或未知错误分布。

以下大体积产物保留在本地并被 Git 忽略：

- `artifacts/test_comparison.json`：四组统一汇总；
- `artifacts/test_*.jsonl`：逐样本提示词、候选代码和公开/隐藏测试轨迹；
- `outputs/qwen25-coder-05b-repair-qlora/`：最终 LoRA 适配器和三个训练检查点；
- `logs/qwen25-coder-05b-repair-qlora.log`：完整训练日志；
- `logs/test_qlora_structured.log`：微调模型正式评测日志。

## 快速开始

安装项目后，一条命令即可在 CPU 上跑完“生成错误、验证、构建 SFT 样本、规则基线评测”。它只执行仓库中人工检查过的四个演示函数，不下载模型：

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements/dev.txt
python scripts/run_demo.py
```

输出位于 `artifacts/demo_run/`。完整开发检查为：

```powershell
pytest
ruff check .
python scripts/verify_release.py
```

生成演示错误样本。下面的 `local` 只允许用于仓库内人工检查过的演示代码：

```powershell
repairbench generate-bugs `
  --tasks data/demo/tasks.jsonl `
  --output data/processed/demo_bugs.jsonl `
  --executor local `
  --allow-unsafe-local
```

验证参考答案、错误样本和公开/隐藏测试：

```powershell
repairbench validate `
  --tasks data/demo/tasks.jsonl `
  --bugs data/processed/demo_bugs.jsonl `
  --executor local `
  --allow-unsafe-local
```

构建 SFT 训练样本：

```powershell
repairbench build-sft `
  --tasks data/demo/tasks.jsonl `
  --bugs data/processed/demo_bugs.jsonl `
  --output data/processed/demo_sft.jsonl `
  --executor local `
  --allow-unsafe-local
```

LoRA 训练前先做数据质量审计：

```powershell
python scripts/audit_sft_data.py `
  --train-tasks data/processed/mbpp/train_audited_tasks.jsonl `
  --validation-tasks data/processed/mbpp/validation_audited_tasks.jsonl `
  --test-tasks data/processed/mbpp/test_audited_tasks.jsonl `
  --train-bugs data/processed/mbpp/train_bugs.jsonl `
  --train-sft data/processed/mbpp/train_sft.jsonl `
  --tokenizer models/qwen2.5-coder-0.5b-instruct `
  --output artifacts/train_sft_quality.json
```

脚本会检查字段完整性、参考答案和错误代码的 AST 可解析性、公开反馈是否有效、SFT prompt 是否能由原始任务重建、completion 是否等于经过审计的参考答案，以及 train/validation/test 是否存在题目或 prompt-答案组合泄漏。它同时给出错误类型分布和长度分位数；`decision=fail` 时不要开始训练，`decision=warn` 时先处理报告中的重复样本或类别偏斜。

当前 MBPP 训练集审计结果保存在 `artifacts/train_sft_quality.md`：676 条 SFT 样本的字段和 prompt/答案一致，代码均可解析，任务级切分无泄漏；但有 2 组完全重复的错误代码，且错误类型为 runtime 425 条、logic 212 条、boundary 39 条，分布偏斜明显。另有 3 个参考实现触发 `invalid escape sequence` 警告。训练前应决定是否去重，并在结果中报告这种类别分布，避免把 NameError 修复能力误当成一般代码修复能力。评测集另有 1 个任务包含重复公开测试，不影响训练数据，但应在正式 benchmark 报告中注明或清理。

如果要生成增加边界变异的 v2 训练集，可以使用：

```powershell
repairbench generate-bugs `
  --tasks data/processed/mbpp/train_audited_tasks.jsonl `
  --output data/processed/mbpp/train_bugs_v2.jsonl `
  --per-task 4 `
  --executor local `
  --allow-unsafe-local

repairbench build-sft `
  --tasks data/processed/mbpp/train_audited_tasks.jsonl `
  --bugs data/processed/mbpp/train_bugs_v2.jsonl `
  --output data/processed/mbpp/train_sft_v2.jsonl `
  --executor local `
  --allow-unsafe-local
```

v2 新增了“比较常量偏移”的受控边界变异；训练集边界样本从 39 增加到 122。再执行 `scripts/deduplicate_training_data.py` 删除语义重复任务后，推荐用于第一轮 LoRA 的文件是 `data/processed/mbpp/train_sft_v2_dedup.jsonl`，共 1241 条样本。对应的审计报告是 `artifacts/train_sft_v2_dedup_quality.md`。

```powershell
python scripts/deduplicate_training_data.py `
  --tasks data/processed/mbpp/train_audited_tasks.jsonl `
  --bugs data/processed/mbpp/train_bugs_v2.jsonl `
  --sft data/processed/mbpp/train_sft_v2.jsonl `
  --tasks-output data/processed/mbpp/train_audited_tasks_v2_dedup.jsonl `
  --bugs-output data/processed/mbpp/train_bugs_v2_dedup.jsonl `
  --sft-output data/processed/mbpp/train_sft_v2_dedup.jsonl `
  --report artifacts/train_sft_v2_dedup.json
```

在真实模型之前，先运行不修改代码的控制组，确认评测管线和隐藏测试隔离正常：

```powershell
repairbench evaluate-noop `
  --tasks data/demo/tasks.jsonl `
  --bugs data/processed/demo_bugs.jsonl `
  --output artifacts/demo_noop.jsonl `
  --condition structured `
  --max-repairs 1 `
  --executor local `
  --allow-unsafe-local
```

这个控制组不是算法基线，只表示“模型完全不修复”的下限。演示集和 MBPP 测试集的 no-op 控制均得到 `repair@1=0`，符合预期。

另外提供一个只处理 `NameError` 的可解释规则基线：

```powershell
repairbench evaluate-heuristic `
  --tasks data/processed/mbpp/test_audited_tasks.jsonl `
  --bugs data/processed/mbpp/test_bugs.jsonl `
  --output artifacts/mbpp_heuristic.jsonl `
  --condition structured `
  --max-repairs 1 `
  --executor local `
  --allow-unsafe-local
```

当前测试集结果为 `repair@1=49.4%`：运行时错误 91.1%、逻辑错误 0%、边界错误 0%。它不是最终模型，而是用来证明执行反馈有效、并给后续代码模型提供一个可解释对照。

多组实验完成后可以统一汇总：

```powershell
python scripts/compare_summaries.py `
  noop=artifacts/mbpp_noop.summary.json `
  heuristic=artifacts/mbpp_heuristic.summary.json `
  --output artifacts/mbpp_comparison.json
```

使用本地 Hugging Face 模型跑评测，需要安装模型依赖：

```powershell
# 先按你的 CUDA/CPU 环境安装 PyTorch；项目不会覆盖已有的 PyTorch
python -m pip install ".[model]"
repairbench evaluate `
  --tasks data/demo/tasks.jsonl `
  --bugs data/processed/demo_bugs.jsonl `
  --model-id Qwen/Qwen2.5-Coder-0.5B-Instruct `
  --condition structured `
  --executor docker `
  --output artifacts/demo_structured.jsonl
```

如果模型仓库无法直接从 Hugging Face 下载，可以把权重放到本地目录后使用同一条命令。项目提供了支持断点临时文件、并行 HTTP Range 和 SHA-256 校验的下载脚本；例如 ModelScope 的 Qwen2.5-Coder-0.5B 权重：

```powershell
python scripts/download_model_ranges.py `
  --url https://modelscope.cn/models/Qwen/Qwen2.5-Coder-0.5B-Instruct/resolve/master/model.safetensors `
  --output models/qwen2.5-coder-0.5b-instruct/model.safetensors `
  --sha256 f9523886352217ded3aeeef552b381af79d568c6d49a4b9e423288cea56b0a44

repairbench evaluate `
  --tasks data/demo/tasks.jsonl `
  --bugs data/processed/demo_bugs.jsonl `
  --model-id models/qwen2.5-coder-0.5b-instruct `
  --condition structured `
  --max-repairs 1 `
  --max-new-tokens 160 `
  --limit 1 `
  --device-map cpu `
  --executor local `
  --allow-unsafe-local `
  --output artifacts/demo_qwen_structured.jsonl
```

本机已验证该 0.5B 权重可以加载并完成生成；CPU 推理较慢，建议先用单个案例或较小的 `--max-new-tokens` 做链路验证。2026-09-29 的一次 `demo_clamp` 单案例结果为 `repair@1=false`：候选代码能被解析并执行，最终失败类型为 `AssertionError`，因此没有把模型失败伪装成成功结果。

`--limit` 只用于 smoke test；正式实验应去掉它，并固定完整的案例列表、生成预算和隐藏测试。

Docker 执行器使用 `python:3.11-slim`，禁用网络并设置内存、CPU、进程数、只读根文件系统和超时限制。首次使用前需自行拉取镜像：

```powershell
docker pull python:3.11-slim
```

Linux 服务器无法访问 Docker daemon 时，可以使用无特权 Bubblewrap 沙箱：

```powershell
repairbench evaluate `
  ... `
  --executor bwrap
```

该执行器使用独立的网络、PID、挂载和临时目录命名空间，并通过 `prlimit` 限制内存、CPU、进程数和文件描述符。它需要系统已安装 `bwrap` 与 `prlimit`。

## 实验设计

四组实验必须使用相同的错误代码、最多修复轮数、生成配置和隐藏测试：

| 条件 | 基础模型 | 给模型的公开测试反馈 |
|---|---|---|
| `none` | 原始模型 | 不提供 |
| `raw` | 原始模型 | 原始执行信息 |
| `structured` | 原始模型 | 结构化错误类型、输入、期望与实际结果 |
| `structured` | LoRA 模型 | 同上 |

主要指标：`repair@1`、`repair@3`、公开测试通过率、隐藏测试通过率、平均修复轮数、生成字符数，以及各错误类型的成功率。

隐藏测试永远不会进入模型提示词。即使候选代码通过公开测试，最终成功也只由隐藏测试判定。

## 数据约定

任务 JSONL：

```json
{"task_id":"train_001","prompt":"...","entry_point":"function_name","canonical_solution":"def ...","public_tests":["assert ..."],"hidden_tests":["assert ..."],"setup_code":""}
```

错误样本 JSONL：

```json
{"bug_id":"train_001__boundary_condition__0","task_id":"train_001","buggy_code":"def ...","bug_type":"boundary_condition","mutation":"comparison_boundary:0"}
```

同一 `task_id` 的所有错误版本必须属于同一个数据划分。`repairbench check-splits` 会拒绝有题目交集的训练集和评测集。

## 准备 MBPP

安装数据依赖后：

```powershell
python -m pip install ".[data]"
python scripts/prepare_mbpp.py --output-dir data/processed/mbpp
```

脚本保留 MBPP 官方任务级划分，并将 `test_list` 用作公开测试、`challenge_test_list` 优先用作隐藏测试。若某条任务没有独立的挑战测试，脚本会从原测试中确定性地留出隐藏测试，并在元数据中标记来源。正式报告中必须说明这一点。

如果 Hugging Face Hub 不可访问，可以使用 Google Research 的本地 JSON/JSONL 镜像：

```powershell
python scripts/prepare_mbpp.py `
  --source-file data/raw/sanitized-mbpp.json `
  --output-dir data/processed/mbpp
```

本地镜像没有携带 Hub 的官方 split 标签，脚本会按任务 ID 排序后确定性地切成 80%/10%/10%，并在任务元数据中标记 `split_source`。这只能作为可复现实验划分，不能在报告中称为官方 split。

转换后先审计参考答案，再生成错误样本：

```powershell
python scripts/audit_tasks.py `
  --tasks data/processed/mbpp/train_tasks.jsonl `
  --output data/processed/mbpp/train_audited_tasks.jsonl `
  --rejected-output data/processed/mbpp/train_rejected.jsonl `
  --executor docker
```

审计只保留同时通过公开测试和隐藏测试的参考实现；被拒任务会保留完整执行结果，便于人工分析。开发阶段可以显式使用 `--executor local --allow-unsafe-local`，但不能把本地执行器当作安全沙箱。

## 复现实验环境

日常开发使用 `requirements/dev.txt`。复现报告中的 RTX 3090 训练环境时，先安装对应 CUDA 12.1 的 PyTorch，再安装本次实验记录的精确顶层版本：

```powershell
python -m pip install -r requirements/torch-cu121.txt
python -m pip install -r requirements/experiment.txt
```

本次记录固定了 PyTorch 2.5.1、Transformers 4.56.2、Datasets 5.0.1、PEFT 0.21.1、TRL 1.14.1、Accelerate 1.15.0 和 bitsandbytes 0.50.2。`pyproject.toml` 的可选依赖使用兼容版本范围，便于开发；正式复现应使用上述两个 requirements 文件。

发布前统一运行：

```powershell
pytest
ruff check .
python scripts/run_demo.py
python scripts/verify_release.py
```

## LoRA 训练

先用 `build-sft` 生成训练数据，再按上一节安装固定训练依赖：

```powershell
python scripts/train_sft.py `
  --train-file data/processed/mbpp/train_sft_v2_dedup.jsonl `
  --model-id models/qwen2.5-coder-0.5b-instruct `
  --output-dir outputs/qwen25-coder-05b-repair-qlora `
  --load-in-4bit `
  --gradient-checkpointing `
  --max-length 1536 `
  --epochs 3 `
  --batch-size 4 `
  --gradient-accumulation 4 `
  --device-map auto
```

本次训练使用 1241 条去重后的 SFT 样本，并将训练样本编码为与推理一致的 system/user/assistant 对话模板。单张 RTX 3090 上完成 234 步、3 个 epoch，用时 183.4 秒；最终训练损失为 0.1588，适配器权重约 4.2 MiB。最长对话模板输入为 1469 tokens，因此使用 `max-length=1536`，没有截断训练样本。

评测微调模型：

```powershell
repairbench evaluate `
  --tasks data/processed/mbpp/test_audited_tasks.jsonl `
  --bugs data/processed/mbpp/test_bugs_v2.jsonl `
  --model-id models/qwen2.5-coder-0.5b-instruct `
  --adapter-id outputs/qwen25-coder-05b-repair-qlora `
  --condition structured `
  --max-repairs 1 `
  --max-new-tokens 256 `
  --executor bwrap `
  --output artifacts/test_qlora_structured.jsonl
```

训练需要 CUDA 版 PyTorch 和 NVIDIA GPU；没有 GPU 时仍可完成数据构建、审计和小规模 CPU 链路验证。0.5B 是本轮受控实验的模型规模，不代表最终目标模型；切换到 1.5B 时需要重新做显存、序列长度和生成速度预算。

## 安全边界

`local` 执行器只有超时和独立子进程，不是安全沙箱，不能运行模型生成或来源不可信的代码。正式采样与评测必须使用 Docker 或更强的隔离环境。本项目的 Docker 配置降低风险，但不构成对抗恶意代码的完整安全承诺。

## 模型错误数据增强实验

第一版训练数据主要来自 AST 变异，运行时错误占比较高。为检验数据分布是否限制了迁移能力，新增了一次预先规定的受控实验：使用同一个基础模型 `Qwen2.5-Coder-0.5B-Instruct`，只根据 MBPP 训练题目生成候选实现，不向生成器提供参考答案或测试；每题最多保留 2 个同时被公开测试和隐藏测试检测到、可解析且入口函数存在的错误。最终得到 469 个模型生成错误，其中逻辑错误 297、运行时错误 171、超时错误 1。

混合集固定为 1241 条：保留 469 条模型错误，并从原 AST 变异数据中按 `boundary_condition=122`、`logic_error=267`、`runtime_error=383` 抽样 772 条。使用固定随机种子 42、相同的 3 epoch QLoRA 配置重新训练，未继续训练旧适配器。实际对话模板编码后的最长样本为 1534 tokens，低于训练上限 1536。构建、哈希和审计记录见 `artifacts/train_sft_mixed_controlled_manifest.json` 与 `artifacts/train_sft_mixed_controlled_quality.md`。

### 结果

所有评测均使用同一批错误案例、单轮 greedy 修复、最多 256 个新 token，并在 Bubblewrap 隔离环境执行。HumanEvalFix 的公开 `check()` 已拆成独立断言，143 个审计案例中有 130 个能提供 expected/actual，隐藏测试不进入提示词。

| 训练数据 | 无反馈 | 结构化反馈 |
|---|---:|---:|
| 不微调 | 55/143 = 38.5% | 38/143 = 26.6% |
| 仅 AST 合成错误 | 42/143 = 29.4% | 41/143 = 28.7% |
| AST + 模型生成错误 | **72/143 = 50.3%** | **71/143 = 49.7%** |

相同提示条件下，混合数据相较仅 AST 数据都增加 30 个成功案例：无反馈配对 McNemar `p=2.27e-7`，结构化反馈 `p=6.04e-7`。MBPP 内部测试也从 106/152 提升到 114/152，但 `p=0.0768`，只能视为方向一致。

反馈消融得到负结果：混合模型无反馈 72/143、结构化反馈 71/143（`p=1.0`）；最多三轮时 `repair@3` 分别为 77/143 和 75/143（`p=0.804`）。因此当前证据支持“模型真实错误改善训练分布”，不支持“模型已经学会有效使用执行反馈”。完整解读见 `artifacts/experiment_report.md`。

### 反馈置乱与 on-policy 轨迹

为直接检查模型是否依赖反馈内容，使用 seed 42–46 将 143 条初始执行反馈分别进行 5 次确定性置换。每个映射都包含 143 个唯一来源且没有自配对，保持反馈集合不变，仅破坏反馈和当前代码的对应关系。5 次置乱的 `repair@1` 为 64、69、59、65、65/143，均值 64.4/143（45.0%），范围 59–69；正确反馈为 71/143（49.7%）。5 次结果都更低，但逐次 McNemar 检验的 `p` 值为 0.143、0.832、0.0169、0.180、0.263，只有 1/5 在未校正条件下小于 0.05。因此这是“模型对反馈内容存在方向性敏感”的探索性证据，不是稳定反馈收益的证明。

从结构化反馈三轮结果中提取“第一轮失败、后续通过公开与隐藏测试”的轨迹，只得到 4 条，其中第 2 轮成功 2 条、第 3 轮成功 2 条，覆盖 excess logic、missing logic、operator misuse 和 variable misuse 各 1 条。该规模不足以支持一次可信的新增微调，因此项目没有用 4 条样本继续训练并宣称提升。

```powershell
# 对 seed 42–46 分别运行；下面以 seed 42 为例
repairbench evaluate `
  --tasks data/processed/humanevalfix_atomized/audited_tasks.jsonl `
  --bugs data/processed/humanevalfix_atomized/audited_bugs.jsonl `
  --model-id models/qwen2.5-coder-0.5b-instruct `
  --adapter-id outputs/qwen25-coder-05b-repair-qlora-mixed-controlled `
  --condition structured `
  --max-repairs 1 `
  --max-new-tokens 256 `
  --executor bwrap `
  --shuffle-feedback-seed 42 `
  --output artifacts/humanevalfix_atomized_mixed_structured_shuffled_seed42.jsonl

python scripts/extract_on_policy_repairs.py `
  --results artifacts/humanevalfix_atomized_mixed_structured_k3.jsonl `
  --output data/processed/on_policy/humanevalfix_later_success_sft.jsonl `
  --summary artifacts/on_policy_later_success_summary.json
```

需要诚实说明：HumanEvalFix 已在开发中被查看，不是最终盲测；目前只有一个训练 seed、模型规模和混合比例，反馈置乱也只有 5 个排列；混合时同时加入模型错误并减少合成运行时错误，尚不能拆分两者各自贡献。若继续，应扩大候选采样获得足量 on-policy 轨迹，并在从未用于开发的新修复集上验证。

## 数据、模型与方法来源

- 基础模型：[Qwen2.5-Coder-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B-Instruct)
- 训练任务：[Mostly Basic Python Problems（MBPP）](https://huggingface.co/datasets/google-research-datasets/mbpp)
- 外部修复评测：[HumanEvalPack](https://huggingface.co/datasets/bigcode/humanevalpack)
- 参数高效微调组件：[PEFT LoRA](https://huggingface.co/docs/peft/main/en/conceptual_guides/lora) 与 [TRL SFTTrainer](https://huggingface.co/docs/trl/sft_trainer)
- 执行反馈研究背景：[CYCLE: Learning to Self-Refine the Code Generation](https://arxiv.org/abs/2403.18746)
- 不可信代码执行的安全参考：[BigCode Evaluation Harness](https://github.com/bigcode-project/bigcode-evaluation-harness)

这些资源提供模型、数据或方法背景；本仓库的混合数据构建、任务级防泄漏、反馈消融和配对统计均由本项目实现。
