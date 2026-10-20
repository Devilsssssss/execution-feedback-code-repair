#!/usr/bin/env bash
set -euo pipefail
cd /data/qianketong/projects/execution-feedback-code-repair
source .venv/bin/activate
export PYTHONPATH=src
ADAPTER=outputs/qwen25-coder-05b-repair-qlora-feedback-contract-v3
until [ -f "$ADAPTER/adapter_config.json" ]; do
  echo "Waiting for $ADAPTER/adapter_config.json" >&2
  sleep 30
done
MODEL=models/qwen2.5-coder-0.5b-instruct
OUT=artifacts
HEF=data/processed/humanevalfix_atomized
CONF=data/processed/humanevalfix_atomized/confirmation_v1
run_eval() { local gpu=$1 tasks=$2 bugs=$3 cond=$4 out=$5; shift 5; CUDA_VISIBLE_DEVICES=$gpu python -m repairbench.cli evaluate --tasks "$tasks" --bugs "$bugs" --model-id "$MODEL" --adapter-id "$ADAPTER" --condition "$cond" --output "$OUT/$out" --executor bwrap --timeout 10 "$@" > "$OUT/$out.log" 2>&1; }
run_eval 1 "$HEF/audited_tasks.jsonl" "$HEF/audited_bugs.jsonl" none humanevalfix_feedback_contract_v3_none --max-repairs 3 --max-new-tokens 256
run_eval 1 "$HEF/audited_tasks.jsonl" "$HEF/audited_bugs.jsonl" structured humanevalfix_feedback_contract_v3_structured --max-repairs 3 --max-new-tokens 256
run_eval 1 "$CONF/tasks.jsonl" "$CONF/bugs.jsonl" none humanevalfix_confirmation_v1_feedback_contract_v3_none --max-repairs 3 --max-new-tokens 256
run_eval 1 "$CONF/tasks.jsonl" "$CONF/bugs.jsonl" structured humanevalfix_confirmation_v1_feedback_contract_v3_structured --max-repairs 3 --max-new-tokens 256
echo ALL_EVALS_DONE
