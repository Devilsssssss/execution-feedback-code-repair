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
CONF=artifacts/humanevalfix_confirmation_v1_feedback_contract_v3_structured.summary.json
MAIN=artifacts/humanevalfix_feedback_contract_v3_structured.summary.json
until .venv/bin/python -c 'import json,sys; a,b=[json.load(open(p)) for p in sys.argv[1:]]; sys.exit(0 if (a.get("cases"),b.get("cases"))==(143,36) else 1)' "$MAIN" "$CONF" 2>/dev/null; do
  echo "Waiting for completed 143-case main and 36-case confirmation structured evaluations" >&2
  sleep 60
done
for SEED in 41 42 43 44 45; do
  out="$OUT/humanevalfix_feedback_contract_v3_structured_shuffled_seed$SEED"
  CUDA_VISIBLE_DEVICES=1 python -m repairbench.cli evaluate --tasks "$HEF/audited_tasks.jsonl" --bugs "$HEF/audited_bugs.jsonl" --model-id "$MODEL" --adapter-id "$ADAPTER" --condition structured --output "$out" --executor bwrap --timeout 10 --max-repairs 1 --max-new-tokens 256 --shuffle-feedback-seed "$SEED" > "$out.log" 2>&1
done
echo ALL_SHUFFLES_DONE
