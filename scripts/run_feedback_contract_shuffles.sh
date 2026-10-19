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
 for SEED in 41 42 43 44 45; do
  out="$OUT/humanevalfix_feedback_contract_v3_structured_shuffled_seed$SEED"
  CUDA_VISIBLE_DEVICES=1 python -m repairbench.cli evaluate --tasks "$HEF/tasks.jsonl" --bugs "$HEF/bugs.jsonl" --model-id "$MODEL" --adapter-id "$ADAPTER" --condition structured --output "$out" --executor bwrap --timeout 10 --max-repairs 1 --shuffle-feedback-seed "$SEED" > "$out.log" 2>&1
 done
 echo ALL_SHUFFLES_DONE
 
