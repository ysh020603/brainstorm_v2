#!/bin/bash
# Brainstorm 批量测试脚本
# 使用方法: bash run_batch.sh
#
# 模型配置统一维护在 config/llm_config.json 中。
# --models 参数按顺序指定每个位置使用的 model key，顺序即 position。

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ===== 配置区域 =====
CONFIG="config/llm_config.json"
MODELS="qwen3_8b_local,qwen3_8b_local,qwen3_8b_local,qwen3_8b_local"
TOPIC="人工智能技术能怎样帮助解决三体问题？"

# ===== 批量运行 =====
for MODE in brainwrite round_robin random; do
  for ROUNDS in 4; do
    echo "=========================================="
    echo "运行: mode=${MODE}, rounds=${ROUNDS}"
    echo "=========================================="
    python main_batch.py \
      --config "$CONFIG" \
      --models "$MODELS" \
      --mode "$MODE" \
      --rounds "$ROUNDS" \
      --topic "$TOPIC"
    echo ""
  done
done

# Leader-Worker 模式单独运行（需指定 leader_ids）
for ROUNDS in 4; do
  echo "=========================================="
  echo "运行: mode=leader_worker, rounds=${ROUNDS}"
  echo "=========================================="
  python main_batch.py \
    --config "$CONFIG" \
    --models "$MODELS" \
    --mode leader_worker \
    --rounds "$ROUNDS" \
    --topic "$TOPIC" \
    --leader_ids "[1]"
  echo ""
done

echo "全部批量测试完成！"
