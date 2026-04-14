#!/bin/bash
# Brainstorm 批量测试脚本
# 使用方法: bash run_batch.sh
#
# 请根据实际情况修改下方的 API_KEY、BASE_URL、MODEL 等参数。

# MODEL_A_NAME="Qwen3-8B"
# MODEL_A_API_KEY="EMPTY"
# MODEL_A_BASE_URL="http://172.18.39.164:8002/v1"

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ===== 配置区域 =====
# API_KEY="${API_KEY:-sk-a81d0b1ef1cb4ad687c7a14f100113e3}"
# BASE_URL="${BASE_URL:-https://api.deepseek.com/v1}"
# MODEL="${MODEL:-deepseek-chat}"

API_KEY="${API_KEY:-EMPTY}"
BASE_URL="${BASE_URL:-http://172.18.39.164:8002/v1}"
MODEL="${MODEL:-Qwen3-8B}"

TOPIC="人工智能技术能怎样帮助解决三体问题？"

AGENTS="[
  {\"name\":\"AI专家\",\"role\":\"请扮演一位顶尖的人工智能研究员，精通机器学习、深度学习、自然语言处理和计算机视觉等前沿领域。\",\"api_key\":\"${API_KEY}\",\"base_url\":\"${BASE_URL}\",\"model\":\"${MODEL}\",\"temperature\":0.7},
  {\"name\":\"数学家\",\"role\":\"请扮演一位经验丰富的数学教授，专注于抽象代数、拓扑学、数论和几何学等领域。\",\"api_key\":\"${API_KEY}\",\"base_url\":\"${BASE_URL}\",\"model\":\"${MODEL}\",\"temperature\":0.8},
  {\"name\":\"生物学家\",\"role\":\"请扮演一位杰出的生物学家，擅长分子生物学、遗传学、生态学和神经科学等领域。\",\"api_key\":\"${API_KEY}\",\"base_url\":\"${BASE_URL}\",\"model\":\"${MODEL}\",\"temperature\":0.7},
  {\"name\":\"管理学家\",\"role\":\"请扮演一位资深的企业管理顾问，拥有丰富的战略规划、组织行为、市场营销和财务管理经验。\",\"api_key\":\"${API_KEY}\",\"base_url\":\"${BASE_URL}\",\"model\":\"${MODEL}\",\"temperature\":0.8}
]"

# ===== 批量运行 =====
for MODE in brainwrite round_robin random; do
  for ROUNDS in 4; do
    echo "=========================================="
    echo "运行: mode=${MODE}, rounds=${ROUNDS}"
    echo "=========================================="
    python main_batch.py \
      --mode "$MODE" \
      --rounds "$ROUNDS" \
      --topic "$TOPIC" \
      --agents "$AGENTS"
    echo ""
  done
done

# Leader-Worker 模式单独运行（需指定 leader_ids）
for ROUNDS in 4; do
  echo "=========================================="
  echo "运行: mode=leader_worker, rounds=${ROUNDS}"
  echo "=========================================="
  python main_batch.py \
    --mode leader_worker \
    --rounds "$ROUNDS" \
    --topic "$TOPIC" \
    --agents "$AGENTS" \
    --leader_ids "[1]"
  echo ""
done

echo "全部批量测试完成！"
