"""集中管理所有讨论环境的交互指令与格式模板。

模板变量说明：
  {body}     — 由 speaker line 拼接而成的多行文本
  {speaker}  — Agent 的 display_name（如 "Agent 1"）
  {content}  — 该 Agent 的发言原文
  {turn_num} — 当前 agent 的第几次发言（1-based）
  {names}    — 逗号分隔的多个 speaker 名称（仅 LeaderWorker）
"""

# ══════════════════════════════════════════════
# 通用（env_base 默认）
# ══════════════════════════════════════════════

INITIAL_PROMPT = "现在请你率先发言，针对讨论主题分享你的观点和思考。"
ROUND_FIRST = "在讨论中，以下参与者率先发表了观点：\n{body}"
ROUND_FOLLOW = "在你上次发言后，以下参与者发表了新的观点：\n{body}"
SPEAKER_LINE = "- {speaker} 说：{content}"

# ══════════════════════════════════════════════
# RoundRobin / Random（圆桌讨论）
# ══════════════════════════════════════════════

ROUNDTABLE_FIRST = "在本轮圆桌讨论中，在你发言之前，以下参与者发表了观点：\n{body}"
ROUNDTABLE_FOLLOW = "在你上次发言后的圆桌讨论中，以下参与者发表了新的观点：\n{body}"
ROUNDTABLE_SPEAKER_LINE = "- {speaker} 说：{content}"

# ══════════════════════════════════════════════
# BrainWrite（脑力书写）
# ══════════════════════════════════════════════

BRAINWRITE_INITIAL = "这是第一轮脑力书写，请你写下你对讨论主题的初始思考和创意。"
BRAINWRITE_ROUND = (
    "在第 {turn_num} 轮讨论中，你收到了传递过来的脑力书写草稿。"
    "请仔细阅读前人的思路，并在此基础上继续延伸你的专业见解。\n"
    "草稿内容如下：\n{body}"
)
BRAINWRITE_SPEAKER_LINE = "- {speaker} 的草稿：{content}"

# ══════════════════════════════════════════════
# LeaderWorker（领导-组员模式）
# ══════════════════════════════════════════════

LEADER_INITIAL = "作为 Leader，请率先给出你的战略方向和指导意见。"
WORKER_INITIAL = "作为组员，请率先提交你对主题的初步分析报告。"

LEADER_ROUND_FIRST = (
    "你收到了来自组员 {names} 的初步分析报告。"
    "作为 Leader，请综合以下信息给出你的指导意见：\n{body}"
)
LEADER_ROUND_FOLLOW = (
    "在你上次指导后，组员 {names} 提交了更新的分析报告。"
    "作为 Leader，请综合以下信息给出进一步的指导意见：\n{body}"
)
LEADER_SPEAKER_LINE = "- {speaker} 汇报称：{content}"

WORKER_ROUND_FIRST = (
    "你收到了来自 Leader 的战略指导。"
    "请根据以下指导制定你的专业方案：\n{body}"
)
WORKER_ROUND_FOLLOW = (
    "在你上次汇报后，Leader 给出了新的战略指导。"
    "请根据以下指导调整你的专业方案：\n{body}"
)
WORKER_SPEAKER_LINE = "- {speaker} 的指导：{content}"
