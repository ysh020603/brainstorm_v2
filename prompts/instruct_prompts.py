"""Centralized management of interaction instructions and format templates
for all discussion environments.

Template variable descriptions:
  {body}     — multi-line text assembled from speaker lines
  {speaker}  — the Agent's display_name (e.g. "Agent 1")
  {content}  — the Agent's original statement
  {turn_num} — the agent's speaking turn number (1-based)
"""

# ══════════════════════════════════════════════
# Unified templates
# ══════════════════════════════════════════════

INITIAL_PROMPT = "Please speak first and share your views and thoughts on the discussion topic. Be concise."

ROUND_FIRST = "Here is the newly received information:\n{body}\nPlease respond based on the above new information. Be concise."
ROUND_FOLLOW = "Here is the newly received information:\n{body}\nPlease respond based on the above new information. Be concise."

SPEAKER_LINE = "- {speaker} say: {content}"
