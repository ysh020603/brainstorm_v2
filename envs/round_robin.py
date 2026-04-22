from __future__ import annotations

from .env_base import EnvBase
from agents.agent_base import AgentBase
from prompts import instruct_prompts as IP


class RoundRobin(EnvBase):
    """轮流发言环境：所有人能看到之前所有轮的发言 + 当前轮排在自己前面的发言。"""

    mode = "round_robin"

    def get_visible_messages(self, agent_id: int) -> list[dict]:
        visible = []
        for entry in self.global_history:
            if entry["round"] < self.current_round:
                visible.append(entry)
            elif entry["round"] == self.current_round:
                if entry["agent_id"] != agent_id:
                    visible.append(entry)
        return visible

    # ------------------------------------------------------------------
    # Prompt 渲染（圆桌讨论模板）
    # ------------------------------------------------------------------

    def format_round_prompt(self, turn_num: int, others_entries: list[dict], agent_id: int) -> str:
        lines = []
        for entry in others_entries:
            speaker = self.get_agent_display_name(entry["agent_id"])
            lines.append(IP.SPEAKER_LINE.format(speaker=speaker, content=entry["content"]))
        body = "\n".join(lines)
        if turn_num == 1:
            return IP.ROUND_FIRST.format(body=body)
        return IP.ROUND_FOLLOW.format(body=body)
