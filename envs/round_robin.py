from __future__ import annotations

from .env_base import EnvBase
from agents.agent_base import AgentBase


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

    def format_round_prompt(self, round_num: int, others_entries: list[dict], agent_id: int) -> str:
        lines = []
        for entry in others_entries:
            speaker = self.get_agent_name(entry["agent_id"])
            if self.get_agent(entry["agent_id"]).is_human:
                speaker = f"人类专家 {speaker}"
            lines.append(f"- {speaker} 说：{entry['content']}")
        return (
            f"在第 {round_num} 轮的圆桌讨论中，在你发言之前，"
            f"以下参与者发表了观点：\n" + "\n".join(lines)
        )
