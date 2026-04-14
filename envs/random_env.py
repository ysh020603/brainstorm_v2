from __future__ import annotations

import random

from .env_base import EnvBase
from agents.agent_base import AgentBase


class RandomEnv(EnvBase):
    """随机发言环境：每轮所有人都发言一次，但顺序随机。可见性同 RoundRobin。"""

    mode = "random"

    def _prepare_round_order(self):
        """每轮随机打乱发言顺序，但保证每人发言一次。"""
        order = list(self.agents)
        random.shuffle(order)
        self.round_order = order

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
