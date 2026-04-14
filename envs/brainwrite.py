from __future__ import annotations

from .env_base import EnvBase
from agents.agent_base import AgentBase


class BrainWrite(EnvBase):
    """脑力书写环境：环形传递，每轮 agent_i 看到 agent_((i-k) mod n) 在第 k 轮的内容。"""

    mode = "brainwrite"

    def get_visible_messages(self, agent_id: int) -> list[dict]:
        """BrainWrite 环形传递可见性。

        在第 R 轮（current_round）发言时，agent_i 可以看到：
        - 第 1 轮: agent_((i - 1) mod n) 的发言
        - 第 2 轮: agent_((i - 2) mod n) 的发言
        - ...
        - 第 R-1 轮: agent_((i - (R-1)) mod n) 的发言
        依此类推，形成环形传递链。
        """
        n = len(self.agents)
        agent_ids = [a.agent_id for a in self.agents]
        idx = agent_ids.index(agent_id)

        visible = []
        for entry in self.global_history:
            r = entry["round"]
            if r >= self.current_round:
                continue
            offset = self.current_round - r
            source_idx = (idx - offset) % n
            source_id = agent_ids[source_idx]
            if entry["agent_id"] == source_id and entry["round"] == r:
                visible.append(entry)
        return visible

    # ------------------------------------------------------------------
    # 逐轮可见性重建（覆盖基类的按消息轮次分组）
    # ------------------------------------------------------------------

    def _group_history_by_round(self, agent_id: int) -> dict[int, dict]:
        """BrainWrite 的可见性是滑动窗口——每轮看到的前任不同。

        按 agent 实际发言轮次重建：第 R 轮发言前，模拟 current_round=R
        的环形可见性规则，收集该轮实际可见的 others。
        """
        n = len(self.agents)
        agent_ids = [a.agent_id for a in self.agents]
        idx = agent_ids.index(agent_id)

        mine_by_round: dict[int, str] = {}
        for entry in self.global_history:
            if entry["agent_id"] == agent_id:
                mine_by_round[entry["round"]] = entry["content"]

        result: dict[int, dict] = {}

        if 1 in mine_by_round or self.current_round == 1:
            result[1] = {"others": [], "mine": mine_by_round.get(1)}

        upper = min(self.current_round, self.max_rounds) + 1
        for speaking_round in range(2, upper):
            others: list[dict] = []
            for r in range(1, speaking_round):
                offset = speaking_round - r
                source_idx = (idx - offset) % n
                source_id = agent_ids[source_idx]
                for entry in self.global_history:
                    if entry["agent_id"] == source_id and entry["round"] == r:
                        others.append(entry)
            result[speaking_round] = {
                "others": others,
                "mine": mine_by_round.get(speaking_round),
            }

        return result

    # ------------------------------------------------------------------
    # Prompt 渲染
    # ------------------------------------------------------------------

    def format_round_prompt(self, round_num: int, others_entries: list[dict], agent_id: int) -> str:
        lines = []
        for entry in others_entries:
            speaker = self.get_agent_name(entry["agent_id"])
            if self.get_agent(entry["agent_id"]).is_human:
                speaker = f"人类专家 {speaker}"
            lines.append(f"- {speaker} 的草稿：{entry['content']}")
        return (
            f"在第 {round_num} 轮讨论中，你收到了传递过来的脑力书写草稿。"
            f"请仔细阅读前人的思路，并在此基础上继续延伸你的专业见解。\n"
            f"草稿内容如下：\n" + "\n".join(lines)
        )

    def format_initial_prompt(self, agent_id: int) -> str:
        return "这是第一轮脑力书写，请你写下你对讨论主题的初始思考和创意。"
