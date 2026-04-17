from __future__ import annotations

from .env_base import EnvBase
from agents.agent_base import AgentBase
from prompts import instruct_prompts as IP


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
    # 逐轮可见性重建（覆盖基类的时间线分组）
    # ------------------------------------------------------------------

    def _build_timeline_groups(self, agent_id: int) -> list[dict]:
        """BrainWrite 的可见性是滑动窗口——每轮看到的前任不同。

        不能使用基类的时间线扫描（全局历史中消息的物理顺序与 BrainWrite
        的环形传递因果关系不一致），需要按 agent 实际发言轮次逐轮重建。
        """
        n = len(self.agents)
        agent_ids = [a.agent_id for a in self.agents]
        idx = agent_ids.index(agent_id)

        mine_by_round: dict[int, str] = {}
        for entry in self.global_history:
            if entry["agent_id"] == agent_id:
                mine_by_round[entry["round"]] = entry["content"]

        groups: list[dict] = []

        if 1 in mine_by_round or self.current_round == 1:
            groups.append({"others": [], "mine": mine_by_round.get(1)})

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
            groups.append({
                "others": others,
                "mine": mine_by_round.get(speaking_round),
            })

        return groups

    # ------------------------------------------------------------------
    # Prompt 渲染
    # ------------------------------------------------------------------

    def format_round_prompt(self, turn_num: int, others_entries: list[dict], agent_id: int) -> str:
        lines = []
        for entry in others_entries:
            speaker = self.get_agent_display_name(entry["agent_id"])
            lines.append(IP.BRAINWRITE_SPEAKER_LINE.format(speaker=speaker, content=entry["content"]))
        body = "\n".join(lines)
        return IP.BRAINWRITE_ROUND.format(turn_num=turn_num, body=body)

    def format_initial_prompt(self, agent_id: int) -> str:
        return IP.BRAINWRITE_INITIAL
