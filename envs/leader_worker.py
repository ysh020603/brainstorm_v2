from __future__ import annotations

from .env_base import EnvBase
from agents.agent_base import AgentBase


class LeaderWorker(EnvBase):
    """领导-组员模式：Worker 与 Leader 双向隔离，只能互相看到对方的发言。"""

    mode = "leader_worker"

    def __init__(
        self,
        agents: list[AgentBase],
        topic: str,
        max_rounds: int,
        leader_ids: list[int],
        log_dir: str | None = None,
    ):
        super().__init__(agents, topic, max_rounds, log_dir)
        self.leader_ids = set(leader_ids)

    def _is_leader(self, agent_id: int) -> bool:
        return agent_id in self.leader_ids

    def _prepare_round_order(self):
        """先 Worker 后 Leader。"""
        workers = [a for a in self.agents if not self._is_leader(a.agent_id)]
        leaders = [a for a in self.agents if self._is_leader(a.agent_id)]
        self.round_order = workers + leaders

    def get_visible_messages(self, agent_id: int) -> list[dict]:
        """双向隔离：Worker 只看 Leader 发言，Leader 只看 Worker 发言。"""
        is_leader = self._is_leader(agent_id)
        visible = []
        for entry in self.global_history:
            entry_is_leader = self._is_leader(entry["agent_id"])
            if is_leader and not entry_is_leader:
                visible.append(entry)
            elif not is_leader and entry_is_leader:
                visible.append(entry)
        return visible

    # ------------------------------------------------------------------
    # Prompt 渲染（Leader / Worker 双视角）
    # ------------------------------------------------------------------

    def format_round_prompt(self, round_num: int, others_entries: list[dict], agent_id: int) -> str:
        if self._is_leader(agent_id):
            names = []
            lines = []
            for entry in others_entries:
                speaker = self.get_agent_name(entry["agent_id"])
                if self.get_agent(entry["agent_id"]).is_human:
                    speaker = f"人类专家 {speaker}"
                names.append(speaker)
                lines.append(f"- {speaker} 汇报称：{entry['content']}")
            names_str = "、".join(names)
            return (
                f"在第 {round_num} 轮汇总中，你收到了来自组员 {names_str} 的分析报告。"
                f"作为 Leader，请综合以下信息给出你的指导意见：\n" + "\n".join(lines)
            )
        else:
            lines = []
            for entry in others_entries:
                speaker = self.get_agent_name(entry["agent_id"])
                if self.get_agent(entry["agent_id"]).is_human:
                    speaker = f"人类专家 {speaker}"
                lines.append(f"- {speaker} 的指导：{entry['content']}")
            return (
                f"在第 {round_num} 轮推进中，你收到了来自 Leader 的最新战略指导。"
                f"请根据以下指导调整你的专业方案：\n" + "\n".join(lines)
            )

    def format_initial_prompt(self, agent_id: int) -> str:
        if self._is_leader(agent_id):
            return "作为 Leader，请率先给出你的战略方向和指导意见。"
        return "作为组员，请率先提交你对主题的初步分析报告。"
