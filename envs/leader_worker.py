from __future__ import annotations

from .env_base import EnvBase
from agents.agent_base import AgentBase
from prompts import instruct_prompts as IP


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
        """先 Leader 后 Worker：Leader 负责定调，Worker 据此执行。"""
        leaders = [a for a in self.agents if self._is_leader(a.agent_id)]
        workers = [a for a in self.agents if not self._is_leader(a.agent_id)]
        self.round_order = leaders + workers

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
    # Prompt 渲染（统一模板）
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
