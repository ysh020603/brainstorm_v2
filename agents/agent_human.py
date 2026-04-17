from __future__ import annotations

from .agent_base import AgentBase


class AgentHuman(AgentBase):
    """人类参与者占位 Agent，等待外部输入。"""

    def __init__(self, agent_id: int, role_background: str = "人类专家"):
        super().__init__(agent_id, role_background)
        self._pending_input: str | None = None

    @property
    def is_human(self) -> bool:
        return True

    def submit_input(self, text: str):
        """外部（Streamlit 或 CLI）将人类输入写入此处。"""
        self._pending_input = text

    def has_pending_input(self) -> bool:
        return self._pending_input is not None

    def talk(self, messages: list[dict]) -> str:
        self.last_messages = messages
        result = self._pending_input
        self._pending_input = None
        return result
