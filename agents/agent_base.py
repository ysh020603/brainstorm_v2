from __future__ import annotations

from enum import Enum


class EnvState(Enum):
    WAITING_LLM = "waiting_llm"
    WAITING_HUMAN = "waiting_human"
    ROUND_COMPLETE = "round_complete"
    FINISHED = "finished"


class AgentBase:
    """所有 Agent 的抽象基类。"""

    def __init__(self, agent_id: int, name: str, role_background: str):
        self.agent_id = agent_id
        self.name = name
        self.role_background = role_background
        self.system_prompt: str | None = None
        self.last_messages: list[dict] | None = None

    @property
    def is_human(self) -> bool:
        return False

    def talk(self, messages: list[dict]) -> str:
        """接收完整 messages 列表并返回生成的发言文本。子类必须实现。"""
        raise NotImplementedError

    def get_agent_info(self) -> dict:
        """返回用于日志的 Agent 元信息。"""
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "type": "human" if self.is_human else "llm",
            "role_background": self.role_background,
        }
