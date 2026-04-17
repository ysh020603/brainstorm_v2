from __future__ import annotations

from enum import Enum


class EnvState(Enum):
    WAITING_LLM = "waiting_llm"
    WAITING_HUMAN = "waiting_human"
    ROUND_COMPLETE = "round_complete"
    FINISHED = "finished"


class AgentBase:
    """所有 Agent 的抽象基类。

    agent_id 不在构造时传入，而是由 EnvBase 构造函数根据 shuffle 后的
    列表顺序动态分配（从 1 开始递增），同时作为展示序号和唯一标识。
    """

    def __init__(self, role_background: str, config_key: str = ""):
        self.agent_id: int = 0
        self.role_background = role_background
        self.config_key = config_key
        self.system_prompt: str | None = None
        self.last_messages: list[dict] | None = None

    @property
    def display_name(self) -> str:
        return f"Agent {self.agent_id}" if self.agent_id else "Agent ?"

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
            "config_key": self.config_key,
            "type": "human" if self.is_human else "llm",
            "role_background": self.role_background,
        }
