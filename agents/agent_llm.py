from __future__ import annotations

from .agent_base import AgentBase
from tools.call_openai import call_openai


class AgentLLM(AgentBase):
    """通过 OpenAI 兼容 API 进行推理的 Agent。"""

    def __init__(
        self,
        agent_id: int,
        name: str,
        role_background: str,
        api_config: dict,
        inference_config: dict,
    ):
        super().__init__(agent_id, name, role_background)
        self.api_config = api_config
        self.inference_config = inference_config

    def talk(self, messages: list[dict]) -> str:
        self.last_messages = messages
        return call_openai(messages, self.api_config, self.inference_config)

    def get_agent_info(self) -> dict:
        info = super().get_agent_info()
        info.update({
            "model": self.inference_config.get("model", "unknown"),
            "temperature": self.inference_config.get("temperature"),
            "top_p": self.inference_config.get("top_p"),
            "max_tokens": self.inference_config.get("max_tokens"),
        })
        return info
