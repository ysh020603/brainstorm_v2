from __future__ import annotations

import json
import os

_DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "config",
    "llm_config.json",
)


def load_llm_config(config_path: str | None = None) -> dict:
    """加载 llm_config.json，返回 llm_agents_pool 字典。"""
    path = config_path or _DEFAULT_CONFIG_PATH
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["llm_agents_pool"]


def build_agent_from_config(model_key: str, pool: dict):
    """根据配置字典构建 AgentLLM 实例。

    Args:
        model_key: llm_agents_pool 中的键名，同时作为 config_key 记录到日志。
        pool: load_llm_config() 返回的字典。
    """
    cfg = pool[model_key]
    api_config = {
        "api_key": cfg["api_key"],
        "base_url": cfg["api_url"],
    }
    inference_config: dict = {
        "model": cfg["model_name"],
        "temperature": cfg["temperature"],
        "is_reasoning": cfg.get("is_reasoning", False),
    }
    if cfg.get("top_p") is not None:
        inference_config["top_p"] = cfg["top_p"]
    if cfg.get("max_tokens") is not None:
        inference_config["max_tokens"] = cfg["max_tokens"]

    role_background = ""
    if cfg.get("enable_identity") and cfg.get("identity_prompt"):
        role_background = cfg["identity_prompt"]

    from agents.agent_llm import AgentLLM
    return AgentLLM(
        role_background=role_background,
        api_config=api_config,
        inference_config=inference_config,
        config_key=model_key,
    )
