import re

from openai import OpenAI

_THINK_PATTERN = re.compile(r"<think>.*?</think>", re.DOTALL)


def _clean_think_tags(text: str) -> str:
    """剔除 <think>...</think> 块并清理首尾空白。"""
    cleaned = _THINK_PATTERN.sub("", text)
    return cleaned.strip()


def call_openai(
    messages: list[dict],
    api_config: dict,
    inference_config: dict,
) -> str:
    """调用 OpenAI 兼容接口，返回模型生成的文本。

    Args:
        messages: 完整的 messages 列表 (system / user / assistant)。
        api_config: 传给 OpenAI 客户端的参数，如 api_key、base_url。
        inference_config: 传给 chat.completions.create 的参数，如 model、temperature。
            可包含 is_reasoning 布尔值控制思考模式。
    """
    client = OpenAI(**api_config)

    call_kwargs = dict(inference_config)
    is_reasoning = call_kwargs.pop("is_reasoning", False)

    if not is_reasoning:
        model_name = call_kwargs.get("model", "")
        if "glm" in model_name.lower():
            call_kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
        else:
            call_kwargs["extra_body"] = {
                "chat_template_kwargs": {"enable_thinking": False}
            }

    completion = client.chat.completions.create(
        messages=messages,
        **call_kwargs,
    )
    raw_content = completion.choices[0].message.content
    return _clean_think_tags(raw_content)
