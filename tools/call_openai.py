import re

from openai import OpenAI

MAX_REPLY_WORDS: int = 500

_THINK_PATTERN = re.compile(r"<think>.*?</think>", re.DOTALL)


def _clean_think_tags(text: str) -> str:
    """剔除 <think>...</think> 块并清理首尾空白。"""
    cleaned = _THINK_PATTERN.sub("", text)
    return cleaned.strip()


def _truncate_by_words(text: str, max_words: int = MAX_REPLY_WORDS) -> str:
    """按空白符分词，超过 max_words 则截断并追加标记。"""
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + " [Truncated]"


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
        model_name = str(call_kwargs.get("model", ""))
        model_name_l = model_name.lower()
        if "kimi" in model_name_l:
            # Kimi Instant Mode 通常要求 temperature=0.6
            call_kwargs["temperature"] = 0.6
            # Kimi 官方 API：用 thinking.type=disabled 关闭 reasoning_content；
            # 同时兼容 vLLM/SGLang 模板变量 thinking=false。
            call_kwargs["extra_body"] = {
                "thinking": {"type": "disabled"},
                "chat_template_kwargs": {"thinking": False},
            }
        elif "glm" in model_name_l or "deepseek" in model_name_l:
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
    cleaned = _clean_think_tags(raw_content)
    return _truncate_by_words(cleaned)
