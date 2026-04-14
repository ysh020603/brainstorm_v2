from openai import OpenAI


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
    """
    client = OpenAI(**api_config)
    completion = client.chat.completions.create(
        messages=messages,
        **inference_config,
    )
    return completion.choices[0].message.content
