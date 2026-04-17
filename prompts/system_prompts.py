MODE_NAMES = {
    "brainwrite": "脑力书写（BrainWrite）",
    "round_robin": "轮流发言（Round Robin）",
    "random": "随机发言（Random）",
    "leader_worker": "领导-组员模式（Leader-Worker）",
}

_TEMPLATE_WITH_ROLE = (
    "你正在参加一场头脑风暴讨论。\n"
    "讨论形式：{mode_display}\n"
    "参与人数：{total_agents}\n"
    "讨论主题：{topic}\n\n"
    "你的角色背景：{role_background}\n\n"
    "请根据你的专业背景，参与头脑风暴并给出你的思考。\n"
    "- 请仔细阅读其他参与者的发言，提出自己的观点。\n"
    "- 生成内容要求有启发性，并提出具体的看法或方案。\n"
    "- 只生成严谨的发言，不要生成其他内容，不需要介绍自己。"
)

_TEMPLATE_NO_ROLE = (
    "你正在参加一场头脑风暴讨论。\n"
    "讨论形式：{mode_display}\n"
    "参与人数：{total_agents}\n"
    "讨论主题：{topic}\n\n"
    "- 请仔细阅读其他参与者的发言，提出自己的观点。\n"
    "- 生成内容要求有启发性，并提出具体的看法或方案。\n"
    "- 只生成严谨的发言，不要生成其他内容，不需要介绍自己。"
)


def build_system_prompt(
    mode: str,
    total_agents: int,
    topic: str,
    role_background: str,
    identity_prompt: str = "",
) -> str:
    mode_display = MODE_NAMES.get(mode, mode)

    if role_background and role_background.strip():
        base = _TEMPLATE_WITH_ROLE.format(
            mode_display=mode_display,
            total_agents=total_agents,
            topic=topic,
            role_background=role_background,
        )
    else:
        base = _TEMPLATE_NO_ROLE.format(
            mode_display=mode_display,
            total_agents=total_agents,
            topic=topic,
        )

    if identity_prompt:
        return identity_prompt + "\n\n" + base
    return base
