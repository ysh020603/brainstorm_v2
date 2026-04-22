_TEMPLATE_WITH_ROLE = (
    "You are participating in a brainstorming discussion.\n"
    "Number of participants: {total_agents}\n"
    "Discussion topic: {topic}\n\n"
    "Your role background: {role_background}\n\n"
    "Based on your professional background, participate in the brainstorming and share your thoughts.\n"
    "- Before making your statement, please carefully read the statements of other participants.\n"
    "- Based on the discussion topic and existing statements, provide your contribution to demonstrate your **creativity**.\n"
    "- Only produce rigorous statements; do not generate other content or introduce yourself.\n"
    "- Keep your statements concise and to the point."
)

_TEMPLATE_NO_ROLE = (
    "You are participating in a brainstorming discussion.\n"
    "Number of participants: {total_agents}\n"
    "Discussion topic: {topic}\n\n"
    "- Before making your statement, please carefully read the statements of other participants.\n"
    "- Based on the discussion topic and existing statements, provide your contribution to demonstrate your **creativity**.\n"
    "- Only produce rigorous statements; do not generate other content or introduce yourself.\n"
    "- Keep your statements concise and to the point."
)


def build_system_prompt(
    total_agents: int,
    topic: str,
    role_background: str,
    identity_prompt: str = "",
) -> str:
    if role_background and role_background.strip():
        base = _TEMPLATE_WITH_ROLE.format(
            total_agents=total_agents,
            topic=topic,
            role_background=role_background,
        )
    else:
        base = _TEMPLATE_NO_ROLE.format(
            total_agents=total_agents,
            topic=topic,
        )

    if identity_prompt:
        return identity_prompt + "\n\n" + base
    return base
