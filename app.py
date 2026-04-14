#!/usr/bin/env python3
"""Brainstorm Streamlit 前端。

启动：streamlit run app.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
from agents.agent_base import EnvState
from agents.agent_llm import AgentLLM
from agents.agent_human import AgentHuman
from envs.brainwrite import BrainWrite
from envs.round_robin import RoundRobin
from envs.random_env import RandomEnv
from envs.leader_worker import LeaderWorker
from prompts.topics import TOPICS, EXPERTS

ENV_MAP = {
    "brainwrite": BrainWrite,
    "round_robin": RoundRobin,
    "random": RandomEnv,
    "leader_worker": LeaderWorker,
}

MODE_LABELS = {
    "brainwrite": "脑力书写 (BrainWrite)",
    "round_robin": "轮流发言 (Round Robin)",
    "random": "随机发言 (Random)",
    "leader_worker": "领导-组员 (Leader-Worker)",
}

AGENT_COLORS = [
    "🔵", "🟢", "🟠", "🟣", "🔴", "🟡", "⚪", "🟤",
]


def init_session_state():
    defaults = {
        "env": None,
        "started": False,
        "discussion_over": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def render_sidebar():
    """侧边栏：配置讨论参数。"""
    st.sidebar.header("讨论配置")

    mode = st.sidebar.selectbox(
        "讨论形式",
        options=list(MODE_LABELS.keys()),
        format_func=lambda x: MODE_LABELS[x],
    )

    topic_key = st.sidebar.selectbox(
        "预设话题",
        options=["自定义"] + list(TOPICS.keys()),
    )
    if topic_key == "自定义":
        topic = st.sidebar.text_area("输入话题", value="", height=80)
    else:
        topic = st.sidebar.text_area("话题内容", value=TOPICS[topic_key], height=80)

    max_rounds = st.sidebar.number_input("讨论轮数", min_value=1, max_value=20, value=3)

    st.sidebar.subheader("Agent 配置")
    num_agents = st.sidebar.number_input("Agent 总数", min_value=2, max_value=8, value=4)

    agent_configs = []
    expert_keys = list(EXPERTS.keys())

    for i in range(num_agents):
        with st.sidebar.expander(f"Agent {i + 1}", expanded=(i == 0)):
            is_human = st.checkbox(f"人类参与者", key=f"human_{i}")
            name = st.text_input("名称", value=f"专家{i + 1}", key=f"name_{i}")

            default_expert = expert_keys[i % len(expert_keys)]
            role_source = st.selectbox(
                "角色来源",
                options=["预设角色", "自定义"],
                key=f"role_source_{i}",
            )
            if role_source == "预设角色":
                expert_choice = st.selectbox(
                    "选择专家",
                    options=expert_keys,
                    index=i % len(expert_keys),
                    key=f"expert_{i}",
                )
                role = EXPERTS[expert_choice]
            else:
                role = st.text_area("角色背景", value="", key=f"role_{i}", height=60)

            if not is_human:
                api_key = st.text_input("API Key", type="password", key=f"api_key_{i}")
                base_url = st.text_input(
                    "Base URL",
                    value="https://open.bigmodel.cn/api/paas/v4",
                    key=f"base_url_{i}",
                )
                model = st.text_input("模型", value="glm-4-flash-250414", key=f"model_{i}")
                temperature = st.slider("Temperature", 0.0, 2.0, 0.7, 0.1, key=f"temp_{i}")
            else:
                api_key = base_url = model = None
                temperature = None

            agent_configs.append({
                "is_human": is_human,
                "name": name,
                "role": role,
                "api_key": api_key,
                "base_url": base_url,
                "model": model,
                "temperature": temperature,
            })

    leader_ids = []
    if mode == "leader_worker":
        st.sidebar.subheader("Leader 指定")
        for i in range(num_agents):
            if st.sidebar.checkbox(
                f"{agent_configs[i]['name']} 是 Leader",
                key=f"leader_{i}",
            ):
                leader_ids.append(i + 1)

    return mode, topic, max_rounds, agent_configs, leader_ids


def build_agents_from_config(configs: list[dict]) -> list:
    agents = []
    for i, cfg in enumerate(configs):
        agent_id = i + 1
        if cfg["is_human"]:
            agents.append(AgentHuman(
                agent_id=agent_id,
                name=cfg["name"],
                role_background=cfg["role"],
            ))
        else:
            api_config = {
                "api_key": cfg["api_key"],
                "base_url": cfg["base_url"],
            }
            inference_config = {"model": cfg["model"]}
            if cfg["temperature"] is not None:
                inference_config["temperature"] = cfg["temperature"]

            agents.append(AgentLLM(
                agent_id=agent_id,
                name=cfg["name"],
                role_background=cfg["role"],
                api_config=api_config,
                inference_config=inference_config,
            ))
    return agents


def render_history(env):
    """渲染聊天历史。"""
    for entry in env.global_history:
        agent = env.get_agent(entry["agent_id"])
        color = AGENT_COLORS[(entry["agent_id"] - 1) % len(AGENT_COLORS)]
        role_icon = "🧑" if agent.is_human else "🤖"
        with st.chat_message(name=entry["agent_name"], avatar=role_icon):
            st.markdown(f"**{color} {entry['agent_name']}** (第{entry['round']}轮)")
            st.markdown(entry["content"])


def render_status(env):
    """渲染状态信息。"""
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("当前轮次", f"{min(env.current_round, env.max_rounds)} / {env.max_rounds}")
    with col2:
        st.metric("总发言数", len(env.global_history))
    with col3:
        state_labels = {
            EnvState.WAITING_LLM: "等待 LLM 响应",
            EnvState.WAITING_HUMAN: "等待人类输入",
            EnvState.ROUND_COMPLETE: "本轮结束",
            EnvState.FINISHED: "讨论结束",
        }
        st.metric("状态", state_labels.get(env.state, str(env.state)))


def main():
    st.set_page_config(page_title="Brainstorm 头脑风暴", page_icon="🧠", layout="wide")
    st.title("🧠 Brainstorm 头脑风暴系统")

    init_session_state()
    mode, topic, max_rounds, agent_configs, leader_ids = render_sidebar()

    if not st.session_state.started:
        st.info("请在左侧配置讨论参数，然后点击下方按钮开始讨论。")
        if st.button("开始讨论", type="primary"):
            if not topic.strip():
                st.error("请输入讨论话题！")
                return
            for i, cfg in enumerate(agent_configs):
                if not cfg["is_human"] and not cfg["api_key"]:
                    st.error(f"Agent {i + 1} ({cfg['name']}) 的 API Key 不能为空！")
                    return

            agents = build_agents_from_config(agent_configs)
            env_cls = ENV_MAP[mode]
            if mode == "leader_worker":
                env = env_cls(
                    agents=agents,
                    topic=topic,
                    max_rounds=max_rounds,
                    leader_ids=leader_ids,
                    log_dir=os.path.join(os.path.dirname(__file__), "log_human"),
                )
            else:
                has_human = any(c["is_human"] for c in agent_configs)
                log_subdir = "log_human" if has_human else "log"
                env = env_cls(
                    agents=agents,
                    topic=topic,
                    max_rounds=max_rounds,
                    log_dir=os.path.join(os.path.dirname(__file__), log_subdir),
                )
            env.init()
            st.session_state.env = env
            st.session_state.started = True
            st.session_state.discussion_over = False
            st.rerun()
        return

    env = st.session_state.env

    render_status(env)
    st.divider()
    render_history(env)

    if env.state == EnvState.FINISHED:
        if not st.session_state.discussion_over:
            log_path = env.save_log()
            st.session_state.discussion_over = True
            st.success(f"讨论结束！日志已保存至: {log_path}")
        else:
            st.success("讨论已结束。")

        if st.button("开始新讨论"):
            st.session_state.started = False
            st.session_state.env = None
            st.session_state.discussion_over = False
            st.rerun()
        return

    if env.state == EnvState.WAITING_HUMAN:
        agent = env._get_current_agent()
        st.info(f"等待 **{agent.name}** 输入（第{env.current_round}轮）")
        with st.form("human_input_form"):
            user_input = st.text_area(
                f"{agent.name} 的发言",
                height=150,
                placeholder="请输入你的观点...",
            )
            submitted = st.form_submit_button("提交发言", type="primary")
            if submitted and user_input.strip():
                agent.submit_input(user_input.strip())
                env.step()
                st.rerun()
            elif submitted:
                st.warning("发言内容不能为空！")

    elif env.state in (EnvState.WAITING_LLM, EnvState.ROUND_COMPLETE):
        agent = env._get_current_agent()
        if st.button(f"下一步: {agent.name} 发言（第{env.current_round}轮）", type="primary"):
            with st.spinner(f"{agent.name} 正在思考..."):
                env.step()
            st.rerun()


if __name__ == "__main__":
    main()
