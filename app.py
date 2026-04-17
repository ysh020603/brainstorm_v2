#!/usr/bin/env python3
"""Brainstorm Streamlit 前端（单人类参与实验）。

启动：streamlit run app.py

Human Evaluation 模式下，LLM Agent 从 llm_agents_pool 中自动盲抽，
用户仅需选择 LLM 数量，无需手动配置具体模型。
"""

import sys
import os
import random

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
from tools.config_loader import load_llm_config, build_agent_from_config

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
        "pending_final_ranking": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _reset_session():
    """清空讨论相关状态，恢复到初始配置界面。"""
    for key in ("started", "discussion_over", "pending_final_ranking"):
        st.session_state[key] = False
    st.session_state["env"] = None
    st.rerun()


def _load_model_pool():
    """加载 LLM 配置池（带缓存避免每次 rerun 重读）。"""
    if "llm_pool" not in st.session_state:
        try:
            st.session_state.llm_pool = load_llm_config()
        except FileNotFoundError:
            st.session_state.llm_pool = {}
    return st.session_state.llm_pool


# ── LLM Agent 自动抽取 ──

def sample_llm_keys(pool: dict, num_llm: int) -> list[str]:
    """从 llm_agents_pool 中抽取 config_key 列表。

    - num_llm <= pool size: 无放回随机抽取
    - num_llm >  pool size: 先全部取出，余额有放回补齐
    """
    keys = list(pool.keys())
    if not keys:
        return []
    if num_llm <= len(keys):
        return random.sample(keys, num_llm)
    result = list(keys)
    remaining = num_llm - len(keys)
    result.extend(random.choices(keys, k=remaining))
    random.shuffle(result)
    return result


def render_sidebar():
    """侧边栏：配置讨论参数。

    Human Evaluation 模式下不再提供 LLM 手动配置，
    用户仅选择 LLM 数量，后台自动从 pool 中盲抽。
    """

    if st.sidebar.button("🔄 重新开始", use_container_width=True):
        _reset_session()

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

    st.sidebar.subheader("Agent 配置")

    pool = _load_model_pool()
    pool_size = len(pool)

    num_llm = st.sidebar.number_input(
        f"LLM Agent 数量（配置池共 {pool_size} 个模型）",
        min_value=2, max_value=7, value=3,
    )
    max_rounds = st.sidebar.number_input(
        "讨论轮数", min_value=1, max_value=20, value=num_llm + 1,
    )

    st.sidebar.subheader("人类参与者角色")
    expert_keys = list(EXPERTS.keys())
    role_source = st.sidebar.selectbox(
        "角色来源", options=["预设角色", "自定义"], key="human_role_source",
    )
    if role_source == "预设角色":
        expert_choice = st.sidebar.selectbox(
            "选择专家", options=expert_keys, key="human_expert",
        )
        human_role = EXPERTS[expert_choice]
    else:
        human_role = st.sidebar.text_area("角色背景", value="", key="human_role", height=60)

    return mode, topic, max_rounds, num_llm, human_role


# ── 构建 Agent 列表 ──

def build_agents(num_llm: int, human_role: str, pool: dict) -> list:
    """构建 Agent 列表：1 个人类 + num_llm 个自动抽取的 LLM。

    agent_id 不在此处设置，由 EnvBase 构造函数根据 shuffle 后的顺序动态分配。
    """
    agents = []

    agents.append(AgentHuman(role_background=human_role))

    sampled_keys = sample_llm_keys(pool, num_llm)
    expert_keys = list(EXPERTS.keys())
    random.shuffle(expert_keys)

    for i, config_key in enumerate(sampled_keys):
        agent = build_agent_from_config(config_key, pool)
        if not agent.role_background:
            agent.role_background = EXPERTS[expert_keys[i % len(expert_keys)]]
        agents.append(agent)

    return agents


# ── LLM 自动连续发言引擎 ──

def auto_advance_llm(env):
    """连续调用 env.step()，直到遇到人类回合或讨论结束。"""
    while env.state not in (EnvState.WAITING_HUMAN, EnvState.FINISHED):
        env.step()


# ── BrainWrite 纸条路线渲染 ──

def render_brainwrite_history(env):
    """BrainWrite 上帝视角：按"纸条路线"分组展示，呈现链式接力。"""
    n = len(env.agents)
    agent_ids = [a.agent_id for a in env.agents]

    paper_slips: dict[int, list[dict]] = {idx: [] for idx in range(n)}

    for entry in env.global_history:
        holder_idx = agent_ids.index(entry["agent_id"])
        r = entry["round"]
        origin_idx = (holder_idx - (r - 1)) % n
        paper_slips[origin_idx].append(entry)

    for origin_idx in range(n):
        origin_agent = env.agents[origin_idx]
        entries = paper_slips[origin_idx]
        if not entries:
            continue

        color = AGENT_COLORS[origin_idx % len(AGENT_COLORS)]
        with st.expander(
            f"{color} 纸条路线 {origin_idx + 1}（由 {origin_agent.display_name} 发起）",
            expanded=True,
        ):
            for entry in sorted(entries, key=lambda e: e["round"]):
                agent = env.get_agent(entry["agent_id"])
                role_icon = "🧑" if agent.is_human else "🤖"
                entry_color = AGENT_COLORS[(entry["agent_id"] - 1) % len(AGENT_COLORS)]
                with st.chat_message(name=entry["agent_name"], avatar=role_icon):
                    st.markdown(
                        f"**{entry_color} {entry['agent_name']}** (第{entry['round']}轮)"
                    )
                    st.markdown(entry["content"])


def render_history(env):
    """渲染全局聊天历史（上帝视角）。BrainWrite 使用纸条路线视图。"""
    if isinstance(env, BrainWrite):
        render_brainwrite_history(env)
        return

    for entry in env.global_history:
        agent = env.get_agent(entry["agent_id"])
        color = AGENT_COLORS[(entry["agent_id"] - 1) % len(AGENT_COLORS)]
        role_icon = "🧑" if agent.is_human else "🤖"
        with st.chat_message(name=entry["agent_name"], avatar=role_icon):
            st.markdown(f"**{color} {entry['agent_name']}** (第{entry['round']}轮)")
            st.markdown(entry["content"])


def render_agent_perspective(env, agent):
    """渲染人类 Agent 的专属可见上下文视角。"""
    messages = env.build_messages_for_agent(agent)
    color = AGENT_COLORS[(agent.agent_id - 1) % len(AGENT_COLORS)]

    for msg in messages:
        if msg["role"] == "system":
            continue
        elif msg["role"] == "assistant":
            with st.chat_message(name=agent.display_name, avatar="🧑"):
                st.markdown(f"**{color} {agent.display_name}** (你的历史发言)")
                st.markdown(msg["content"])
        elif msg["role"] == "user":
            with st.chat_message(name="讨论进展", avatar="📋"):
                st.markdown(msg["content"])


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


# ── 最终排名机制（所有讨论形式通用） ──

def render_final_ranking_form(env):
    """讨论结束后，人类对所有其他 Agent 进行一次总排名。

    排名结果使用 position（动态展示序号）和 config_key（配置键名）标识，
    不再使用旧的 agent_id / agent_name 字符串。

    Returns True 表示排名已提交，False 表示尚未提交或校验失败。
    """
    others = [a for a in env.agents if not a.is_human]
    if not others:
        return False

    st.subheader("📊 最终排名 — 综合评价其他专家表现")
    st.info("请根据整场讨论中各位专家的综合表现进行排名（1 = 最佳）")

    num_others = len(others)

    with st.form("final_ranking_form"):
        rankings = {}
        for idx, agent in enumerate(others):
            color = AGENT_COLORS[(agent.agent_id - 1) % len(AGENT_COLORS)]
            rank = st.selectbox(
                f"{color} {agent.display_name} 的排名",
                options=list(range(1, num_others + 1)),
                index=idx,
                key=f"final_rank_{agent.agent_id}",
            )
            rankings[agent.agent_id] = rank

        submitted = st.form_submit_button("提交排名", type="primary")
        if submitted:
            rank_values = list(rankings.values())
            if len(set(rank_values)) != len(rank_values):
                st.error("排名不能重复！请为每位专家分配不同的名次。")
                return False

            ranking_data = []
            for agent in others:
                ranking_data.append({
                    "position": agent.agent_id,
                    "config_key": agent.config_key,
                    "rank": rankings[agent.agent_id],
                })
            ranking_data.sort(key=lambda x: x["rank"])

            env.final_rankings = ranking_data
            st.session_state.pending_final_ranking = False
            return True

    return False


def main():
    st.set_page_config(page_title="Brainstorm 头脑风暴", page_icon="🧠", layout="wide")
    st.title("🧠 Brainstorm 头脑风暴系统")

    init_session_state()
    mode, topic, max_rounds, num_llm, human_role = render_sidebar()

    # ── 尚未开始：等待用户点击"开始讨论" ──
    if not st.session_state.started:
        st.info("请在左侧配置讨论参数，然后点击下方按钮开始讨论。")
        if st.button("开始讨论", type="primary"):
            if not topic.strip():
                st.error("请输入讨论话题！")
                return

            pool = _load_model_pool()
            agents = build_agents(num_llm, human_role, pool)
            random.shuffle(agents)

            env_cls = ENV_MAP[mode]
            if mode == "leader_worker":
                leader_ids = [i + 1 for i, a in enumerate(agents) if a.is_human]
                env = env_cls(
                    agents=agents,
                    topic=topic,
                    max_rounds=max_rounds,
                    leader_ids=leader_ids,
                    log_dir=os.path.join(os.path.dirname(__file__), "log_human"),
                )
            else:
                env = env_cls(
                    agents=agents,
                    topic=topic,
                    max_rounds=max_rounds,
                    log_dir=os.path.join(os.path.dirname(__file__), "log_human"),
                )
            env.init()
            st.session_state.env = env
            st.session_state.started = True
            st.session_state.discussion_over = False
            st.session_state.pending_final_ranking = False

            with st.spinner("专家们正在激烈讨论中..."):
                auto_advance_llm(env)
            st.rerun()
        return

    # ── 讨论已开始 ──
    env = st.session_state.env

    render_status(env)
    st.divider()

    if env.state == EnvState.WAITING_HUMAN:
        agent = env._get_current_agent()
        st.subheader(f"🧑 {agent.display_name} 的视角（第{env.current_round}轮）")
        render_agent_perspective(env, agent)

        with st.expander("📜 查看完整讨论记录（上帝视角）"):
            render_history(env)

        st.info(f"轮到 **{agent.display_name}** 发言，请根据上方的讨论上下文输入你的观点。")
        with st.form("human_input_form"):
            user_input = st.text_area(
                f"{agent.display_name} 的发言",
                height=150,
                placeholder="请输入你的观点...",
            )
            submitted = st.form_submit_button("提交发言", type="primary")
            if submitted and user_input.strip():
                agent.submit_input(user_input.strip())
                env.step()
                with st.spinner("其他专家正在激烈讨论中..."):
                    auto_advance_llm(env)

                if env.state == EnvState.FINISHED:
                    st.session_state.pending_final_ranking = True

                st.rerun()
            elif submitted:
                st.warning("发言内容不能为空！")
    else:
        render_history(env)

        if env.state == EnvState.FINISHED:
            if st.session_state.pending_final_ranking:
                if render_final_ranking_form(env):
                    st.rerun()
                return

            if not st.session_state.discussion_over:
                log_path = env.save_log()
                st.session_state.discussion_over = True
                st.success(f"讨论结束！日志已保存至: {log_path}")
            else:
                st.success("讨论已结束。")

            if st.button("开始新讨论"):
                _reset_session()


if __name__ == "__main__":
    main()
