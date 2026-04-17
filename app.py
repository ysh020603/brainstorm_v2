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
from prompts.topics import TOPICS
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
    if "topics_pool" not in st.session_state:
        import copy
        st.session_state.topics_pool = copy.deepcopy(TOPICS)


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

    topics_pool = st.session_state.topics_pool
    domain_keys = list(topics_pool.keys())

    if st.sidebar.button("🎲 随机抽取话题"):
        all_topics = [t for ts in topics_pool.values() for t in ts]
        if all_topics:
            st.session_state["random_topic"] = random.choice(all_topics)
        else:
            st.sidebar.warning("话题池为空，无法抽取！")

    topic_key = st.sidebar.selectbox(
        "预设话题",
        options=["自定义"] + domain_keys,
    )
    if topic_key == "自定义":
        default_topic = st.session_state.get("random_topic", "")
        topic = st.sidebar.text_area("输入话题", value=default_topic, height=80)
    else:
        domain_topics = topics_pool.get(topic_key, [])
        selected_topic = st.sidebar.selectbox(
            "选择话题",
            options=domain_topics if domain_topics else ["（该领域暂无话题）"],
            key="topic_select_in_domain",
        )
        topic = st.sidebar.text_area(
            "话题内容",
            value=selected_topic if domain_topics else "",
            height=80,
        )

    with st.sidebar.expander("➕ 添加新话题"):
        new_domain = st.text_input("领域名称", key="new_topic_domain")
        new_topic_text = st.text_area("话题内容", key="new_topic_text", height=60)
        if st.button("添加", key="btn_add_topic"):
            if new_topic_text.strip():
                domain = new_domain.strip() if new_domain.strip() else "未分类"
                if domain not in st.session_state.topics_pool:
                    st.session_state.topics_pool[domain] = []
                st.session_state.topics_pool[domain].append(new_topic_text.strip())
                st.rerun()
            else:
                st.warning("话题内容不能为空！")

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

    return mode, topic, max_rounds, num_llm


# ── 构建 Agent 列表 ──

def build_agents(num_llm: int, pool: dict) -> list:
    """构建 Agent 列表：1 个人类 + num_llm 个自动抽取的 LLM。

    agent_id 不在此处设置，由 EnvBase 构造函数根据 shuffle 后的顺序动态分配。
    LLM 的角色严格由外部配置文件决定，不再兜底分配。
    """
    agents = []

    agents.append(AgentHuman())

    sampled_keys = sample_llm_keys(pool, num_llm)

    for config_key in sampled_keys:
        agent = build_agent_from_config(config_key, pool)
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

def _init_ranking_state(others):
    """初始化排名 session_state，每个 Agent 默认分配不同名次。"""
    if "ranking_selections" not in st.session_state:
        st.session_state.ranking_selections = {
            a.agent_id: idx + 1 for idx, a in enumerate(others)
        }


def _on_rank_change(agent_id: int, num_others: int, all_agent_ids: list[int]):
    """排名下拉框回调：当用户更改某个 Agent 的排名时，自动交换冲突名次。

    同时同步更新被交换 Agent 对应的 widget key，确保 UI 联动生效。
    """
    new_rank = st.session_state[f"final_rank_{agent_id}"]
    old_selections = st.session_state.ranking_selections
    old_rank = old_selections.get(agent_id)

    for aid, r in old_selections.items():
        if aid != agent_id and r == new_rank:
            old_selections[aid] = old_rank
            st.session_state[f"final_rank_{aid}"] = old_rank
            break

    old_selections[agent_id] = new_rank


def render_final_ranking_form(env):
    """讨论结束后，人类对所有其他 Agent 进行一次总排名。

    通过联动机制确保排名互不重复：当一个名次被选中后，
    持有该名次的其他 Agent 会自动交换到原名次。

    Returns True 表示排名已提交，False 表示尚未提交或校验失败。
    """
    others = [a for a in env.agents if not a.is_human]
    if not others:
        return False

    st.subheader("📊 最终排名 — 综合评价其他专家表现")
    st.info("请根据整场讨论中各位专家的综合表现进行排名（1 = 最佳）。名次会自动联动，确保不重复。")

    num_others = len(others)
    all_agent_ids = [a.agent_id for a in others]
    _init_ranking_state(others)

    for agent in others:
        color = AGENT_COLORS[(agent.agent_id - 1) % len(AGENT_COLORS)]
        current_rank = st.session_state.ranking_selections[agent.agent_id]
        st.selectbox(
            f"{color} {agent.display_name} 的排名",
            options=list(range(1, num_others + 1)),
            index=current_rank - 1,
            key=f"final_rank_{agent.agent_id}",
            on_change=_on_rank_change,
            args=(agent.agent_id, num_others, all_agent_ids),
        )

    if st.button("提交排名", type="primary"):
        rankings = st.session_state.ranking_selections
        rank_values = list(rankings.values())
        expected = set(range(1, num_others + 1))
        if set(rank_values) != expected or len(rank_values) != num_others:
            st.error(
                f"排名无效！请为每位专家分配从 1 到 {num_others} 的不重复名次，"
                "不允许并列排名。请重新调整后再提交。"
            )
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
        del st.session_state["ranking_selections"]
        return True

    return False


def main():
    st.set_page_config(page_title="Brainstorm 头脑风暴", page_icon="🧠", layout="wide")
    st.title("🧠 Brainstorm 头脑风暴系统")

    init_session_state()
    mode, topic, max_rounds, num_llm = render_sidebar()

    # ── 尚未开始：等待用户点击"开始讨论" ──
    if not st.session_state.started:
        st.info("请在左侧配置讨论参数，然后点击下方按钮开始讨论。")
        if st.button("开始讨论", type="primary"):
            if not topic.strip():
                st.error("请输入讨论话题！")
                return

            pool = _load_model_pool()
            agents = build_agents(num_llm, pool)
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
        with st.form("human_input_form", clear_on_submit=True):
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
