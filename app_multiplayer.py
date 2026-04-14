#!/usr/bin/env python3
"""Brainstorm 局域网多人联机前端。

启动：streamlit run app_multiplayer.py --server.address 0.0.0.0
"""

import sys
import os
import uuid

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
from streamlit_autorefresh import st_autorefresh

from agents.agent_base import EnvState
from envs.brainwrite import BrainWrite
from prompts.topics import TOPICS, EXPERTS
import room_manager as rm

MODE_LABELS = {
    "brainwrite": "脑力书写 (BrainWrite)",
    "round_robin": "轮流发言 (Round Robin)",
    "random": "随机发言 (Random)",
    "leader_worker": "领导-组员 (Leader-Worker)",
}

AGENT_COLORS = [
    "🔵", "🟢", "🟠", "🟣", "🔴", "🟡", "⚪", "🟤",
]


def _get_session_id() -> str:
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    return st.session_state.session_id


def init_session_state():
    defaults = {
        "room_id": None,
        "my_agent_id": None,
        "phase": "lobby",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _reset_session():
    for key in ("room_id", "my_agent_id", "phase"):
        if key in st.session_state:
            del st.session_state[key]
    st.rerun()


# ── 渲染辅助函数（从 app.py 复制核心逻辑，避免修改原文件） ──

def render_brainwrite_history(env):
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
            f"{color} 纸条路线 {origin_idx + 1}（由 {origin_agent.name} 发起）",
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
    messages = env.build_messages_for_agent(agent)
    color = AGENT_COLORS[(agent.agent_id - 1) % len(AGENT_COLORS)]
    for msg in messages:
        if msg["role"] == "system":
            continue
        elif msg["role"] == "assistant":
            with st.chat_message(name=agent.name, avatar="🧑"):
                st.markdown(f"**{color} {agent.name}** (你的历史发言)")
                st.markdown(msg["content"])
        elif msg["role"] == "user":
            with st.chat_message(name="讨论进展", avatar="📋"):
                st.markdown(msg["content"])


def render_status(env):
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


# ═══════════════════════════════════════════════════════════
# Phase 1: 大厅
# ═══════════════════════════════════════════════════════════

def render_lobby():
    st.header("大厅")
    tab_create, tab_join = st.tabs(["创建房间", "加入房间"])

    with tab_create:
        _render_create_room()

    with tab_join:
        _render_join_room()


def _render_create_room():
    st.subheader("创建新房间")

    mode = st.selectbox(
        "讨论形式",
        options=list(MODE_LABELS.keys()),
        format_func=lambda x: MODE_LABELS[x],
        key="create_mode",
    )

    topic_key = st.selectbox(
        "预设话题",
        options=["自定义"] + list(TOPICS.keys()),
        key="create_topic_key",
    )
    if topic_key == "自定义":
        topic = st.text_area("输入话题", value="", height=80, key="create_topic")
    else:
        topic = st.text_area("话题内容", value=TOPICS[topic_key], height=80, key="create_topic")

    num_agents = st.number_input(
        "Agent 总数", min_value=3, max_value=5, value=4, key="create_num_agents",
    )
    max_rounds = st.number_input(
        "讨论轮数", min_value=1, max_value=20, value=num_agents, key="create_max_rounds",
    )

    human_seats = st.multiselect(
        "人类参与者（选择哪些 Agent 由真人操控，至少 2 个）",
        options=list(range(1, num_agents + 1)),
        default=[1, 2] if num_agents >= 2 else [1],
        format_func=lambda x: f"Agent {x}",
        key="create_human_seats",
    )

    if mode == "leader_worker":
        leader_choices = st.multiselect(
            "Leader（从人类参与者中选择）",
            options=human_seats if human_seats else [1],
            default=[human_seats[0]] if human_seats else [1],
            format_func=lambda x: f"Agent {x}",
            key="create_leaders",
        )
    else:
        leader_choices = []

    agent_configs = []
    expert_keys = list(EXPERTS.keys())
    for i in range(num_agents):
        is_human = (i + 1) in human_seats
        label = f"Agent {i + 1} {'👤 人类' if is_human else '🤖 LLM'}"
        with st.expander(label, expanded=(i == 0)):
            name = st.text_input("名称", value=f"专家{i + 1}", key=f"cr_name_{i}")
            role_source = st.selectbox(
                "角色来源",
                options=["预设角色", "自定义"],
                key=f"cr_role_source_{i}",
            )
            if role_source == "预设角色":
                expert_choice = st.selectbox(
                    "选择专家",
                    options=expert_keys,
                    index=i % len(expert_keys),
                    key=f"cr_expert_{i}",
                )
                role = EXPERTS[expert_choice]
            else:
                role = st.text_area("角色背景", value="", key=f"cr_role_{i}", height=60)
            agent_configs.append({"name": name, "role": role})

    if st.button("创建房间", type="primary", key="btn_create"):
        if not topic.strip():
            st.error("请输入讨论话题！")
            return
        if len(human_seats) < 2:
            st.error("联机模式至少需要 2 个人类参与者！")
            return

        room_id = rm.create_room(
            mode=mode,
            topic=topic,
            max_rounds=max_rounds,
            agent_configs=agent_configs,
            human_seats=human_seats,
            leader_ids=leader_choices,
        )
        st.session_state.room_id = room_id
        st.session_state.phase = "claim_host"
        st.rerun()


def _render_join_room():
    st.subheader("加入已有房间")
    room_id = st.text_input("输入房间号（4 位数字）", max_chars=4, key="join_room_id")

    if st.button("加入", type="primary", key="btn_join"):
        room = rm.join_room(room_id)
        if room is None:
            st.error("房间不存在，请检查房间号！")
            return
        unclaimed = rm.get_unclaimed_seats(room_id)
        if not unclaimed:
            st.error("房间座位已满！")
            return
        st.session_state.room_id = room_id
        st.session_state.phase = "claim_guest"
        st.rerun()


# ═══════════════════════════════════════════════════════════
# Phase 1.5: 认领座位
# ═══════════════════════════════════════════════════════════

def render_claim_seat(is_host: bool):
    room_id = st.session_state.room_id
    room = rm.get_room(room_id)
    if room is None:
        st.error("房间已失效")
        _reset_session()
        return

    st.header(f"房间 {room_id}")
    st.info(f"讨论模式: {MODE_LABELS[room.mode]} | 话题: {room.topic}")

    unclaimed = rm.get_unclaimed_seats(room_id)

    if not unclaimed:
        st.warning("所有座位已被认领！")
        if st.button("返回大厅"):
            _reset_session()
        return

    st.subheader("选择你的座位")
    for aid in unclaimed:
        cfg = room.agent_configs[aid - 1]
        col1, col2 = st.columns([3, 1])
        with col1:
            st.write(f"**Agent {aid} — {cfg['name']}**")
            st.caption(cfg["role"][:100] + ("..." if len(cfg["role"]) > 100 else ""))
        with col2:
            if st.button(f"认领 Agent {aid}", key=f"claim_{aid}"):
                sid = _get_session_id()
                ok = rm.claim_seat(room_id, aid, sid)
                if ok:
                    st.session_state.my_agent_id = aid
                    st.session_state.phase = "waiting_players"
                    st.rerun()
                else:
                    st.error("认领失败，座位可能已被占！")
                    st.rerun()


# ═══════════════════════════════════════════════════════════
# Phase 2: 等待玩家
# ═══════════════════════════════════════════════════════════

def render_waiting_players():
    room_id = st.session_state.room_id
    room = rm.get_room(room_id)
    if room is None:
        st.error("房间已失效")
        _reset_session()
        return

    st_autorefresh(interval=3000, key="wait_refresh")

    st.header("等待玩家加入")
    st.markdown(f"### 房间号: `{room_id}`")
    st.info("请将房间号告知你的队友，等待他们加入...")

    st.subheader("座位状态")
    for aid in room.human_seats:
        cfg = room.agent_configs[aid - 1]
        if aid in room.claimed_seats:
            is_me = (aid == st.session_state.my_agent_id)
            tag = " (你)" if is_me else ""
            st.success(f"Agent {aid} — {cfg['name']}：已认领{tag}")
        else:
            st.warning(f"Agent {aid} — {cfg['name']}：等待中...")

    if rm.is_room_ready(room_id):
        st.balloons()
        if not room.initial_advance_done:
            with room.llm_lock:
                if not room.initial_advance_done:
                    rm.auto_advance_llm(room.env)
                    room.initial_advance_done = True
        st.session_state.phase = "discussion"
        st.rerun()


# ═══════════════════════════════════════════════════════════
# Phase 3: 讨论
# ═══════════════════════════════════════════════════════════

def render_discussion():
    room_id = st.session_state.room_id
    room = rm.get_room(room_id)
    if room is None:
        st.error("房间已失效")
        _reset_session()
        return

    env = room.env
    my_agent_id = st.session_state.my_agent_id
    my_agent = env.get_agent(my_agent_id)

    render_status(env)
    st.divider()

    if env.state == EnvState.FINISHED:
        st.session_state.phase = "ranking"
        st.rerun()
        return

    if env.state == EnvState.WAITING_HUMAN:
        current_agent = env._get_current_agent()
        if current_agent.agent_id == my_agent_id:
            _render_my_turn(room, env, my_agent)
        else:
            _render_waiting_turn(env, current_agent)
    elif env.state == EnvState.WAITING_LLM:
        st_autorefresh(interval=2000, key="llm_wait_refresh")
        st.info("AI 正在思考中...")
        render_history(env)
    else:
        render_history(env)


def _render_my_turn(room, env, my_agent):
    st.subheader(f"🧑 {my_agent.name} 的视角（第{env.current_round}轮）")
    render_agent_perspective(env, my_agent)

    with st.expander("📜 查看完整讨论记录（上帝视角）"):
        render_history(env)

    st.info(f"轮到 **{my_agent.name}** 发言，请根据上方的讨论上下文输入你的观点。")
    with st.form("human_input_form"):
        user_input = st.text_area(
            f"{my_agent.name} 的发言",
            height=150,
            placeholder="请输入你的观点...",
        )
        submitted = st.form_submit_button("提交发言", type="primary")
        if submitted and user_input.strip():
            with room.llm_lock:
                my_agent.submit_input(user_input.strip())
                env.step()
                rm.auto_advance_llm(env)
            if env.state == EnvState.FINISHED:
                st.session_state.phase = "ranking"
            st.rerun()
        elif submitted:
            st.warning("发言内容不能为空！")


def _render_waiting_turn(env, current_agent):
    st_autorefresh(interval=3000, key="turn_wait_refresh")
    color = AGENT_COLORS[(current_agent.agent_id - 1) % len(AGENT_COLORS)]
    st.warning(f"正在等待 {color} **{current_agent.name}** 思考并输入...")
    with st.expander("📜 查看当前讨论记录", expanded=True):
        render_history(env)


# ═══════════════════════════════════════════════════════════
# Phase 4: 排名
# ═══════════════════════════════════════════════════════════

def render_ranking():
    room_id = st.session_state.room_id
    room = rm.get_room(room_id)
    if room is None:
        st.error("房间已失效")
        _reset_session()
        return

    env = room.env
    my_agent_id = st.session_state.my_agent_id

    st.header("讨论结束")
    with st.expander("📜 查看完整讨论记录", expanded=False):
        render_history(env)

    already_submitted = my_agent_id in room.rankings_submitted

    if not already_submitted:
        _render_ranking_form(room, env, my_agent_id)
    else:
        st.success("你已提交排名，等待其他玩家...")

    if rm.all_rankings_submitted(room_id):
        st.session_state.phase = "done"
        st.rerun()
    elif already_submitted:
        st_autorefresh(interval=3000, key="rank_wait_refresh")
        st.subheader("排名提交状态")
        for aid in room.human_seats:
            cfg = room.agent_configs[aid - 1]
            if aid in room.rankings_submitted:
                st.success(f"Agent {aid} — {cfg['name']}：已提交")
            else:
                st.warning(f"Agent {aid} — {cfg['name']}：等待中...")


def _render_ranking_form(room, env, my_agent_id: int):
    others = [a for a in env.agents if a.agent_id != my_agent_id]
    if not others:
        rm.submit_ranking(room.env, my_agent_id, [])
        st.rerun()
        return

    st.subheader("📊 最终排名 — 综合评价其他专家表现")
    st.info("请根据整场讨论中各位专家的综合表现进行排名（1 = 最佳）")

    num_others = len(others)
    room_id = st.session_state.room_id

    with st.form("ranking_form"):
        rankings = {}
        for idx, agent in enumerate(others):
            color = AGENT_COLORS[(agent.agent_id - 1) % len(AGENT_COLORS)]
            role_icon = "🧑" if agent.is_human else "🤖"
            rank = st.selectbox(
                f"{color} {role_icon} {agent.name} 的排名",
                options=list(range(1, num_others + 1)),
                index=idx,
                key=f"rank_{agent.agent_id}",
            )
            rankings[agent.agent_id] = rank

        submitted = st.form_submit_button("提交排名", type="primary")
        if submitted:
            rank_values = list(rankings.values())
            if len(set(rank_values)) != len(rank_values):
                st.error("排名不能重复！请为每位专家分配不同的名次。")
                return

            ranking_data = []
            for agent in others:
                ranking_data.append({
                    "agent_id": agent.agent_id,
                    "agent_name": agent.name,
                    "rank": rankings[agent.agent_id],
                })
            ranking_data.sort(key=lambda x: x["rank"])

            rm.submit_ranking(room_id, my_agent_id, ranking_data)
            st.rerun()


# ═══════════════════════════════════════════════════════════
# Phase 5: 完成
# ═══════════════════════════════════════════════════════════

def render_done():
    room_id = st.session_state.room_id
    room = rm.get_room(room_id)
    if room is None:
        st.error("房间已失效")
        _reset_session()
        return

    env = room.env

    log_path = rm.save_and_get_log(room_id)
    st.success(f"讨论结束！日志已保存至: {log_path}")

    with st.expander("📜 查看完整讨论记录", expanded=False):
        render_history(env)

    if st.button("开始新讨论"):
        _reset_session()


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════

def main():
    st.set_page_config(page_title="Brainstorm 联机版", page_icon="🌐", layout="wide")
    st.title("🌐 Brainstorm 头脑风暴 — 联机版")

    init_session_state()

    if st.sidebar.button("🔄 返回大厅", use_container_width=True):
        _reset_session()

    phase = st.session_state.phase

    if phase == "lobby":
        render_lobby()
    elif phase == "claim_host":
        render_claim_seat(is_host=True)
    elif phase == "claim_guest":
        render_claim_seat(is_host=False)
    elif phase == "waiting_players":
        render_waiting_players()
    elif phase == "discussion":
        render_discussion()
    elif phase == "ranking":
        render_ranking()
    elif phase == "done":
        render_done()
    else:
        st.error(f"未知阶段: {phase}")
        _reset_session()


if __name__ == "__main__":
    main()
