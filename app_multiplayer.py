#!/usr/bin/env python3
"""Brainstorm 局域网多人联机前端。

启动：streamlit run app_multiplayer.py --server.address 0.0.0.0

Human Evaluation 模式下，LLM Agent 从 llm_agents_pool 中自动盲抽，
用户仅需选择 LLM 数量，无需手动配置具体模型。
"""

import sys
import os
import uuid

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
from streamlit_autorefresh import st_autorefresh

from agents.agent_base import EnvState
from envs.brainwrite import BrainWrite
from prompts.topics import TOPICS
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


# ── 渲染辅助函数 ──

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
            with st.chat_message(name=agent.display_name, avatar="🧑"):
                st.markdown(f"**{color} {agent.display_name}** (你的历史发言)")
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
        domain_topics = TOPICS.get(topic_key, [])
        selected = st.selectbox(
            "选择话题", options=domain_topics if domain_topics else ["（暂无话题）"],
            key="create_topic_select",
        )
        topic = st.text_area(
            "话题内容",
            value=selected if domain_topics else "",
            height=80, key="create_topic",
        )

    num_humans = st.number_input(
        "人类参与者数量", min_value=2, max_value=4, value=2, key="create_num_humans",
    )

    pool = rm._load_pool()
    pool_size = len(pool)

    num_llm = st.number_input(
        f"LLM Agent 数量（配置池共 {pool_size} 个模型）",
        min_value=1, max_value=5, value=2, key="create_num_llm",
    )

    max_rounds = st.number_input(
        "讨论轮数", min_value=1, max_value=20, value=num_humans + num_llm,
        key="create_max_rounds",
    )

    if st.button("创建房间", type="primary", key="btn_create"):
        if not topic.strip():
            st.error("请输入讨论话题！")
            return
        if num_humans < 2:
            st.error("联机模式至少需要 2 个人类参与者！")
            return

        room_id = rm.create_room(
            mode=mode,
            topic=topic,
            max_rounds=max_rounds,
            num_humans=num_humans,
            num_llm=num_llm,
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
        agent = room.env.get_agent(aid)
        col1, col2 = st.columns([3, 1])
        with col1:
            st.write(f"**{agent.display_name}**")
            role_text = agent.role_background or ""
            if role_text:
                st.caption(role_text[:100] + ("..." if len(role_text) > 100 else ""))
        with col2:
            if st.button(f"认领 {agent.display_name}", key=f"claim_{aid}"):
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
        agent = room.env.get_agent(aid)
        if aid in room.claimed_seats:
            is_me = (aid == st.session_state.my_agent_id)
            tag = " (你)" if is_me else ""
            st.success(f"{agent.display_name}：已认领{tag}")
        else:
            st.warning(f"{agent.display_name}：等待中...")

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
    st.subheader(f"🧑 {my_agent.display_name} 的视角（第{env.current_round}轮）")
    render_agent_perspective(env, my_agent)

    with st.expander("📜 查看完整讨论记录（上帝视角）"):
        render_history(env)

    st.info(f"轮到 **{my_agent.display_name}** 发言，请根据上方的讨论上下文输入你的观点。")
    with st.form("human_input_form"):
        user_input = st.text_area(
            f"{my_agent.display_name} 的发言",
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
    st.warning(f"正在等待 {color} **{current_agent.display_name}** 思考并输入...")
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
            agent = env.get_agent(aid)
            if aid in room.rankings_submitted:
                st.success(f"{agent.display_name}：已提交")
            else:
                st.warning(f"{agent.display_name}：等待中...")


def _init_mp_ranking_state(others):
    """初始化联机版排名 session_state，每个 Agent 默认分配不同名次。"""
    if "mp_ranking_selections" not in st.session_state:
        st.session_state.mp_ranking_selections = {
            a.agent_id: idx + 1 for idx, a in enumerate(others)
        }


def _on_mp_rank_change(agent_id: int):
    """联机版排名下拉框回调：自动交换冲突名次。"""
    new_rank = st.session_state[f"rank_{agent_id}"]
    sel = st.session_state.mp_ranking_selections
    old_rank = sel.get(agent_id)

    for aid, r in sel.items():
        if aid != agent_id and r == new_rank:
            sel[aid] = old_rank
            break

    sel[agent_id] = new_rank


def _render_ranking_form(room, env, my_agent_id: int):
    others = [a for a in env.agents if a.agent_id != my_agent_id]
    room_id = st.session_state.room_id

    if not others:
        rm.submit_ranking(room_id, my_agent_id, [])
        st.rerun()
        return

    st.subheader("📊 最终排名 — 综合评价其他专家表现")
    st.info("请根据整场讨论中各位专家的综合表现进行排名（1 = 最佳）。名次会自动联动，确保不重复。")

    num_others = len(others)
    _init_mp_ranking_state(others)

    for agent in others:
        color = AGENT_COLORS[(agent.agent_id - 1) % len(AGENT_COLORS)]
        role_icon = "🧑" if agent.is_human else "🤖"
        current_rank = st.session_state.mp_ranking_selections[agent.agent_id]
        st.selectbox(
            f"{color} {role_icon} {agent.display_name} 的排名",
            options=list(range(1, num_others + 1)),
            index=current_rank - 1,
            key=f"rank_{agent.agent_id}",
            on_change=_on_mp_rank_change,
            args=(agent.agent_id,),
        )

    if st.button("提交排名", type="primary"):
        rankings = st.session_state.mp_ranking_selections
        rank_values = list(rankings.values())
        if len(set(rank_values)) != len(rank_values):
            st.error("排名不能重复！请为每位专家分配不同的名次。")
            return

        ranking_data = []
        for agent in others:
            ranking_data.append({
                "position": agent.agent_id,
                "config_key": agent.config_key,
                "rank": rankings[agent.agent_id],
            })
        ranking_data.sort(key=lambda x: x["rank"])

        rm.submit_ranking(room_id, my_agent_id, ranking_data)
        del st.session_state["mp_ranking_selections"]
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
