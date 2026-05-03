#!/usr/bin/env python3
"""Experiment 2 Streamlit entrypoint: 1 human + 3 LLM round robin."""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st

from agents.agent_base import EnvState
from agents.agent_human import AgentHuman
from envs.round_robin import RoundRobin
from prompts.topics import TOPICS
from tools.call_openai import call_openai
from tools.config_loader import build_agent_from_config, load_llm_config


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
LOG_ROOT = os.path.join(PROJECT_ROOT, "log_ex2_1human3LLM")
ELO_FILE = os.path.join(LOG_ROOT, "elo_scores.json")
TRANSLATION_CONFIG_PATH = os.path.join(PROJECT_ROOT, "translation_config", "config.json")
MODE = "round_robin"
NUM_LLM = 3
NUM_HUMAN = 1
MAX_ROUNDS = NUM_LLM + NUM_HUMAN
INITIAL_ELO = 1200.0
K_FACTOR = 32.0
PRE_STAGE_MIN_APPEARANCES = 3
COUNT_RANGE_LIMIT = 3

AGENT_COLORS = [
    "🔵", "🟢", "🟠", "🟣", "🔴", "🟡", "⚪", "🟤",
]


def _all_topics() -> list[str]:
    return [topic for topic_list in TOPICS.values() for topic in topic_list]


def _topic_slug(topic: str, max_len: int = 80) -> str:
    cleaned = re.sub(r"[^\w\u4e00-\u9fff]+", "_", topic).strip("_")
    return cleaned[:max_len] or "topic"


def _safe_username(username: str) -> str:
    cleaned = re.sub(r"[^\w\u4e00-\u9fff-]+", "_", username.strip()).strip("_")
    return cleaned or "anonymous"


def _read_json(path: str, default):
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def _write_json_atomic(path: str, data: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def init_session_state() -> None:
    defaults = {
        "env": None,
        "started": False,
        "discussion_over": False,
        "pending_final_ranking": False,
        "selected_topic": None,
        "selected_model_keys": [],
        "translation_cache": {},
        "sidebar_translation": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def render_login() -> str | None:
    st.sidebar.header("用户登录")
    username = st.sidebar.text_input(
        "Username",
        value=st.session_state.get("username", ""),
        key="username",
        placeholder="请输入用户名后开始实验",
    ).strip()
    if not username:
        st.warning("请输入 Username 后再开始实验。")
        return None
    return username


def _reset_session() -> None:
    for key in (
        "started",
        "discussion_over",
        "pending_final_ranking",
        "selected_topic",
        "selected_model_keys",
        "ranking_selections",
    ):
        if key in st.session_state:
            del st.session_state[key]
    st.session_state["env"] = None
    init_session_state()
    st.rerun()


def _load_model_pool() -> dict:
    if "llm_pool" not in st.session_state:
        try:
            st.session_state.llm_pool = load_llm_config()
        except FileNotFoundError:
            st.session_state.llm_pool = {}
    return st.session_state.llm_pool


def _load_translation_config() -> dict:
    if not os.path.exists(TRANSLATION_CONFIG_PATH):
        raise FileNotFoundError(f"未找到翻译配置文件：{TRANSLATION_CONFIG_PATH}")
    with open(TRANSLATION_CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _topic_dir(topic: str) -> str:
    return os.path.join(LOG_ROOT, _topic_slug(topic))


def _user_stats_path(topic: str, username: str) -> str:
    return os.path.join(_topic_dir(topic), f"user_stats_{_safe_username(username)}.json")


def _count_topic_logs(topic: str) -> int:
    topic_dir = _topic_dir(topic)
    if not os.path.isdir(topic_dir):
        return 0
    total = 0
    for filename in os.listdir(topic_dir):
        if not filename.endswith(".json"):
            continue
        if filename == "elo_scores.json" or filename.startswith("user_stats_"):
            continue
        total += 1
    return total


def _user_has_discussed_topic(topic: str, username: str) -> bool:
    stats = _read_json(_user_stats_path(topic, username), {})
    return bool(stats.get("runs"))


def select_topic_for_user(username: str) -> tuple[str, dict[str, int], set[str]]:
    topics = _all_topics()
    counts = {topic: _count_topic_logs(topic) for topic in topics}
    discussed = {topic for topic in topics if _user_has_discussed_topic(topic, username)}
    candidates = [topic for topic in topics if topic not in discussed] or topics
    min_count = min(counts[topic] for topic in candidates)
    tied = [topic for topic in candidates if counts[topic] == min_count]
    return random.choice(tied), counts, discussed


def load_elo_scores(pool: dict) -> dict:
    data = _read_json(ELO_FILE, {})
    models = data.setdefault("models", {})
    for model_key in pool:
        record = models.setdefault(model_key, {})
        record.setdefault("elo", INITIAL_ELO)
        record.setdefault("appearances", 0)
    for stale_key in list(models):
        if stale_key not in pool:
            models.pop(stale_key)
    data.setdefault("history_combinations", [])
    data["updated_at"] = datetime.now().isoformat()
    _write_json_atomic(ELO_FILE, data)
    return data


def _combo_hash(model_keys: list[str]) -> str:
    return hashlib.sha256("|".join(sorted(model_keys)).encode("utf-8")).hexdigest()


def _history_hashes(elo_data: dict) -> set[str]:
    hashes = set()
    for combo in elo_data.get("history_combinations", []):
        if isinstance(combo, str):
            hashes.add(combo)
        elif isinstance(combo, list):
            hashes.add(_combo_hash(combo))
    return hashes


def _sample_pre_stage(models: dict, history_hashes: set[str]) -> list[str]:
    keys = list(models.keys())
    best_combo: list[str] | None = None
    best_signature: tuple | None = None
    for _ in range(300):
        picked: list[str] = []
        remaining = keys[:]
        while len(picked) < NUM_LLM and remaining:
            min_appearances = min(models[key]["appearances"] for key in remaining)
            tied = [key for key in remaining if models[key]["appearances"] == min_appearances]
            chosen = random.choice(tied)
            picked.append(chosen)
            remaining.remove(chosen)
        signature = (
            _combo_hash(picked) in history_hashes,
            max(models[key]["appearances"] for key in picked),
            sum(models[key]["appearances"] for key in picked),
        )
        if best_signature is None or signature < best_signature:
            best_combo = picked
            best_signature = signature
        if signature[0] is False:
            return picked
    return best_combo or random.sample(keys, NUM_LLM)


def _sample_stable_stage(models: dict) -> list[str]:
    ordered = sorted(models, key=lambda key: (-models[key]["elo"], models[key]["appearances"], key))
    windows: list[tuple[float, int, list[str]]] = []
    for start in range(0, len(ordered) - NUM_LLM + 1):
        combo = ordered[start:start + NUM_LLM]
        appearances = [models[key]["appearances"] for key in combo]
        count_range = max(appearances) - min(appearances)
        if count_range <= COUNT_RANGE_LIMIT:
            elo_spread = max(models[key]["elo"] for key in combo) - min(models[key]["elo"] for key in combo)
            windows.append((elo_spread, count_range, combo))
    if windows:
        min_signature = min((spread, count_range) for spread, count_range, _ in windows)
        tied = [combo for spread, count_range, combo in windows if (spread, count_range) == min_signature]
        return random.choice(tied)

    least_seen = sorted(models, key=lambda key: (models[key]["appearances"], -models[key]["elo"], key))
    return least_seen[:NUM_LLM]


def sample_llm_keys(pool: dict) -> list[str]:
    if len(pool) < NUM_LLM:
        raise ValueError(f"模型池只有 {len(pool)} 个模型，无法抽取 {NUM_LLM} 个 LLM。")
    elo_data = load_elo_scores(pool)
    models = elo_data["models"]
    if any(record["appearances"] < PRE_STAGE_MIN_APPEARANCES for record in models.values()):
        sampled = _sample_pre_stage(models, _history_hashes(elo_data))
    else:
        sampled = _sample_stable_stage(models)
    elo_data["history_combinations"].append(sampled)
    elo_data["history_combinations"] = elo_data["history_combinations"][-500:]
    elo_data["updated_at"] = datetime.now().isoformat()
    _write_json_atomic(ELO_FILE, elo_data)
    return sampled


def build_agents(model_keys: list[str], pool: dict) -> list:
    agents = [AgentHuman()]
    agents.extend(build_agent_from_config(config_key, pool) for config_key in model_keys)
    return agents


def auto_advance_llm(env) -> None:
    while env.state not in (EnvState.WAITING_HUMAN, EnvState.FINISHED):
        env.step()


def llm_display_map(env) -> dict[int, str]:
    llm_agents = [agent for agent in env.agents if not agent.is_human]
    return {agent.agent_id: f"Agent {idx + 1}" for idx, agent in enumerate(llm_agents)}


def display_label(env, agent_id: int) -> str:
    agent = env.get_agent(agent_id)
    if agent.is_human:
        return "Human"
    return llm_display_map(env).get(agent_id, f"Agent {agent_id}")


def mask_agent_names(env, text: str) -> str:
    masked = text
    placeholders: dict[str, str] = {}
    for agent in env.agents:
        placeholder = f"__EX2_AGENT_{agent.agent_id}__"
        placeholders[placeholder] = display_label(env, agent.agent_id)
        masked = masked.replace(agent.display_name, placeholder)
    for placeholder, label in placeholders.items():
        masked = masked.replace(placeholder, label)
    return masked


def translate_text(
    text: str,
    topic: str,
    direction: str,
    cache_prefix: str,
) -> str:
    if not text.strip():
        return ""
    cache_key = hashlib.sha256(
        f"{cache_prefix}|{direction}|{topic}|{text}".encode("utf-8")
    ).hexdigest()
    cache = st.session_state.translation_cache
    if cache_key in cache:
        return cache[cache_key]

    try:
        cfg = _load_translation_config()
        api_config = {"api_key": cfg["api_key"], "base_url": cfg["base_url"]}
        inference_config = {
            "model": cfg["model"],
            "temperature": cfg.get("temperature", 0.2),
            "is_reasoning": False,
        }
        if cfg.get("top_p") is not None:
            inference_config["top_p"] = cfg["top_p"]
        if cfg.get("max_tokens") is not None:
            inference_config["max_tokens"] = cfg["max_tokens"]

        if direction == "en_to_zh":
            system_prompt = cfg.get(
                "system_prompt",
                "你是一个专业的辅助翻译系统。请将用户提供的文本翻译成中文。保持专业、流畅，不改变原意。请直接输出翻译结果，不要包含任何多余的解释或对话语。",
            )
            user_prompt = (
                f"当前正在进行对应 Topic 的讨论，话题是：[{topic}]。\n"
                "请结合此语境将以下英文发言翻译为中文，保持专业、流畅，不改变原意。"
                "请直接输出翻译结果，不要包含解释。\n\n"
                f"{text}"
            )
        elif direction == "zh_to_en":
            system_prompt = (
                "你是一个专业的辅助翻译系统。请将用户提供的中文文本翻译成自然、清晰、适合头脑风暴讨论的英文。"
                "保持原意，不添加新观点。请直接输出英文翻译结果，不要包含任何多余解释。"
            )
            user_prompt = (
                f"当前正在进行对应 Topic 的讨论，话题是：[{topic}]，"
                "请结合此语境将以下用户的中文发言翻译为英文：\n\n"
                f"{text}"
            )
        else:
            raise ValueError(f"未知翻译方向：{direction}")

        translation = call_openai(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            api_config,
            inference_config,
        )
    except Exception as exc:
        translation = f"（翻译失败：{type(exc).__name__}: {exc}）"
    cache[cache_key] = translation
    return translation


def render_agent_content(content: str, topic: str) -> None:
    st.markdown("**原文**")
    st.markdown(content)
    with st.expander("中文翻译", expanded=False):
        st.markdown(translate_text(content, topic, "en_to_zh", cache_prefix="agent"))


def render_sidebar_translation(topic: str) -> None:
    st.sidebar.subheader("中文发言辅助翻译")
    source = st.sidebar.text_area(
        "输入中文发言",
        key="sidebar_translation_source",
        height=120,
        placeholder="输入中文后点击获取英文翻译",
    )
    if st.sidebar.button("获取英文翻译", use_container_width=True):
        if source.strip():
            st.session_state.sidebar_translation = translate_text(
                source.strip(),
                topic,
                "zh_to_en",
                cache_prefix="human_sidebar",
            )
        else:
            st.session_state.sidebar_translation = "请输入需要翻译的中文内容。"
    if st.session_state.sidebar_translation:
        st.sidebar.text_area(
            "英文翻译结果",
            value=st.session_state.sidebar_translation,
            height=140,
            key="sidebar_translation_result",
        )


def render_history(env, pool: dict) -> None:
    for entry in env.global_history:
        agent = env.get_agent(entry["agent_id"])
        role_icon = "🧑" if agent.is_human else "🤖"
        entry_color = AGENT_COLORS[(entry["agent_id"] - 1) % len(AGENT_COLORS)]
        name = display_label(env, entry["agent_id"])
        with st.chat_message(name=name, avatar=role_icon):
            st.markdown(f"**{entry_color} {name}** (第{entry['round']}轮)")
            if agent.is_human:
                st.markdown(entry["content"])
            else:
                render_agent_content(entry["content"], env.topic)


def render_agent_perspective(env, agent, pool: dict) -> None:
    messages = env.build_messages_for_agent(agent)
    color = AGENT_COLORS[(agent.agent_id - 1) % len(AGENT_COLORS)]

    for msg in messages:
        if msg["role"] == "system":
            continue
        if msg["role"] == "assistant":
            with st.chat_message(name="Human", avatar="🧑"):
                st.markdown(f"**{color} Human** (你的历史发言)")
                st.markdown(msg["content"])
        elif msg["role"] == "user":
            with st.chat_message(name="讨论进展", avatar="📋"):
                st.markdown(mask_agent_names(env, msg["content"]))

    with st.expander("Agent 发言翻译视图", expanded=False):
        for entry in env.global_history:
            if env.get_agent(entry["agent_id"]).is_human:
                continue
            st.markdown(f"**{display_label(env, entry['agent_id'])}** (第{entry['round']}轮)")
            render_agent_content(entry["content"], env.topic)
            st.divider()


def render_status(env, username: str) -> None:
    col1, col2, col3, col4 = st.columns(4)
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
    with col4:
        st.metric("Username", username)


def _init_ranking_state(others) -> None:
    if "ranking_selections" not in st.session_state:
        st.session_state.ranking_selections = {
            agent.agent_id: idx + 1 for idx, agent in enumerate(others)
        }


def _on_rank_change(agent_id: int) -> None:
    new_rank = st.session_state[f"final_rank_{agent_id}"]
    old_selections = st.session_state.ranking_selections
    old_rank = old_selections.get(agent_id)

    for other_id, rank in old_selections.items():
        if other_id != agent_id and rank == new_rank:
            old_selections[other_id] = old_rank
            st.session_state[f"final_rank_{other_id}"] = old_rank
            break

    old_selections[agent_id] = new_rank


def update_elo_scores(ranking_data: list[dict], pool: dict) -> None:
    elo_data = load_elo_scores(pool)
    models = elo_data["models"]
    ranked_keys = [item["config_key"] for item in ranking_data]
    old_scores = {key: float(models[key]["elo"]) for key in ranked_keys}
    rank_by_key = {item["config_key"]: item["rank"] for item in ranking_data}

    for model_i in ranked_keys:
        delta_sum = 0.0
        for model_j in ranked_keys:
            if model_i == model_j:
                continue
            expected = 1.0 / (1.0 + 10 ** ((old_scores[model_j] - old_scores[model_i]) / 400.0))
            actual = 1.0 if rank_by_key[model_i] < rank_by_key[model_j] else 0.0
            delta_sum += actual - expected
        models[model_i]["elo"] = old_scores[model_i] + (K_FACTOR / 2.0) * delta_sum
        models[model_i]["appearances"] = int(models[model_i].get("appearances", 0)) + 1

    elo_data["updated_at"] = datetime.now().isoformat()
    _write_json_atomic(ELO_FILE, elo_data)


def render_final_ranking_form(env, pool: dict) -> bool:
    others = [agent for agent in env.agents if not agent.is_human]
    if not others:
        return False

    st.subheader("最终排名：综合评价其他专家表现")
    st.info("请根据整场讨论中各位专家的综合表现进行排名（1 = 最佳）。名次会自动联动，且不允许并列。")

    num_others = len(others)
    _init_ranking_state(others)

    for agent in others:
        color = AGENT_COLORS[(agent.agent_id - 1) % len(AGENT_COLORS)]
        current_rank = st.session_state.ranking_selections[agent.agent_id]
        st.selectbox(
            f"{color} {display_label(env, agent.agent_id)} 的排名",
            options=list(range(1, num_others + 1)),
            index=current_rank - 1,
            key=f"final_rank_{agent.agent_id}",
            on_change=_on_rank_change,
            args=(agent.agent_id,),
        )

    if st.button("提交排名", type="primary"):
        rankings = st.session_state.ranking_selections
        rank_values = list(rankings.values())
        expected = set(range(1, num_others + 1))
        if set(rank_values) != expected or len(rank_values) != num_others:
            st.error(f"排名无效！请为每位专家分配从 1 到 {num_others} 的不重复名次，不允许并列排名。")
            return False

        ranking_data = []
        for agent in others:
            ranking_data.append({
                "position": agent.agent_id,
                "display_name": display_label(env, agent.agent_id),
                "config_key": agent.config_key,
                "rank": rankings[agent.agent_id],
            })
        ranking_data.sort(key=lambda item: item["rank"])

        env.final_rankings = ranking_data
        update_elo_scores(ranking_data, pool)
        st.session_state.pending_final_ranking = False
        del st.session_state["ranking_selections"]
        return True

    return False


def save_ex2_log(env, username: str, model_keys: list[str]) -> str:
    topic_dir = _topic_dir(env.topic)
    log_path = env.save_log(log_dir=topic_dir)
    log_data = _read_json(log_path, {})
    log_data["user_name"] = username
    log_data.setdefault("metadata", {})["user_name"] = username
    log_data["metadata"]["masked_agent_names"] = llm_display_map(env)
    _write_json_atomic(log_path, log_data)

    stats_path = _user_stats_path(env.topic, username)
    stats = _read_json(
        stats_path,
        {
            "user_name": username,
            "topic": env.topic,
            "runs": [],
        },
    )
    stats["runs"].append({
        "timestamp": datetime.now().isoformat(),
        "agent_combination": model_keys,
        "max_rounds": env.max_rounds,
        "completed_round": min(env.current_round, env.max_rounds),
        "log_path": log_path,
    })
    _write_json_atomic(stats_path, stats)
    return log_path


def render_sidebar(username: str, pool: dict) -> str:
    if st.sidebar.button("重新开始", use_container_width=True):
        _reset_session()

    topic, counts, discussed = select_topic_for_user(username)
    if st.session_state.selected_topic is None:
        st.session_state.selected_topic = topic
    selected_topic = st.session_state.selected_topic

    st.sidebar.header("实验配置")
    st.sidebar.info(
        f"固定模式：{MODE}\n\n"
        f"参与者：{NUM_HUMAN} Human + {NUM_LLM} LLM Agents\n\n"
        f"讨论轮数：{MAX_ROUNDS}"
    )
    st.sidebar.subheader("本次抽取 Topic")
    st.sidebar.markdown(selected_topic)
    st.sidebar.caption(
        f"该 Topic 全局次数：{counts.get(selected_topic, 0)}；"
        f"你已讨论 Topic 数：{len(discussed)} / {len(_all_topics())}"
    )
    render_sidebar_translation(selected_topic)
    return selected_topic


def main() -> None:
    st.set_page_config(page_title="Brainstorm EX2", page_icon="🧠", layout="wide")
    st.title("Brainstorm Experiment 2：1 Human + 3 LLM RoundRobin")

    username = render_login()
    if username is None:
        return

    init_session_state()
    pool = _load_model_pool()
    topic = render_sidebar(username, pool)

    if not st.session_state.started:
        st.info("系统已按全局统计和个人历史自动抽取 Topic。点击下方按钮开始讨论。")
        if st.button("开始讨论", type="primary"):
            if not pool:
                st.error("模型池为空，无法开始实验。")
                return
            try:
                model_keys = sample_llm_keys(pool)
                agents = build_agents(model_keys, pool)
            except Exception as exc:
                st.error(f"初始化失败：{type(exc).__name__}: {exc}")
                return

            env = RoundRobin(
                agents=agents,
                topic=topic,
                max_rounds=MAX_ROUNDS,
                log_dir=_topic_dir(topic),
            )
            env.init()
            st.session_state.env = env
            st.session_state.selected_model_keys = model_keys
            st.session_state.started = True
            st.session_state.discussion_over = False
            st.session_state.pending_final_ranking = False

            with st.spinner("专家们正在讨论中..."):
                auto_advance_llm(env)
            st.rerun()
        return

    env = st.session_state.env
    render_status(env, username)
    st.divider()

    if env.state == EnvState.WAITING_HUMAN:
        agent = env._get_current_agent()
        st.subheader(f"Human 的视角（第{env.current_round}轮）")
        render_agent_perspective(env, agent, pool)

        with st.expander("查看完整讨论记录（上帝视角）"):
            render_history(env, pool)

        st.info("轮到 Human 发言，请根据上方的讨论上下文输入你的观点。")
        with st.form("human_input_form", clear_on_submit=True):
            user_input = st.text_area(
                "Human 的发言",
                height=150,
                placeholder="请输入你的英文观点；可先在侧边栏将中文翻译为英文。",
            )
            submitted = st.form_submit_button("提交发言", type="primary")
            if submitted and user_input.strip():
                agent.submit_input(user_input.strip())
                env.step()
                with st.spinner("其他专家正在讨论中..."):
                    auto_advance_llm(env)

                if env.state == EnvState.FINISHED:
                    st.session_state.pending_final_ranking = True

                st.rerun()
            elif submitted:
                st.warning("发言内容不能为空！")
    else:
        render_history(env, pool)

        if env.state == EnvState.FINISHED:
            if st.session_state.pending_final_ranking:
                if render_final_ranking_form(env, pool):
                    st.rerun()
                return

            if not st.session_state.discussion_over:
                log_path = save_ex2_log(env, username, st.session_state.selected_model_keys)
                st.session_state.discussion_over = True
                st.success(f"讨论结束！日志已保存至: {log_path}")
            else:
                st.success("讨论已结束。")

            if st.button("开始新讨论"):
                _reset_session()


if __name__ == "__main__":
    main()
