"""
人类 Agent-Level CPSS 评估系统 — Streamlit 独立入口

启动方式：
    streamlit run human_cpss_ui/app_human_cpss.py

功能：
    - 用户登录与独立的历史标注追踪（与全局排序系统物理隔离）
    - 实验日志脱敏展示（Agent 身份随机重映射）
    - 按 Agent 独立打分（CPSS 55 维双极语义量表，1-7 分）
    - 标注结果写回原 JSON 的 `human_eval_per_agent_<UserName>` 字段
    - 文件锁保证并发安全
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import streamlit as st
from filelock import FileLock

# ═══════════════════════════════════════════════════════════════
# 后台配置项
# ═══════════════════════════════════════════════════════════════

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

# 待标注日志目录列表（绝对路径或相对于 PROJECT_ROOT 的相对路径）
# 系统会递归扫描这些目录下所有 *.json 文件作为「总任务池」。
TARGET_EVAL_DIRS: list[str] = [
    "log",
]

# 专属用户记录空间，与全局排序系统的 user_log/ 物理隔离
USER_LOG_DIR = "human_user_log"

MODE_LABELS: dict[str, str] = {
    "brainwrite": "脑力书写 (BrainWrite)",
    "round_robin": "轮流发言 (Round Robin)",
    "random": "随机发言 (Random)",
    "leader_worker": "领导-组员 (Leader-Worker)",
}

AGENT_AVATARS: list[str] = ["🔵", "🟢", "🟠", "🔴", "🟣", "🟡", "⚪", "🟤"]

# ═══════════════════════════════════════════════════════════════
# CPSS 55-item 词库（与 cpss_evaluator.py 严格对齐）
# ═══════════════════════════════════════════════════════════════

CPSS_ITEMS: list[dict[str, Any]] = [
    {"id": 1,  "left": "Over Used",      "right": "Fresh",            "key": "Q01_OverUsed_Fresh"},
    {"id": 2,  "left": "Stale",          "right": "Startling",        "key": "Q02_Stale_Startling"},
    {"id": 3,  "left": "Illogical",      "right": "Logical",          "key": "Q03_Illogical_Logical"},
    {"id": 4,  "left": "Usual",          "right": "Unusual",          "key": "Q04_Usual_Unusual"},
    {"id": 5,  "left": "Inadequate",     "right": "Adequate",         "key": "Q05_Inadequate_Adequate"},
    {"id": 6,  "left": "Original",       "right": "Conventional",     "key": "Q06_Original_Conventional"},
    {"id": 7,  "left": "Trendy",         "right": "Outdated",         "key": "Q07_Trendy_Outdated"},
    {"id": 8,  "left": "Unique",         "right": "Ordinary",         "key": "Q08_Unique_Ordinary"},
    {"id": 9,  "left": "Functional",     "right": "Nonfunctional",    "key": "Q09_Functional_Nonfunctional"},
    {"id": 10, "left": "Useful",         "right": "Useless",          "key": "Q10_Useful_Useless"},
    {"id": 11, "left": "Irrelevant",     "right": "Relevant",         "key": "Q11_Irrelevant_Relevant"},
    {"id": 12, "left": "Trivial",        "right": "Important",        "key": "Q12_Trivial_Important"},
    {"id": 13, "left": "Novel",          "right": "Predictable",      "key": "Q13_Novel_Predictable"},
    {"id": 14, "left": "Surprising",     "right": "Commonplace",      "key": "Q14_Surprising_Commonplace"},
    {"id": 15, "left": "Germane",        "right": "Inappropriate",    "key": "Q15_Germane_Inappropriate"},
    {"id": 16, "left": "Resourceful",    "right": "Unresourceful",    "key": "Q16_Resourceful_Unresourceful"},
    {"id": 17, "left": "Inoperable",     "right": "Workable",         "key": "Q17_Inoperable_Workable"},
    {"id": 18, "left": "Tasteful",       "right": "Tasteless",        "key": "Q18_Tasteful_Tasteless"},
    {"id": 19, "left": "Organic",        "right": "Contrived",        "key": "Q19_Organic_Contrived"},
    {"id": 20, "left": "Well Made",      "right": "Poorly Made",      "key": "Q20_WellMade_PoorlyMade"},
    {"id": 21, "left": "Valuable",       "right": "Worthless",        "key": "Q21_Valuable_Worthless"},
    {"id": 22, "left": "Shocking",       "right": "Old-Fashioned",    "key": "Q22_Shocking_OldFashioned"},
    {"id": 23, "left": "Elaborate",      "right": "Simple",           "key": "Q23_Elaborate_Simple"},
    {"id": 24, "left": "Misunderstood",  "right": "Understood",       "key": "Q24_Misunderstood_Understood"},
    {"id": 25, "left": "Exciting",       "right": "Dull",             "key": "Q25_Exciting_Dull"},
    {"id": 26, "left": "Inspired",       "right": "Uninspired",       "key": "Q26_Inspired_Uninspired"},
    {"id": 27, "left": "Hostile",        "right": "Inviting",         "key": "Q27_Hostile_Inviting"},
    {"id": 28, "left": "Elegant",        "right": "Inelegant",        "key": "Q28_Elegant_Inelegant"},
    {"id": 29, "left": "Valid",          "right": "Invalid",          "key": "Q29_Valid_Invalid"},
    {"id": 30, "left": "Expressive",     "right": "Unexpressive",     "key": "Q30_Expressive_Unexpressive"},
    {"id": 31, "left": "Ambitious",      "right": "Unambitious",      "key": "Q31_Ambitious_Unambitious"},
    {"id": 32, "left": "Vital",          "right": "Unimportant",      "key": "Q32_Vital_Unimportant"},
    {"id": 33, "left": "Effective",      "right": "Ineffective",      "key": "Q33_Effective_Ineffective"},
    {"id": 34, "left": "Progressive",    "right": "Regressive",       "key": "Q34_Progressive_Regressive"},
    {"id": 35, "left": "Imaginative",    "right": "Unimaginative",    "key": "Q35_Imaginative_Unimaginative"},
    {"id": 36, "left": "Avant-Garde",    "right": "Old-Guard",        "key": "Q36_AvantGarde_OldGuard"},
    {"id": 37, "left": "Radical",        "right": "Old Hat",          "key": "Q37_Radical_OldHat"},
    {"id": 38, "left": "Unpolished",     "right": "Polished",         "key": "Q38_Unpolished_Polished"},
    {"id": 39, "left": "Complete",       "right": "Incomplete",       "key": "Q39_Complete_Incomplete"},
    {"id": 40, "left": "Cohesive",       "right": "Disjointed",       "key": "Q40_Cohesive_Disjointed"},
    {"id": 41, "left": "Needed",         "right": "Unneeded",         "key": "Q41_Needed_Unneeded"},
    {"id": 42, "left": "Meticulous",     "right": "Careless",         "key": "Q42_Meticulous_Careless"},
    {"id": 43, "left": "Revolutionary",  "right": "Pedestrian",       "key": "Q43_Revolutionary_Pedestrian"},
    {"id": 44, "left": "Pleasurable",    "right": "Unpleasant",       "key": "Q44_Pleasurable_Unpleasant"},
    {"id": 45, "left": "Crude",          "right": "Well-Crafted",     "key": "Q45_Crude_WellCrafted"},
    {"id": 46, "left": "Visionary",      "right": "Mundane",          "key": "Q46_Visionary_Mundane"},
    {"id": 47, "left": "Insightful",     "right": "Trite",            "key": "Q47_Insightful_Trite"},
    {"id": 48, "left": "Desire",         "right": "Undesirable",      "key": "Q48_Desire_Undesirable"},
    {"id": 49, "left": "Deliberate",     "right": "Random",           "key": "Q49_Deliberate_Random"},
    {"id": 50, "left": "Appealing",      "right": "Unappealing",      "key": "Q50_Appealing_Unappealing"},
    {"id": 51, "left": "Detailed",       "right": "Sketchy",          "key": "Q51_Detailed_Sketchy"},
    {"id": 52, "left": "Feasible",       "right": "Unfeasible",       "key": "Q52_Feasible_Unfeasible"},
    {"id": 53, "left": "Meaningful",     "right": "Meaningless",      "key": "Q53_Meaningful_Meaningless"},
    {"id": 54, "left": "Flexible",       "right": "Inflexible",       "key": "Q54_Flexible_Inflexible"},
    {"id": 55, "left": "Overused",       "right": "New",              "key": "Q55_Overused_New"},
]

# ═══════════════════════════════════════════════════════════════
# 工具函数：用户历史
# ═══════════════════════════════════════════════════════════════


def get_user_history_path(user_name: str) -> Path:
    return BASE_DIR / USER_LOG_DIR / f"{user_name}_history.json"


def load_user_history(user_name: str) -> set[str]:
    path = get_user_history_path(user_name)
    if not path.exists():
        return set()
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return set()
    return set(data.get("annotated_files", []))


def save_user_history(user_name: str, annotated_files: set[str]) -> None:
    path = get_user_history_path(user_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {"annotated_files": sorted(annotated_files)},
            f, ensure_ascii=False, indent=2,
        )


# ═══════════════════════════════════════════════════════════════
# 工具函数：日志收集
# ═══════════════════════════════════════════════════════════════


def _resolve_dir(p: str) -> Path:
    """支持绝对/相对路径，相对路径相对于 PROJECT_ROOT 解析。"""
    path = Path(p)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def collect_log_files() -> list[str]:
    """递归扫描 TARGET_EVAL_DIRS 下所有 .json 文件，返回相对 PROJECT_ROOT 的路径。"""
    seen: set[str] = set()
    files: list[str] = []
    for d in TARGET_EVAL_DIRS:
        abs_dir = _resolve_dir(d)
        if not abs_dir.is_dir():
            continue
        for json_file in sorted(abs_dir.rglob("*.json")):
            # 跳过疑似锁/状态文件
            name = json_file.name
            if name.endswith(".lock") or name == "batch_experiment_state.json":
                continue
            try:
                rel_path = str(json_file.resolve().relative_to(PROJECT_ROOT))
            except ValueError:
                rel_path = str(json_file.resolve())
            if rel_path in seen:
                continue
            seen.add(rel_path)
            files.append(rel_path)
    return files


def load_log_data(rel_path: str) -> dict:
    abs_path = PROJECT_ROOT / rel_path
    with open(abs_path, "r", encoding="utf-8") as f:
        return json.load(f)


# ═══════════════════════════════════════════════════════════════
# 工具函数：脱敏映射
# ═══════════════════════════════════════════════════════════════


def _extract_config_key(agent: dict) -> str:
    if "config_key" in agent and agent["config_key"]:
        return agent["config_key"]
    if "name" in agent and agent["name"]:
        return agent["name"]
    return str(agent.get("position", agent.get("agent_id", "")))


def build_anonymization_map(log_data: dict, seed: str) -> dict[int, dict]:
    """构建 {原始 agent_id -> {display_name, display_idx, config_key, type, original_agent_name}}。

    使用确定性种子保证同一文件、同一用户每次打开看到一致的脱敏映射。
    """
    agents = log_data["metadata"]["agents"]
    n = len(agents)

    rng = random.Random(seed)
    display_indices = list(range(1, n + 1))
    rng.shuffle(display_indices)

    mapping: dict[int, dict] = {}
    for i, agent in enumerate(agents):
        orig_id = agent["agent_id"]
        mapping[orig_id] = {
            "display_name": f"Agent {chr(ord('A') + display_indices[i] - 1)}",
            "display_idx": display_indices[i],
            "config_key": _extract_config_key(agent),
            "type": agent.get("type", "llm"),
            "position": agent.get("position", orig_id),
            "agent_id": orig_id,
            "original_agent_name": f"Agent {orig_id}",
        }
    return mapping


def anonymize_content(content: str, anon_map: dict[int, dict]) -> str:
    """将正文中出现的原始 Agent 名称替换为脱敏后名称。

    使用占位符两阶段替换，避免数字越界（如 "Agent 1" 误命中 "Agent 11"）。
    """
    if not content:
        return content
    sorted_ids = sorted(anon_map.keys(), reverse=True)

    placeholders: dict[int, str] = {}
    for orig_id in sorted_ids:
        orig_name = anon_map[orig_id]["original_agent_name"]
        placeholder = f"\x00ANON{orig_id}\x00"
        placeholders[orig_id] = placeholder
        content = content.replace(orig_name, placeholder)

    for orig_id in sorted_ids:
        content = content.replace(placeholders[orig_id], anon_map[orig_id]["display_name"])

    return content


# ═══════════════════════════════════════════════════════════════
# 工具函数：Agent 发言聚合（参考 cpss_evaluator.extract_agent_ideas）
# ═══════════════════════════════════════════════════════════════


def extract_agent_ideas(data: dict) -> dict[str, dict[str, Any]]:
    """聚合每个 Agent 的发言为一段连贯文本，并携带 agent_id / position / config_key。"""
    agent_ideas: dict[str, list[str]] = {}
    agent_meta: dict[str, dict[str, Any]] = {}

    agent_id_to_config: dict[int, str] = {}
    agent_id_to_position: dict[int, int] = {}
    for a in data.get("metadata", {}).get("agents", []) or []:
        try:
            aid = int(a.get("agent_id"))
        except (TypeError, ValueError):
            continue
        ck = a.get("config_key")
        if isinstance(ck, str) and ck:
            agent_id_to_config[aid] = ck
        if "position" in a:
            try:
                agent_id_to_position[aid] = int(a["position"])
            except (TypeError, ValueError):
                pass

    for turn in data.get("global_history", []):
        content = (turn.get("content") or "").strip()
        if not content:
            continue
        role = turn.get("role", "")
        if role in ["system", "moderator"]:
            continue

        try:
            agent_id_int: int | None = int(turn["agent_id"]) if turn.get("agent_id") is not None else None
        except (TypeError, ValueError):
            agent_id_int = None

        agent_name = turn.get("agent_name") or (
            f"Agent {agent_id_int}" if agent_id_int is not None else "Agent ?"
        )

        agent_ideas.setdefault(agent_name, []).append(content)

        if agent_name not in agent_meta:
            config_key = turn.get("config_key")
            if (not isinstance(config_key, str) or not config_key) and agent_id_int is not None:
                config_key = agent_id_to_config.get(agent_id_int)
            position = agent_id_to_position.get(agent_id_int, agent_id_int) if agent_id_int is not None else None
            agent_meta[agent_name] = {
                "agent_id": agent_id_int,
                "position": position,
                "config_key": config_key,
            }

    out: dict[str, dict[str, Any]] = {}
    for name, texts in agent_ideas.items():
        meta = agent_meta.get(name, {})
        out[name] = {
            "idea_content": "\n\n---\n\n".join(texts),
            "agent_id": meta.get("agent_id"),
            "position": meta.get("position"),
            "config_key": meta.get("config_key"),
        }
    return out


# ═══════════════════════════════════════════════════════════════
# 工具函数：写回打分结果
# ═══════════════════════════════════════════════════════════════


def save_human_evaluation(
    rel_path: str,
    user_name: str,
    evaluation_payload: dict[str, dict[str, Any]],
) -> None:
    """以 FileLock 安全地将打分写入 `human_eval_per_agent_<user_name>` 字段。"""
    abs_path = PROJECT_ROOT / rel_path
    lock_path = str(abs_path) + ".lock"
    field_key = f"human_eval_per_agent_{user_name}"

    with FileLock(lock_path, timeout=10):
        with open(abs_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        data[field_key] = evaluation_payload

        with open(abs_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


# ═══════════════════════════════════════════════════════════════
# Streamlit 页面入口
# ═══════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="人类 CPSS Agent 打分系统",
    page_icon="🧠",
    layout="wide",
)

# ──────────────────────────────────────
# 1. 登录模块
# ──────────────────────────────────────

if "h_logged_in" not in st.session_state:
    st.session_state.h_logged_in = False
    st.session_state.h_user_name = ""

if not st.session_state.h_logged_in:
    st.title("🧠 人类 Agent-Level CPSS 打分系统")
    st.markdown("---")
    st.markdown("#### 请输入您的姓名以开始独立 Agent 打分工作")
    col_input, col_btn = st.columns([3, 1])
    with col_input:
        name_input = st.text_input(
            "User Name",
            placeholder="例如：Senhao",
            label_visibility="collapsed",
        )
    with col_btn:
        login_clicked = st.button("确认登录", type="primary", use_container_width=True)

    if login_clicked:
        cleaned = (name_input or "").strip()
        if not cleaned:
            st.error("用户名不能为空。")
        elif any(c in cleaned for c in r"\/:*?\"<>|"):
            st.error("用户名包含非法字符，请勿使用 \\ / : * ? \" < > | 等符号。")
        else:
            st.session_state.h_user_name = cleaned
            st.session_state.h_logged_in = True
            st.rerun()
    st.stop()

# ──────────────────────────────────────
# 2. 侧边栏：用户信息 + 文件选择
# ──────────────────────────────────────

user_name: str = st.session_state.h_user_name

with st.sidebar:
    st.markdown(f"### 👤 {user_name}")
    if st.button("退出登录", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

    st.divider()

    annotated_set = load_user_history(user_name)
    all_files = collect_log_files()
    pending_files = [f for f in all_files if f not in annotated_set]

    col_a, col_b = st.columns(2)
    col_a.metric("已标注", f"{len(annotated_set)}")
    col_b.metric("待标注", f"{len(pending_files)}")

    st.caption(f"扫描目录：{', '.join(TARGET_EVAL_DIRS) or '(空)'}")
    st.divider()

    if not pending_files:
        st.success("🎉 所有文件已标注完成！")
        st.balloons()
        st.title("🧠 人类 Agent-Level CPSS 打分系统")
        st.info("当前没有待标注的文件，请等待新的实验日志或检查 `TARGET_EVAL_DIRS` 配置。")
        st.stop()

    selected_file = st.selectbox(
        "选择待标注文件",
        pending_files,
        format_func=lambda x: os.path.basename(x),
    )
    st.caption(f"路径：`{selected_file}`")

# ──────────────────────────────────────
# 3. 数据加载与脱敏
# ──────────────────────────────────────

try:
    log_data = load_log_data(selected_file)
except Exception as e:
    st.error(f"无法加载日志文件：{e}")
    st.stop()

metadata = log_data.get("metadata", {})
if not metadata or "agents" not in metadata:
    st.error("⚠️ 该 JSON 文件缺少 metadata.agents 字段，无法识别 Agent 列表。")
    st.stop()

seed = f"{selected_file}||{user_name}"
anon_map = build_anonymization_map(log_data, seed)
agent_ideas_dict = extract_agent_ideas(log_data)

if not agent_ideas_dict:
    st.error("⚠️ 该日志中没有任何有效 Agent 发言（global_history 为空）。")
    st.stop()

# 反查：原始 agent_name (如 "Agent 1") -> 脱敏 display info
name_to_display: dict[str, dict] = {}
for orig_id, info in anon_map.items():
    name_to_display[info["original_agent_name"]] = info
# 对于在 history 中但不在 metadata.agents 中的兜底（罕见）
for agent_name in agent_ideas_dict.keys():
    if agent_name not in name_to_display:
        name_to_display[agent_name] = {
            "display_name": agent_name,
            "display_idx": 999,
            "config_key": agent_ideas_dict[agent_name].get("config_key") or "unknown",
            "type": "llm",
            "position": agent_ideas_dict[agent_name].get("position"),
            "agent_id": agent_ideas_dict[agent_name].get("agent_id"),
            "original_agent_name": agent_name,
        }

# 按 display_idx 排序，确保 Agent A、Agent B... 顺序展示
sorted_agent_names: list[str] = sorted(
    agent_ideas_dict.keys(),
    key=lambda n: name_to_display[n]["display_idx"],
)

# ──────────────────────────────────────
# 4. 主界面：元信息 + 讨论记录
# ──────────────────────────────────────

st.title("🧠 人类 Agent-Level CPSS 打分系统")
st.markdown(f"**当前文件：** `{selected_file}`")

meta_cols = st.columns(4)
meta_cols[0].metric("讨论模式", MODE_LABELS.get(metadata.get("mode", ""), metadata.get("mode", "—")))
meta_cols[1].metric("总轮数", metadata.get("max_rounds", "—"))
meta_cols[2].metric("参与人数", metadata.get("total_agents", len(metadata.get("agents", []))))
meta_cols[3].metric("Agent 数量", len(sorted_agent_names))
st.info(f"**讨论话题：** {metadata.get('topic', '(未知话题)')}")

with st.expander("💬 查看完整讨论记录（已脱敏）", expanded=False):
    global_history = log_data.get("global_history", [])
    current_round = 0
    for entry in global_history:
        if entry.get("role") in ["system", "moderator"]:
            continue
        round_no = entry.get("round", 0)
        if round_no != current_round:
            current_round = round_no
            st.markdown(f"#### 第 {current_round} 轮")
        orig_id = entry.get("agent_id")
        if orig_id not in anon_map:
            continue
        info = anon_map[orig_id]
        display_name = info["display_name"]
        avatar = AGENT_AVATARS[(info["display_idx"] - 1) % len(AGENT_AVATARS)]
        content = anonymize_content(entry.get("content", ""), anon_map)
        with st.chat_message(name=display_name, avatar=avatar):
            st.markdown(f"**{display_name}**")
            st.markdown(content)

st.divider()

# ──────────────────────────────────────
# 5. 按 Agent 分 Tab 的 CPSS 打分表单
# ──────────────────────────────────────

st.markdown("### 🏆 按 Agent 独立打分（CPSS 55 维双极语义量表）")
st.markdown(
    "请切换不同 **Agent 标签页**，对每个 Agent 的「综合表现/创意提案」在 55 个双极维度上分别打分。"
    "每个维度为 **1-7** 分，**1** 偏向左侧形容词，**7** 偏向右侧形容词，**4** 为中性。"
    "全部打分完成后再点击页面底部的「📤 提交当前文件标注」。"
)

# 为每次切换文件清空残留 widget 状态
if st.session_state.get("_h_current_file") != selected_file:
    keys_to_clear = [k for k in st.session_state.keys() if k.startswith("h_score_")]
    for k in keys_to_clear:
        del st.session_state[k]
    st.session_state._h_current_file = selected_file


def _radio_key(agent_name: str, item_key: str) -> str:
    return f"h_score_{selected_file}::{agent_name}::{item_key}"


with st.form("human_cpss_form", clear_on_submit=False):
    tab_labels = []
    for name in sorted_agent_names:
        info = name_to_display[name]
        avatar = AGENT_AVATARS[(info["display_idx"] - 1) % len(AGENT_AVATARS)]
        tab_labels.append(f"{avatar} {info['display_name']}")

    tabs = st.tabs(tab_labels)

    for tab, agent_name in zip(tabs, sorted_agent_names):
        info = name_to_display[agent_name]
        idea_content = agent_ideas_dict[agent_name]["idea_content"]
        anonymized_idea = anonymize_content(idea_content, anon_map)
        avatar = AGENT_AVATARS[(info["display_idx"] - 1) % len(AGENT_AVATARS)]

        with tab:
            st.markdown(f"#### {avatar} {info['display_name']} — 综合发言/创意提案")
            with st.container(border=True):
                st.markdown(anonymized_idea)

            st.markdown("#### 📊 CPSS 55 维评分")
            st.caption("提示：每个维度需选择一个 1-7 整数；未填写项会在提交时阻断。")

            for item in CPSS_ITEMS:
                cols = st.columns([0.6, 2.0, 5.0, 2.0])
                cols[0].markdown(
                    f"<div style='padding-top:6px'><b>Q{item['id']:02d}</b></div>",
                    unsafe_allow_html=True,
                )
                cols[1].markdown(
                    f"<div style='padding-top:6px;text-align:right'>"
                    f"<i>1 ⬅ {item['left']}</i></div>",
                    unsafe_allow_html=True,
                )
                cols[2].radio(
                    label=f"Q{item['id']}_{agent_name}",
                    options=[1, 2, 3, 4, 5, 6, 7],
                    index=None,
                    horizontal=True,
                    key=_radio_key(agent_name, item["key"]),
                    label_visibility="collapsed",
                )
                cols[3].markdown(
                    f"<div style='padding-top:6px'>"
                    f"<i>{item['right']} ➡ 7</i></div>",
                    unsafe_allow_html=True,
                )

    st.markdown("")
    submit_col, _spacer = st.columns([1, 3])
    with submit_col:
        submitted = st.form_submit_button(
            "📤 提交当前文件标注",
            type="primary",
            use_container_width=True,
        )

# ──────────────────────────────────────
# 6. 提交校验 + 写回 + 状态刷新
# ──────────────────────────────────────

if submitted:
    missing: list[tuple[str, str]] = []
    evaluation_payload: dict[str, dict[str, Any]] = {}

    for agent_name in sorted_agent_names:
        info = name_to_display[agent_name]
        scores: dict[str, int] = {}
        for item in CPSS_ITEMS:
            val = st.session_state.get(_radio_key(agent_name, item["key"]))
            if val is None:
                missing.append((info["display_name"], f"Q{item['id']:02d}"))
            else:
                scores[item["key"]] = int(val)

        evaluation_payload[agent_name] = {
            "agent_id": info.get("agent_id"),
            "position": info.get("position"),
            "config_key": info.get("config_key"),
            "scores": scores,
        }

    if missing:
        # 按 Agent 聚合，给出友好的提示
        agg: dict[str, list[str]] = {}
        for d_name, q_id in missing:
            agg.setdefault(d_name, []).append(q_id)
        msg_lines = [
            f"⚠️ 共有 {len(missing)} 项评分尚未填写，无法提交。请在下列 Agent 标签页中补齐："
        ]
        for d_name, qs in sorted(agg.items()):
            preview = ", ".join(qs[:8]) + ("..." if len(qs) > 8 else "")
            msg_lines.append(f"- **{d_name}** 缺 {len(qs)} 项：{preview}")
        st.error("\n".join(msg_lines))
        st.stop()

    try:
        save_human_evaluation(selected_file, user_name, evaluation_payload)
    except Exception as e:
        st.error(f"写入标注结果失败：{e}")
        st.stop()

    annotated_set.add(selected_file)
    save_user_history(user_name, annotated_set)

    # 清理本次表单残留状态
    keys_to_clear = [k for k in list(st.session_state.keys()) if k.startswith("h_score_")]
    for k in keys_to_clear:
        del st.session_state[k]
    if "_h_current_file" in st.session_state:
        del st.session_state["_h_current_file"]

    st.success(
        f"✅ 已成功为 {len(sorted_agent_names)} 个 Agent 写入 CPSS 打分。"
        f"字段名：`human_eval_per_agent_{user_name}`。页面即将自动跳转下一文件..."
    )
    st.rerun()
