"""
第三方标注评估系统 — Streamlit 独立入口

启动方式：
    streamlit run app_eval_ui.py

功能：
    - 用户登录与历史标注追踪
    - 实验日志脱敏展示（Agent 身份随机重映射）
    - 拖拽/下拉式排序标注
    - 标注结果持久化写入原 JSON（3port 字段）+ 用户日志
"""

import json
import os
import random
import streamlit as st
from pathlib import Path
from filelock import FileLock

# ═══════════════════════════════════════════════════════════════
# 后台配置项
# ═══════════════════════════════════════════════════════════════

TARGET_LOG_DIRS: list[str] = [
    "log_human",
    "log_human_2",
]

USER_LOG_DIR = "user_log"
BASE_DIR = Path(__file__).resolve().parent

MODE_LABELS = {
    "brainwrite": "脑力书写 (BrainWrite)",
    "round_robin": "轮流发言 (Round Robin)",
    "random": "随机发言 (Random)",
    "leader_worker": "领导-组员 (Leader-Worker)",
}

AGENT_AVATARS = ["🔵", "🟢", "🟠", "🔴", "🟣", "🟡", "⚪", "🟤"]


# ═══════════════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════════════

def get_user_history_path(user_name: str) -> Path:
    return BASE_DIR / USER_LOG_DIR / f"{user_name}_history.json"


def load_user_history(user_name: str) -> set[str]:
    path = get_user_history_path(user_name)
    if not path.exists():
        return set()
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return set(data.get("annotated_files", []))


def save_user_history(user_name: str, annotated_files: set[str]):
    path = get_user_history_path(user_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {"annotated_files": sorted(annotated_files)},
            f, ensure_ascii=False, indent=2,
        )


def collect_log_files() -> list[str]:
    """遍历配置的日志目录，收集所有 JSON 文件的相对路径。"""
    files = []
    for log_dir in TARGET_LOG_DIRS:
        abs_dir = BASE_DIR / log_dir
        if not abs_dir.is_dir():
            continue
        for json_file in sorted(abs_dir.glob("*.json")):
            rel_path = str(json_file.relative_to(BASE_DIR))
            files.append(rel_path)
    return files


def load_log_data(rel_path: str) -> dict:
    abs_path = BASE_DIR / rel_path
    with open(abs_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _extract_config_key(agent: dict) -> str:
    """兼容不同版本日志中的身份标识字段（config_key / name / position）。"""
    if "config_key" in agent:
        return agent["config_key"]
    if "name" in agent:
        return agent["name"]
    return str(agent.get("position", agent["agent_id"]))


def build_anonymization_map(log_data: dict, seed: str) -> dict[int, dict]:
    """
    构建脱敏映射表。使用确定性种子保证同一文件、同一用户
    每次打开看到的映射始终一致。

    返回 {原始 agent_id: {display_name, display_idx, config_key, type, original_agent_name}}
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
            "display_name": f"Agent {display_indices[i]}",
            "display_idx": display_indices[i],
            "config_key": _extract_config_key(agent),
            "type": agent.get("type", "llm"),
            "original_agent_name": f"Agent {orig_id}",
        }
    return mapping


def anonymize_content(content: str, anon_map: dict[int, dict]) -> str:
    """将消息正文中出现的原始 Agent 名称替换为脱敏后的名称。"""
    sorted_ids = sorted(anon_map.keys(), reverse=True)

    placeholders = {}
    for orig_id in sorted_ids:
        orig_name = anon_map[orig_id]["original_agent_name"]
        placeholder = f"\x00ANON{orig_id}\x00"
        placeholders[orig_id] = placeholder
        content = content.replace(orig_name, placeholder)

    for orig_id in sorted_ids:
        content = content.replace(placeholders[orig_id], anon_map[orig_id]["display_name"])

    return content


def validate_strict_total_order(rankings: dict[str, int], n: int) -> tuple[bool, str]:
    """严格全序校验：排名值必须恰好构成 {1, 2, ..., N} 且无重复。"""
    rank_values = list(rankings.values())
    expected = set(range(1, n + 1))
    if set(rank_values) != expected:
        duplicates = sorted({v for v in rank_values if rank_values.count(v) > 1})
        missing = sorted(expected - set(rank_values))
        parts = []
        if duplicates:
            parts.append(f"存在重复排名 {duplicates}")
        if missing:
            parts.append(f"缺少排名 {missing}")
        return False, "；".join(parts)
    return True, ""


def save_annotation(rel_path: str, user_name: str, ranking_results: list[dict]):
    """将标注结果写入原 JSON 文件的 3port 字段，使用文件锁防止并发覆盖。"""
    abs_path = BASE_DIR / rel_path
    lock_path = str(abs_path) + ".lock"

    with FileLock(lock_path, timeout=10):
        with open(abs_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if "3port" not in data:
            data["3port"] = {}
        data["3port"][user_name] = ranking_results

        with open(abs_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


# ═══════════════════════════════════════════════════════════════
# Streamlit 页面
# ═══════════════════════════════════════════════════════════════

st.set_page_config(page_title="头脑风暴标注系统", page_icon="📋", layout="wide")

# ──────────────────────────────────────
# 1. 登录模块
# ──────────────────────────────────────

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.user_name = ""

if not st.session_state.logged_in:
    st.title("📋 头脑风暴第三方标注系统")
    st.markdown("---")
    st.markdown("#### 请输入您的姓名以开始标注工作")

    col_input, col_btn = st.columns([3, 1])
    with col_input:
        name_input = st.text_input(
            "User Name", placeholder="例如：Senhao", label_visibility="collapsed",
        )
    with col_btn:
        login_clicked = st.button("确认登录", type="primary", use_container_width=True)

    if login_clicked:
        cleaned = name_input.strip()
        if not cleaned:
            st.error("用户名不能为空。")
        else:
            st.session_state.user_name = cleaned
            st.session_state.logged_in = True
            st.rerun()
    st.stop()

# ──────────────────────────────────────
# 2. 侧边栏：用户信息 + 文件选择
# ──────────────────────────────────────

user_name: str = st.session_state.user_name

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

    st.metric("已标注", f"{len(annotated_set)} 个")
    st.metric("待标注", f"{len(pending_files)} 个")

    st.divider()

    if not pending_files:
        st.success("🎉 所有文件已标注完成！")
        st.title("📋 头脑风暴第三方标注系统")
        st.info("当前没有待标注的文件。请等待新的实验数据。")
        st.stop()

    selected_file = st.selectbox(
        "选择待标注文件",
        pending_files,
        format_func=lambda x: os.path.basename(x) if "/" in x else x,
    )

# ──────────────────────────────────────
# 3. 数据加载与脱敏
# ──────────────────────────────────────

log_data = load_log_data(selected_file)
metadata = log_data["metadata"]

seed = f"{selected_file}||{user_name}"
anon_map = build_anonymization_map(log_data, seed)

st.title("📋 头脑风暴第三方标注系统")

# 文件元信息
st.markdown(f"**当前文件：** `{selected_file}`")
meta_cols = st.columns(3)
meta_cols[0].metric("讨论模式", MODE_LABELS.get(metadata["mode"], metadata["mode"]))
meta_cols[1].metric("总轮数", metadata["max_rounds"])
meta_cols[2].metric("参与人数", metadata["total_agents"])
st.info(f"**讨论话题：** {metadata['topic']}")

st.divider()

# ──────────────────────────────────────
# 4. 对局内容展示（脱敏）
# ──────────────────────────────────────

st.markdown("### 💬 讨论记录")

global_history = log_data["global_history"]
current_round = 0

for entry in global_history:
    if entry["round"] != current_round:
        current_round = entry["round"]
        st.markdown(f"#### 第 {current_round} 轮")

    orig_id = entry["agent_id"]
    info = anon_map[orig_id]
    display_name = info["display_name"]
    avatar = AGENT_AVATARS[(info["display_idx"] - 1) % len(AGENT_AVATARS)]

    content = anonymize_content(entry["content"], anon_map)

    with st.chat_message(name=display_name, avatar=avatar):
        st.markdown(f"**{display_name}**")
        st.markdown(content)

st.divider()

# ──────────────────────────────────────
# 5. 排序标注模块（联动交换机制，参考 app.py / app_multiplayer.py）
# ──────────────────────────────────────

display_agents = sorted(anon_map.items(), key=lambda x: x[1]["display_idx"])
n_agents = len(display_agents)
rank_options = list(range(1, n_agents + 1))

# display_name 列表，用于联动交换时索引
_eval_agent_names = [info["display_name"] for _, info in display_agents]


def _init_eval_ranking_state():
    """初始化排名 session_state，每个 Agent 默认分配不同名次。"""
    if "eval_ranking_selections" not in st.session_state or \
       st.session_state.get("_eval_ranking_file") != selected_file:
        st.session_state._eval_ranking_file = selected_file
        st.session_state.eval_ranking_selections = {
            name: idx + 1 for idx, name in enumerate(_eval_agent_names)
        }


def _on_eval_rank_change(agent_name: str):
    """排名下拉框回调：当用户更改某个 Agent 的排名时，自动交换冲突名次。

    同时同步更新被交换 Agent 对应的 widget key，确保 UI 联动生效。
    """
    new_rank = st.session_state[f"eval_rank_{agent_name}"]
    sel = st.session_state.eval_ranking_selections
    old_rank = sel.get(agent_name)

    for name, r in sel.items():
        if name != agent_name and r == new_rank:
            sel[name] = old_rank
            st.session_state[f"eval_rank_{name}"] = old_rank
            break

    sel[agent_name] = new_rank


_init_eval_ranking_state()

st.markdown("### 🏆 排序标注")
st.markdown("请根据各 Agent 在讨论中的**表现质量**进行排名（**1 = 最佳**，数字越小越好）。名次会自动联动，确保不重复。")

n_cols = min(n_agents, 4)
cols = st.columns(n_cols)

for i, (orig_id, info) in enumerate(display_agents):
    col = cols[i % n_cols]
    avatar = AGENT_AVATARS[(info["display_idx"] - 1) % len(AGENT_AVATARS)]
    display_name = info["display_name"]
    current_rank = st.session_state.eval_ranking_selections[display_name]
    with col:
        st.selectbox(
            f"{avatar} {display_name} 的排名",
            options=rank_options,
            index=current_rank - 1,
            key=f"eval_rank_{display_name}",
            on_change=_on_eval_rank_change,
            args=(display_name,),
        )

st.markdown("")
submit_col, spacer = st.columns([1, 3])
with submit_col:
    submitted = st.button("📤 提交标注", type="primary", use_container_width=True)

if submitted:
    rankings = st.session_state.eval_ranking_selections

    valid, err_msg = validate_strict_total_order(rankings, n_agents)
    if not valid:
        st.error(
            f"⚠️ 排名无效！请为每位 Agent 分配从 1 到 {n_agents} 的不重复名次，"
            "不允许并列排名。请重新调整后再提交。"
        )
    else:
        reverse_map = {
            info["display_name"]: {"position": orig_id, "config_key": info["config_key"]}
            for orig_id, info in anon_map.items()
        }

        ranking_results = []
        for display_name, rank in sorted(rankings.items(), key=lambda x: x[1]):
            ref = reverse_map[display_name]
            ranking_results.append({
                "position": ref["position"],
                "config_key": ref["config_key"],
                "rank": rank,
            })

        try:
            save_annotation(selected_file, user_name, ranking_results)
        except Exception as e:
            st.error(f"写入标注结果失败：{e}")
            st.stop()

        annotated_set.add(selected_file)
        save_user_history(user_name, annotated_set)

        del st.session_state["eval_ranking_selections"]
        del st.session_state["_eval_ranking_file"]

        st.success("✅ 标注已提交！页面即将刷新...")
        st.rerun()
