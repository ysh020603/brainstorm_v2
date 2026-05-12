"""
第三方标注评估系统 — Streamlit 独立入口

启动方式：
    streamlit run app_eval_ui.py

功能：
    - 用户登录与历史标注追踪
    - 实验日志脱敏展示（Agent 身份随机重映射）
    - 并发调用大模型 API 对脱敏文本进行翻译（包含缓存机制）
    - 拖拽/下拉式排序标注
    - 智能推荐待标注文件（全局模式平衡、次数均衡、Hash 打散）
    - 限定仅支持 round_robin 模式
    - ELO 分数计算（支持历史记录首次初始化与增量更新）写入 user_log
"""

import json
import os
import random
import hashlib
import collections
import streamlit as st
import openai
import concurrent.futures
from pathlib import Path
from filelock import FileLock
from typing import Any

# ═══════════════════════════════════════════════════════════════
# 后台配置项
# ═══════════════════════════════════════════════════════════════

TARGET_LOG_DIRS: list[str] = [
    "/data2/brainstorm/brainstorm_v2/log_experiment/ex1_4LLM",
]

USER_LOG_DIR = "user_log"
BASE_DIR = Path(__file__).resolve().parent
GLOBAL_ELO_FILE = BASE_DIR / USER_LOG_DIR / "global_elo_scores.json"

# ELO 配置
INITIAL_ELO = 1200.0
K_FACTOR = 32.0

MODE_LABELS = {
    "brainwrite": "脑力书写 (BrainWrite)",
    "round_robin": "轮流发言 (Round Robin)",
    "random": "随机发言 (Random)",
    "leader_worker": "领导-组员 (Leader-Worker)",
}

AGENT_AVATARS = ["🔵", "🟢", "🟠", "🔴", "🟣", "🟡", "⚪", "🟤"]


# ═══════════════════════════════════════════════════════════════
# 翻译核心功能
# ═══════════════════════════════════════════════════════════════

TRANSLATION_CONFIG_PATH = BASE_DIR / "translation_config" / "config.json"

def _load_translation_config() -> dict[str, Any]:
    if not TRANSLATION_CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"未找到辅助翻译配置文件 {TRANSLATION_CONFIG_PATH}。"
            f"请复制 translation_config/example.json 为 config.json 并填写真实 API 凭据。"
        )
    with open(TRANSLATION_CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def _single_translate(text: str) -> str:
    if not text or not text.strip():
        return ""
    try:
        cfg = _load_translation_config()
    except Exception as e:
        return f"[翻译配置加载失败: {str(e)}]"

    client = openai.OpenAI(base_url=cfg["base_url"], api_key=cfg["api_key"])

    call_kwargs: dict[str, Any] = {
        "model": cfg["model"],
        "messages": [
            {
                "role": "system",
                "content": cfg.get("system_prompt", "你是一个专业的辅助翻译系统。请将用户提供的文本翻译成中文。保持专业、流畅，不改变原意。请直接输出翻译结果。"),
            },
            {"role": "user", "content": text},
        ],
    }
    
    is_reasoning = False
    if is_reasoning is False:
        model_name = str(call_kwargs.get("model", "")).lower()
        if "kimi" in model_name:
            call_kwargs["temperature"] = 0.6
            call_kwargs["extra_body"] = {"thinking": {"type": "disabled"}, "chat_template_kwargs": {"thinking": False}}
        elif "glm" in model_name or "deepseek" in model_name:
            call_kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
        else:
            call_kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}

    try:
        response = client.chat.completions.create(**call_kwargs)
        return response.choices[0].message.content
    except Exception as e:
        return f"[翻译调用异常: {str(e)}]"

@st.cache_data(show_spinner="首次加载该文件，正在并发翻译讨论记录中，请稍候...")
def get_translated_history(_global_history: list, _anon_map: dict, file_id: str) -> dict:
    translations = {}
    tasks = {}
    try:
        max_workers = int(_load_translation_config().get("max_workers", 10))
    except Exception:
        max_workers = 10

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        for idx, entry in enumerate(_global_history):
            content = anonymize_content(entry["content"], _anon_map)
            tasks[executor.submit(_single_translate, content)] = idx
            
        for future in concurrent.futures.as_completed(tasks):
            idx = tasks[future]
            try:
                translations[idx] = future.result()
            except Exception as e:
                translations[idx] = f"[翻译失败: {e}]"
    return translations

# ═══════════════════════════════════════════════════════════════
# 数据加载与 ELO 功能
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
        json.dump({"annotated_files": sorted(annotated_files)}, f, ensure_ascii=False, indent=2)

def collect_log_files() -> list[str]:
    files = []
    for log_dir in TARGET_LOG_DIRS:
        abs_dir = BASE_DIR / log_dir
        if not abs_dir.is_dir():
            continue
        for json_file in sorted(abs_dir.rglob("*.json")):
            if "leader_worker" in json_file.parts:
                continue
            files.append(str(json_file.relative_to(BASE_DIR)))
    return files

def load_log_data(rel_path: str) -> dict:
    with open(BASE_DIR / rel_path, "r", encoding="utf-8") as f:
        return json.load(f)

@st.cache_data(show_spinner=False)
def get_all_files_metadata(files: list[str]) -> dict[str, dict]:
    """批量缓存并解析所有文件的 metadata，以供智能排序和过滤使用"""
    meta_dict = {}
    for f in files:
        try:
            abs_path = BASE_DIR / f
            with open(abs_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
                meta_dict[f] = data.get("metadata", {})
        except Exception:
            meta_dict[f] = {}
    return meta_dict

def get_global_annotation_stats() -> tuple[dict[str, int], set[str]]:
    """返回：{文件路径: 累计被标次数}, {被标过的文件全集}"""
    global_file_counts = collections.defaultdict(int)
    global_annotated = set()
    user_log_dir = BASE_DIR / USER_LOG_DIR
    if user_log_dir.is_dir():
        for history_file in user_log_dir.glob("*_history.json"):
            try:
                with open(history_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for file_path in data.get("annotated_files", []):
                        global_file_counts[file_path] += 1
                        global_annotated.add(file_path)
            except Exception:
                pass
    return global_file_counts, global_annotated

def _apply_elo_update(ranking_data: list[dict], elo_state: dict):
    """核心 ELO 计算逻辑"""
    models = elo_state.setdefault("models", {})
    ranked_keys = [item["config_key"] for item in ranking_data]
    
    for key in ranked_keys:
        if key not in models:
            models[key] = {"elo": INITIAL_ELO, "appearances": 0}
            
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
        models[model_i]["appearances"] += 1

def init_global_elo_if_needed(all_valid_files: list[str]):
    """如果 global_elo_scores.json 不存在，遍历历史文件重建 ELO 分数"""
    if GLOBAL_ELO_FILE.exists():
        return
        
    GLOBAL_ELO_FILE.parent.mkdir(parents=True, exist_ok=True)
    elo_state = {"models": {}}
    
    user_log_dir = BASE_DIR / USER_LOG_DIR
    if user_log_dir.is_dir():
        for history_file in user_log_dir.glob("*_history.json"):
            user_name = history_file.name.replace("_history.json", "")
            try:
                with open(history_file, "r", encoding="utf-8") as f:
                    h_data = json.load(f)
                for rel_path in h_data.get("annotated_files", []):
                    if rel_path not in all_valid_files:
                        continue # 忽略已经被过滤掉的非 round_robin 文件
                    abs_path = BASE_DIR / rel_path
                    with open(abs_path, "r", encoding="utf-8") as lf:
                        log_data = json.load(lf)
                    
                    if "3port" in log_data and user_name in log_data["3port"]:
                        _apply_elo_update(log_data["3port"][user_name], elo_state)
            except Exception:
                pass
                
    with open(GLOBAL_ELO_FILE, "w", encoding="utf-8") as f:
        json.dump(elo_state, f, ensure_ascii=False, indent=2)

def update_and_save_elo(ranking_results: list[dict]):
    """增量更新并保存 ELO 分数"""
    lock_path = str(GLOBAL_ELO_FILE) + ".lock"
    with FileLock(lock_path, timeout=10):
        if GLOBAL_ELO_FILE.exists():
            with open(GLOBAL_ELO_FILE, "r", encoding="utf-8") as f:
                elo_state = json.load(f)
        else:
            elo_state = {"models": {}}
            
        _apply_elo_update(ranking_results, elo_state)
        
        with open(GLOBAL_ELO_FILE, "w", encoding="utf-8") as f:
            json.dump(elo_state, f, ensure_ascii=False, indent=2)

# ═══════════════════════════════════════════════════════════════
# 工具函数 (脱敏与校验)
# ═══════════════════════════════════════════════════════════════

def _extract_config_key(agent: dict) -> str:
    if "config_key" in agent:
        return agent["config_key"]
    if "name" in agent:
        return agent["name"]
    return str(agent.get("position", agent["agent_id"]))

def build_anonymization_map(log_data: dict, seed: str) -> dict[int, dict]:
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
    rank_values = list(rankings.values())
    expected = set(range(1, n + 1))
    if set(rank_values) != expected:
        return False, "排名不满足严格全序"
    return True, ""

def save_annotation(rel_path: str, user_name: str, ranking_results: list[dict]):
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
        name_input = st.text_input("User Name", placeholder="例如：Senhao", label_visibility="collapsed")
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
# 2. 侧边栏：核心智能排序与文件分配
# ──────────────────────────────────────

user_name: str = st.session_state.user_name

# 加载并过滤数据：只允许 round_robin
_all_collected_files = collect_log_files()
_file_meta_dict = get_all_files_metadata(_all_collected_files)
all_valid_files = [f for f in _all_collected_files if _file_meta_dict.get(f, {}).get("mode") == "round_robin"]

# 初始化 ELO 历史记录（如需要补偿计算）
init_global_elo_if_needed(all_valid_files)

# 统计全局状态以支持智能推荐
global_file_counts, global_annotated_set = get_global_annotation_stats()
annotated_set = load_user_history(user_name)

# 统计 Topic 频率 (在已标注数据中的出现频次，用于保持平衡)
topic_counts = collections.defaultdict(int)
for f, count in global_file_counts.items():
    topic = _file_meta_dict.get(f, {}).get("topic", "Unknown")
    topic_counts[topic] += count

pending_files = [f for f in all_valid_files if f not in annotated_set]

# 核心排序算法：多维度计分优先级排序
def smart_pending_sort_key(filepath):
    meta = _file_meta_dict.get(filepath, {})
    topic = meta.get("topic", "Unknown")
    
    # 优先级1: 该文件全局被标注的总次数 (越少越先推荐)
    f_count = global_file_counts.get(filepath, 0)
    # 优先级2: 该 Topic 全局被标注的总次数 (越少越先推荐，实现均衡)
    t_count = topic_counts.get(topic, 0)
    # 优先级3: 稳定的 Hash 散列 (相同次数下随机打散，防止扎堆同一个文件夹)
    hash_val = int(hashlib.md5(filepath.encode('utf-8')).hexdigest()[:8], 16)
    
    return (f_count, t_count, hash_val)

pending_files.sort(key=smart_pending_sort_key)

with st.sidebar:
    st.markdown(f"### 👤 {user_name}")
    if st.button("退出登录", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

    st.divider()
    st.metric("已标注 (限定 RoundRobin)", f"{len(annotated_set)} 个")
    st.metric("待标注 (限定 RoundRobin)", f"{len(pending_files)} 个")
    st.divider()

    if not pending_files:
        st.success("🎉 所有文件已标注完成！")
        st.title("📋 头脑风暴第三方标注系统")
        st.info("当前没有待标注的 round_robin 文件。请等待新的实验数据。")
        st.stop()

    def format_file_option(x):
        base_name = os.path.basename(x) if "/" in x else x
        topic = _file_meta_dict.get(x, {}).get("topic", "未知主题")[:15] + "..." # 截断显示
        
        prefix = "[他人已标]" if x in global_annotated_set else "[新数据]"
        return f"{prefix} [{topic}] {base_name}"

    selected_file = st.selectbox("选择待标注文件 (已按缺口智能排序)", pending_files, format_func=format_file_option)

# ──────────────────────────────────────
# 3. 数据加载与脱敏
# ──────────────────────────────────────

log_data = load_log_data(selected_file)
metadata = log_data["metadata"]

seed = f"{selected_file}||{user_name}"
anon_map = build_anonymization_map(log_data, seed)

st.title("📋 头脑风暴第三方标注系统")

st.markdown(f"**当前文件：** `{selected_file}`")
meta_cols = st.columns(3)
meta_cols[0].metric("讨论模式", MODE_LABELS.get(metadata["mode"], metadata["mode"]))
meta_cols[1].metric("总轮数", metadata["max_rounds"])
meta_cols[2].metric("参与人数", metadata["total_agents"])
st.info(f"**讨论话题：** {metadata.get('topic', 'N/A')}")

st.divider()

# ──────────────────────────────────────
# 4. 对局内容展示（脱敏 + 并发翻译）
# ──────────────────────────────────────

st.markdown("### 💬 讨论记录")
global_history = log_data["global_history"]
current_round = 0
translated_history = get_translated_history(global_history, anon_map, selected_file)

for idx, entry in enumerate(global_history):
    if entry["round"] != current_round:
        current_round = entry["round"]
        st.markdown(f"#### 第 {current_round} 轮")

    orig_id = entry["agent_id"]
    info = anon_map[orig_id]
    display_name = info["display_name"]
    avatar = AGENT_AVATARS[(info["display_idx"] - 1) % len(AGENT_AVATARS)]

    content = anonymize_content(entry["content"], anon_map)
    zh_translation = translated_history.get(idx, "")

    with st.chat_message(name=display_name, avatar=avatar):
        st.markdown(f"**{display_name}**")
        st.markdown(content)
        if zh_translation:
            st.caption("以下为系统自动翻译的内容：")
            st.info(zh_translation)

st.divider()

# ──────────────────────────────────────
# 5. 排序标注模块（联动交换机制）
# ──────────────────────────────────────

display_agents = sorted(anon_map.items(), key=lambda x: x[1]["display_idx"])
n_agents = len(display_agents)
rank_options = list(range(1, n_agents + 1))
_eval_agent_names = [info["display_name"] for _, info in display_agents]

def _init_eval_ranking_state():
    if "eval_ranking_selections" not in st.session_state or st.session_state.get("_eval_ranking_file") != selected_file:
        st.session_state._eval_ranking_file = selected_file
        st.session_state.eval_ranking_selections = {name: idx + 1 for idx, name in enumerate(_eval_agent_names)}

def _on_eval_rank_change(agent_name: str):
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
        st.error(f"⚠️ 排名无效！请为每位 Agent 分配从 1 到 {n_agents} 的不重复名次。")
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
            # 1. 写入原数据 json 3port 字段
            save_annotation(selected_file, user_name, ranking_results)
            # 2. ELO 分数增量更新写入 user_log
            update_and_save_elo(ranking_results)
            # 3. 记录已标注集合
            annotated_set.add(selected_file)
            save_user_history(user_name, annotated_set)
            
            del st.session_state["eval_ranking_selections"]
            del st.session_state["_eval_ranking_file"]
            st.success("✅ 标注已提交且 ELO 分数已更新！页面即将刷新...")
            st.rerun()
        except Exception as e:
            st.error(f"写入标注或 ELO 更新失败：{e}")