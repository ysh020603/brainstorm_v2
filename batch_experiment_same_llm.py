#!/usr/bin/env python3
"""多模型单体 AI 讨论组批量实验脚本（Same LLM）。

设计目标
========
针对预设的 Topics，对于模型池中的 **每一个模型**：
让该模型同时扮演 4 个参赛者（Agent），运行 Round Robin 格式的讨论。
每个 Topic 下，每个模型都要执行 4 局自我讨论。

核心保证
--------
1) 断点续传：状态写入新的 STATE_FILE，崩溃后再次运行可从未完成处继续。
2) 失败重试：遇到 API 错误会自动进行指数退避重试，失败达到上限才跳过。
3) 日志隔离：使用全新的目录存放本次实验日志。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import traceback
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents.agent_base import EnvState
from envs.round_robin import RoundRobin
from tools.config_loader import build_agent_from_config, load_llm_config


# ============================================================
# 一、全局配置与数据定义
# ============================================================

# 1) Topic 列表（硬编码）。日常调试时 `#` 注释掉不想跑的 topic 即可。
TOPICS: list[str] = [
    "Features for a next-generation smartphone.",
    "Ways to reduce food waste at home.",
    "Ideas for making public transport more enjoyable.",
    "Solutions to reduce plastic pollution in oceans.",
    "Ideas to protect wildlife in cities.",
    "Creative uses for drones in everyday life.",
]

# 2) 实验参数
FORMAT: str = "round_robin"
TARGET_RUNS: int = 4        # 每个 (Topic, Model) 组合要跑的局数
GROUP_SIZE: int = 4         # 每局参赛 Agent 数量（全由同一个模型驱动）
MAX_ROUNDS: int = 4         # 单局讨论的最大轮数

# 3) 路径与重试参数 (新建专属日志目录)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
LOG_ROOT = os.path.join(PROJECT_ROOT, "log_same_llm", "log")
STATE_FILE = os.path.join(PROJECT_ROOT, "log_same_llm", "batch_state.json")
DEFAULT_CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "llm_config.json")

MAX_GAME_RETRIES: int = 3       # 单局最多重试次数
RETRY_BACKOFF_SEC: float = 5.0  # 重试基准等待秒数（指数退避）


# ============================================================
# 二、工具：Topic slug / 状态读写 / 日志格式化
# ============================================================

def _topic_slug(topic: str, max_len: int = 40) -> str:
    """将 topic 转换为安全的目录名（去标点、空格替换）。"""
    cleaned = re.sub(r"[^\w\u4e00-\u9fff]+", "_", topic).strip("_")
    return cleaned[:max_len] or "topic"


def _now() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _print(msg: str) -> None:
    print(f"[{_now()}] {msg}", flush=True)


def _load_state() -> dict:
    if not os.path.exists(STATE_FILE):
        return {"runs": {}}
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        _print(f"⚠️ 状态文件损坏，将从头开始：{STATE_FILE}")
        return {"runs": {}}


def _save_state(state: dict) -> None:
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    tmp = STATE_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    os.replace(tmp, STATE_FILE)


def _run_key(topic: str, model_key: str) -> str:
    """状态文件中 (Topic, Model) 的唯一键。"""
    return f"{_topic_slug(topic, max_len=80)}::{model_key}"


# ============================================================
# 三、执行单局（含重试）
# ============================================================

def _run_single_game(
    topic: str,
    model_keys: list[str],
    pool: dict,
    log_dir: str,
) -> tuple[str | None, Exception | None]:
    """执行单场讨论。返回 (log_path, error)。"""
    try:
        # 为同一模型实例化 4 个不同的 Agent 对象
        agents = [build_agent_from_config(k, pool) for k in model_keys]
        env = RoundRobin(
            agents=agents,
            topic=topic,
            max_rounds=MAX_ROUNDS,
            log_dir=log_dir,
        )
        env.init()

        step_count = 0
        while env.state != EnvState.FINISHED:
            env.step()
            step_count += 1

        log_path = env.save_log()
        return log_path, None
    except Exception as exc:  # noqa: BLE001
        return None, exc


def _attempt_with_retries(
    topic: str,
    model_keys: list[str],
    pool: dict,
    log_dir: str,
) -> str | None:
    """单局重试（指数退避）。"""
    last_err: Exception | None = None
    for attempt in range(1, MAX_GAME_RETRIES + 1):
        log_path, err = _run_single_game(topic, model_keys, pool, log_dir)
        if log_path is not None:
            return log_path
        last_err = err
        wait = RETRY_BACKOFF_SEC * (2 ** (attempt - 1))
        _print(
            f"  ⚠️ 第 {attempt}/{MAX_GAME_RETRIES} 次尝试失败："
            f"{type(err).__name__}: {err}；{wait:.1f}s 后重试。"
        )
        time.sleep(wait)
        
    if last_err is not None:
        _print(f"  ❌ 重试用尽，最后异常栈：")
        traceback.print_exception(type(last_err), last_err, last_err.__traceback__)
    return None


# ============================================================
# 四、调度循环
# ============================================================

def _run_model_on_topic(
    topic: str,
    model_key: str,
    pool: dict,
    state: dict,
    dry_run: bool = False,
) -> None:
    """运行特定 Topic 和特定模型的所有局数。"""
    key = _run_key(topic, model_key)
    run_state = state["runs"].setdefault(
        key,
        {
            "topic": topic,
            "model": model_key,
            "completed": 0,
            "log_paths": [],
        },
    )

    completed = run_state["completed"]
    _print(
        f"▶ Topic={topic!r} | Model={model_key} | 目标 {TARGET_RUNS} 局 | 已完成 {completed} 局"
    )

    log_dir = os.path.join(LOG_ROOT, _topic_slug(topic), model_key)
    if not dry_run:
        os.makedirs(log_dir, exist_ok=True)

    # 需要用 4 个该模型组成讨论组
    group = [model_key] * GROUP_SIZE

    for i in range(completed + 1, TARGET_RUNS + 1):
        if dry_run:
            _print(f"   [ ] 第 {i}/{TARGET_RUNS} 局 (干跑)")
            continue

        _print(f"   ▷ 第 {i}/{TARGET_RUNS} 局 | 模型: {group}")
        log_path = _attempt_with_retries(topic, group, pool, log_dir)

        if log_path is None:
            _print(f"   ✗ 第 {i} 局彻底失败，跳过。")
            continue

        run_state["completed"] = i
        run_state["log_paths"].append(log_path)
        _save_state(state)
        _print(f"   ✓ 第 {i} 局完成 → {log_path}")


# ============================================================
# 五、主入口
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="单模型自我讨论组批量实验")
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help=f"llm_config.json 路径（默认: {DEFAULT_CONFIG_PATH}）",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅打印调度计划，不实际触发 API 调用",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="忽略并清空已有状态，从头开始（删除 STATE_FILE）",
    )
    parser.add_argument(
        "--only-topic",
        type=int,
        default=None,
        help="只跑 TOPICS 列表中第 N 个 Topic（1-based 索引），便于单测",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    pool = load_llm_config(args.config)
    if not pool:
        _print(f"❌ 模型池为空，请检查 {args.config}")
        return 2

    active_topics = [t for t in TOPICS if t and t.strip()]
    if not active_topics:
        _print("❌ TOPICS 列表为空（或全部被注释），无事可做。")
        return 2

    if args.only_topic is not None:
        idx = args.only_topic - 1
        if not (0 <= idx < len(active_topics)):
            _print(f"❌ --only-topic={args.only_topic} 越界（共 {len(active_topics)} 个有效 Topic）")
            return 2
        active_topics = [active_topics[idx]]

    if args.reset and os.path.exists(STATE_FILE):
        os.remove(STATE_FILE)
        _print(f"🗑  已删除状态文件 {STATE_FILE}")

    state = _load_state()

    _print("=" * 60)
    _print(f"模型数量 M = {len(pool)}")
    _print(f"有效 Topic 数量 T = {len(active_topics)}")
    _print(f"讨论模式 = {FORMAT}")
    _print(f"目标总局数 = {len(active_topics) * len(pool) * TARGET_RUNS} 局")
    _print(f"状态文件: {STATE_FILE}")
    _print(f"日志根目录: {LOG_ROOT}")
    _print("=" * 60)

    for t_idx, topic in enumerate(active_topics, 1):
        _print(f"\n###### Topic {t_idx}/{len(active_topics)}: {topic!r} ######")
        for model_key in sorted(pool.keys()):
            _run_model_on_topic(topic, model_key, pool, state, dry_run=args.dry_run)

    _print("\n" + "=" * 60)
    if args.dry_run:
        _print("✓ 干跑结束（未触发任何 API 调用）。")
        return 0

    _print("🎉 全部任务结束。最终配额追踪：")
    for key, run_state in state["runs"].items():
        completed = run_state.get('completed', 0)
        flag = "OK" if completed == TARGET_RUNS else f"⚠ 未满: {completed}/{TARGET_RUNS}"
        _print(f"  {key} | {flag}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())