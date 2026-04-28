#!/usr/bin/env python3
"""多模型纯 AI 讨论组批量实验（leader_worker 2-Leader / 2-Worker 专用版）。

与通用版 ``batch_experiment.py`` 的差异
======================================
1) 只跑 ``leader_worker`` 一种讨论形式；其它 format 全部不参与。
2) Leader/Worker 比例固定为 ``2 : 2``（即 ``leader_ids = [1, 2]``，
   每局 4 个 Agent 中 position=1、2 担任 Leader，position=3、4 担任 Worker）。
3) 日志保存在 ``<LOG_ROOT>/<topic_slug>/leader_worker_22/`` 下，
   与原 ``leader_worker``（默认 1 Leader / 3 Worker）的结果分目录互不污染。
4) 状态文件独立为 ``batch_experiment_leader_worker_22_state.json``，
   断点续传、配额追踪与原脚本完全隔离。

其余所有保证（精确配额、防死锁调度、断点续传、重试与回滚）均与
``batch_experiment.py`` 保持一致。

使用示例
--------
    # 默认：跑全部 6 个 Topic × leader_worker(2-2)
    python batch_experiment_leader_worker_22.py

    # 仅干跑（dry-run）查看调度计划
    python batch_experiment_leader_worker_22.py --dry-run

    # 重置状态从头跑
    python batch_experiment_leader_worker_22.py --reset

    # 只跑 TOPICS 中的某一个 Topic（1-based）
    python batch_experiment_leader_worker_22.py --only-topic 3
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import traceback
from datetime import datetime
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents.agent_base import EnvState
from envs.leader_worker import LeaderWorker
from tools.config_loader import build_agent_from_config, load_llm_config


# ============================================================
# 一、全局配置与数据定义
# ============================================================

# 1) Topic 列表（与 batch_experiment.py 一致；调试时注释掉不想跑的即可）
TOPICS: list[str] = [
    "Features for a next-generation smartphone.",
    "Ways to reduce food waste at home.",
    "Ideas for making public transport more enjoyable.",
    "Solutions to reduce plastic pollution in oceans.",
    "Ideas to protect wildlife in cities.",
    "Creative uses for drones in everyday life.",
]

# 2) 本脚本固定 Format / Env / 子目录名
FORMAT_NAME: str = "leader_worker"          # Env 类型仍是 leader_worker
LOG_SUBDIR_NAME: str = "leader_worker_22"   # 但日志写入这个独立子目录
ENV_CLS: Any = LeaderWorker

# 3) 实验参数
TARGET_PER_MODEL: int = 4   # 每个模型在该 (Topic, Format) 上的出场次数
GROUP_SIZE: int = 4         # 每局参赛模型数
MAX_ROUNDS: int = 4         # 单局讨论的最大轮数

# 关键差异：2 Leader + 2 Worker
LEADER_IDS: list[int] = [1, 2]

# 4) 路径与重试参数
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
LOG_ROOT = os.path.join(PROJECT_ROOT, "log_test", "log")
STATE_FILE = os.path.join(
    PROJECT_ROOT, "log_test", "batch_experiment_leader_worker_22_state.json"
)
DEFAULT_CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "llm_config.json")

MAX_GAME_RETRIES: int = 3       # 单局最多重试次数
MAX_QUEUE_REINSERTS: int = 5    # 单局允许被重新入队的最大次数
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
        _print(f"⚠️  状态文件损坏，将从头开始：{STATE_FILE}")
        return {"runs": {}}


def _save_state(state: dict) -> None:
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    tmp = STATE_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    os.replace(tmp, STATE_FILE)


def _run_key(topic: str) -> str:
    """状态文件中 (Topic, Format=leader_worker_22) 的唯一键。"""
    return f"{_topic_slug(topic, max_len=80)}::{LOG_SUBDIR_NAME}"


# ============================================================
# 三、防死锁调度算法（与通用版一致）
# ============================================================

def _tiebreak_hash(model: str, game_idx: int, salt: str = "brainstorm-v2-lw22") -> int:
    """确定性次级排序键：基于 hashlib，让每局的 tie-break 顺序不同，
    从而促使搭档组合在不同局中尽量分散。"""
    digest = hashlib.md5(f"{salt}|{game_idx}|{model}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def generate_schedule(
    models: list[str],
    target: int = TARGET_PER_MODEL,
    group_size: int = GROUP_SIZE,
) -> list[list[str]]:
    """生成 (Topic, Format) 下完整的对局名单。

    采用 "剩余配额降序 + 局号哈希混淆" 的贪心：
    - 主键：剩余配额降序（保证 N ≥ group_size 时不会死锁）
    - 次键：每局用 hashlib(game_idx + model) 旋转，让搭档组合在不同局充分洗牌
    """
    if len(models) < group_size:
        raise ValueError(
            f"模型池仅 {len(models)} 个，少于每局所需的 {group_size} 个，无法成局。"
        )
    total_slots = target * len(models)
    if total_slots % group_size != 0:
        raise ValueError(
            f"target({target}) × N({len(models)}) = {total_slots} 不能被 "
            f"group_size({group_size}) 整除，无法精确分配。"
        )

    remaining: dict[str, int] = {m: target for m in models}
    total_games = total_slots // group_size
    schedule: list[list[str]] = []

    for game_idx in range(total_games):
        candidates = sorted(
            (m for m, r in remaining.items() if r > 0),
            key=lambda m: (-remaining[m], _tiebreak_hash(m, game_idx), m),
        )
        if len(candidates) < group_size:
            raise RuntimeError(
                f"调度死锁：第 {game_idx + 1} 局只剩 {len(candidates)} 个可选模型。"
                f" remaining={remaining}"
            )
        picked = candidates[:group_size]
        schedule.append(picked)
        for m in picked:
            remaining[m] -= 1

    leftover = {m: r for m, r in remaining.items() if r != 0}
    if leftover:
        raise RuntimeError(f"调度结束后仍有未清零的配额：{leftover}")
    return schedule


# ============================================================
# 四、执行单局（含重试 + 回滚）
# ============================================================

def _run_single_game(
    topic: str,
    model_keys: list[str],
    pool: dict,
    log_dir: str,
) -> tuple[str | None, Exception | None]:
    """执行单场 leader_worker(2-2) 讨论。返回 (log_path, error)。失败时 log_path=None。"""
    try:
        agents = [build_agent_from_config(k, pool) for k in model_keys]
        env = ENV_CLS(
            agents=agents,
            topic=topic,
            max_rounds=MAX_ROUNDS,
            log_dir=log_dir,
            leader_ids=LEADER_IDS,
        )
        env.init()

        while env.state != EnvState.FINISHED:
            env.step()

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
    """单局重试（指数退避）。完全失败返回 None，调用方决定是否回滚。"""
    last_err: Exception | None = None
    for attempt in range(1, MAX_GAME_RETRIES + 1):
        log_path, err = _run_single_game(topic, model_keys, pool, log_dir)
        if log_path is not None:
            return log_path
        last_err = err
        wait = RETRY_BACKOFF_SEC * (2 ** (attempt - 1))
        _print(
            f"  ⚠️  第 {attempt}/{MAX_GAME_RETRIES} 次尝试失败："
            f"{type(err).__name__}: {err}；{wait:.1f}s 后重试。"
        )
        time.sleep(wait)
    if last_err is not None:
        _print(f"  ❌ 重试用尽，最后异常栈：")
        traceback.print_exception(type(last_err), last_err, last_err.__traceback__)
    return None


# ============================================================
# 五、(Topic) 调度循环（Format 固定为 leader_worker_22）
# ============================================================

def _run_topic(
    topic: str,
    pool: dict,
    state: dict,
    dry_run: bool = False,
) -> None:
    models = sorted(pool.keys())
    schedule = generate_schedule(models)
    n_games = len(schedule)

    key = _run_key(topic)
    run_state = state["runs"].setdefault(
        key,
        {
            "topic": topic,
            "format": LOG_SUBDIR_NAME,
            "leader_ids": LEADER_IDS,
            "models_snapshot": models,
            "completed": 0,
            "tracker": {m: 0 for m in models},
            "log_paths": [],
        },
    )

    if run_state.get("models_snapshot") != models:
        _print(
            f"⚠️  模型池发生变化 (key={key})。原: {run_state.get('models_snapshot')} "
            f"现: {models}。该组合将重置。"
        )
        run_state["models_snapshot"] = models
        run_state["completed"] = 0
        run_state["tracker"] = {m: 0 for m in models}
        run_state["log_paths"] = []

    completed = run_state["completed"]
    _print(
        f"▶ Topic={topic!r} | Format={LOG_SUBDIR_NAME} (leaders={LEADER_IDS}) "
        f"| 计划 {n_games} 局 | 已完成 {completed} 局"
    )

    if dry_run:
        for idx, group in enumerate(schedule, 1):
            mark = "✓" if idx <= completed else " "
            _print(f"   [{mark}] 第 {idx:>2}/{n_games} 局: {group}")
        return

    log_dir = os.path.join(LOG_ROOT, _topic_slug(topic), LOG_SUBDIR_NAME)
    os.makedirs(log_dir, exist_ok=True)

    queue: list[tuple[int, list[str], int]] = [
        (idx, group, 0) for idx, group in enumerate(schedule, 1) if idx > completed
    ]

    while queue:
        idx, group, reinserts = queue.pop(0)
        _print(f"   ▷ 第 {idx:>2}/{n_games} 局 | 模型: {group}")
        log_path = _attempt_with_retries(topic, group, pool, log_dir)

        if log_path is None:
            if reinserts >= MAX_QUEUE_REINSERTS:
                _print(
                    f"   ✗ 第 {idx} 局重新入队 {reinserts} 次仍失败，跳过。"
                    f" 该组合 (Topic={topic!r}, Format={LOG_SUBDIR_NAME}) 配额不再精确。"
                )
                continue
            _print(
                f"   ↩ 第 {idx} 局回滚（不扣配额），重新入队（第 {reinserts + 1} 次）。"
            )
            queue.append((idx, group, reinserts + 1))
            time.sleep(RETRY_BACKOFF_SEC * (2 ** reinserts))
            continue

        for m in group:
            run_state["tracker"][m] = run_state["tracker"].get(m, 0) + 1
        run_state["completed"] = max(run_state["completed"], idx)
        run_state["log_paths"].append(log_path)
        _save_state(state)
        _print(f"   ✓ 第 {idx} 局完成 → {log_path}")


# ============================================================
# 六、主入口
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="多模型纯 AI 讨论组批量实验（leader_worker 2-Leader / 2-Worker 专用）"
    )
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

    if GROUP_SIZE - len(LEADER_IDS) <= 0 or len(LEADER_IDS) <= 0:
        _print(
            f"❌ LEADER_IDS={LEADER_IDS} 与 GROUP_SIZE={GROUP_SIZE} 不构成 2-2 配置。"
        )
        return 2

    active_topics = [t for t in TOPICS if t and t.strip()]
    if not active_topics:
        _print("❌ TOPICS 列表为空（或全部被注释），无事可做。")
        return 2

    if args.only_topic is not None:
        idx = args.only_topic - 1
        if not (0 <= idx < len(active_topics)):
            _print(
                f"❌ --only-topic={args.only_topic} 越界（共 {len(active_topics)} 个有效 Topic）"
            )
            return 2
        active_topics = [active_topics[idx]]

    if args.reset and os.path.exists(STATE_FILE):
        os.remove(STATE_FILE)
        _print(f"🗑  已删除状态文件 {STATE_FILE}")

    state = _load_state()

    _print("=" * 60)
    _print(f"模型池规模 N = {len(pool)}（每个 Topic 共 {len(pool)} 局）")
    _print(f"有效 Topic: {len(active_topics)} 个")
    _print(f"Format: {LOG_SUBDIR_NAME}（基于 {FORMAT_NAME}，leader_ids={LEADER_IDS}）")
    _print(f"目标总局数: {len(active_topics) * len(pool)}")
    _print(f"状态文件: {STATE_FILE}")
    _print(f"日志根目录: {LOG_ROOT}（每 Topic 写入 .../<topic>/{LOG_SUBDIR_NAME}/）")
    _print("=" * 60)

    for t_idx, topic in enumerate(active_topics, 1):
        _print(f"\n###### Topic {t_idx}/{len(active_topics)}: {topic!r} ######")
        _run_topic(topic, pool, state, dry_run=args.dry_run)

    _print("\n" + "=" * 60)
    if args.dry_run:
        _print("✓ 干跑结束（未触发任何 API 调用）。")
        return 0

    _print("🎉 全部任务结束。最终配额追踪：")
    for key, run_state in state["runs"].items():
        tracker = run_state.get("tracker", {})
        bad = {m: c for m, c in tracker.items() if c != TARGET_PER_MODEL}
        flag = "OK" if not bad else f"⚠ 偏差: {bad}"
        _print(f"  {key} | completed={run_state.get('completed', 0)} | {flag}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
