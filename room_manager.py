"""全局房间状态管理器 —— 跨 Streamlit Session 共享讨论房间。

Streamlit 单进程多线程模型下，模块级变量天然跨 Session 共享。
本模块通过全局字典 + threading.Lock 提供线程安全的房间生命周期管理。
"""

from __future__ import annotations

import json
import os
import random
import string
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime

from agents.agent_base import EnvState
from agents.agent_human import AgentHuman
from agents.agent_llm import AgentLLM
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

_LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "log_human_2")


@dataclass
class RoomState:
    env: object
    mode: str
    topic: str
    human_seats: list[int]
    claimed_seats: dict[int, str] = field(default_factory=dict)
    leader_ids: list[int] = field(default_factory=list)
    llm_lock: threading.Lock = field(default_factory=threading.Lock)
    created_at: float = field(default_factory=time.time)
    rankings_submitted: dict[int, list[dict]] = field(default_factory=dict)
    discussion_saved: bool = False
    log_path: str | None = None
    initial_advance_done: bool = False


_rooms: dict[str, RoomState] = {}
_rooms_lock = threading.Lock()


def _generate_room_id() -> str:
    """生成不重复的 4 位数字房间号。"""
    while True:
        rid = "".join(random.choices(string.digits, k=4))
        if rid not in _rooms:
            return rid


def _load_pool():
    try:
        return load_llm_config()
    except FileNotFoundError:
        return {}


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


def _build_agents(num_humans: int, num_llm: int, human_roles: list[str]) -> list:
    """构建 Agent 列表：多个人类 + 自动抽取的 LLM。

    agent_id 由 EnvBase 构造函数在 shuffle 后根据列表顺序动态分配。
    """
    pool = _load_pool()
    agents = []

    for i in range(num_humans):
        role = human_roles[i] if i < len(human_roles) else "人类专家"
        agents.append(AgentHuman(role_background=role))

    sampled_keys = sample_llm_keys(pool, num_llm)
    expert_keys = list(EXPERTS.keys())
    random.shuffle(expert_keys)

    for i, config_key in enumerate(sampled_keys):
        agent = build_agent_from_config(config_key, pool)
        if not agent.role_background:
            agent.role_background = EXPERTS[expert_keys[i % len(expert_keys)]]
        agents.append(agent)

    return agents


def create_room(
    mode: str,
    topic: str,
    max_rounds: int,
    num_humans: int,
    num_llm: int,
    human_roles: list[str] | None = None,
) -> str:
    """创建房间，实例化 env 并初始化，返回房间号。

    LLM Agent 从 pool 中自动盲抽，不再接受手动配置。
    """
    with _rooms_lock:
        room_id = _generate_room_id()

        agents = _build_agents(num_humans, num_llm, human_roles or [])
        random.shuffle(agents)

        env_cls = ENV_MAP[mode]
        if mode == "leader_worker":
            leader_ids = [i + 1 for i, a in enumerate(agents) if a.is_human]
            env = env_cls(
                agents=agents,
                topic=topic,
                max_rounds=max_rounds,
                leader_ids=leader_ids,
                log_dir=_LOG_DIR,
            )
        else:
            env = env_cls(
                agents=agents,
                topic=topic,
                max_rounds=max_rounds,
                log_dir=_LOG_DIR,
            )
        env.init()

        human_seats = [a.agent_id for a in agents if a.is_human]

        room = RoomState(
            env=env,
            mode=mode,
            topic=topic,
            human_seats=human_seats,
            leader_ids=list(leader_ids) if mode == "leader_worker" else [],
        )
        _rooms[room_id] = room
        return room_id


def get_room(room_id: str) -> RoomState | None:
    return _rooms.get(room_id)


def join_room(room_id: str) -> RoomState | None:
    return _rooms.get(room_id)


def claim_seat(room_id: str, agent_id: int, session_id: str) -> bool:
    """认领座位。成功返回 True，座位已被占或不合法返回 False。"""
    with _rooms_lock:
        room = _rooms.get(room_id)
        if room is None:
            return False
        if agent_id not in room.human_seats:
            return False
        if agent_id in room.claimed_seats:
            return room.claimed_seats[agent_id] == session_id
        room.claimed_seats[agent_id] = session_id
        return True


def is_room_ready(room_id: str) -> bool:
    room = _rooms.get(room_id)
    if room is None:
        return False
    return len(room.claimed_seats) >= len(room.human_seats)


def get_unclaimed_seats(room_id: str) -> list[int]:
    room = _rooms.get(room_id)
    if room is None:
        return []
    return [s for s in room.human_seats if s not in room.claimed_seats]


def auto_advance_llm(env) -> None:
    """连续推进 LLM 发言，直到遇到人类回合或结束。"""
    while env.state not in (EnvState.WAITING_HUMAN, EnvState.FINISHED):
        env.step()


def submit_ranking(room_id: str, agent_id: int, ranking_data: list[dict]) -> None:
    with _rooms_lock:
        room = _rooms.get(room_id)
        if room is None:
            return
        room.rankings_submitted[agent_id] = ranking_data


def all_rankings_submitted(room_id: str) -> bool:
    room = _rooms.get(room_id)
    if room is None:
        return False
    return all(s in room.rankings_submitted for s in room.human_seats)


def save_and_get_log(room_id: str) -> str | None:
    """保存讨论日志到 log_human_2/，返回文件路径。线程安全，仅保存一次。"""
    with _rooms_lock:
        room = _rooms.get(room_id)
        if room is None:
            return None
        if room.discussion_saved:
            return room.log_path

        env = room.env
        os.makedirs(_LOG_DIR, exist_ok=True)

        human_count = len(room.human_seats)
        ts = datetime.now().strftime("%Y%m%d%H%M")
        filename = f"{room.mode}_{len(env.agents)}_{human_count}_{ts}.json"
        path = os.path.join(_LOG_DIR, filename)

        final_messages = {}
        for agent in env.agents:
            msgs = env.build_messages_for_agent(agent)
            final_messages[str(agent.agent_id)] = msgs

        position_map = [
            {
                "position": a.agent_id,
                "config_key": a.config_key,
                "type": "human" if a.is_human else "llm",
                "model": getattr(a, "inference_config", {}).get("model", "human"),
            }
            for a in env.agents
        ]

        log_data = {
            "metadata": {
                "mode": room.mode,
                "topic": room.topic,
                "max_rounds": env.max_rounds,
                "total_agents": len(env.agents),
                "human_count": human_count,
                "timestamp": datetime.now().isoformat(),
                "agents": [a.get_agent_info() for a in env.agents],
                "position_map": position_map,
                "room_id": room_id,
            },
            "global_history": env.global_history,
            "final_messages": final_messages,
        }

        if room.rankings_submitted:
            rankings = {}
            for human_aid, rdata in room.rankings_submitted.items():
                rankings[str(human_aid)] = rdata
            log_data["final_rankings"] = rankings

        with open(path, "w", encoding="utf-8") as f:
            json.dump(log_data, f, ensure_ascii=False, indent=2)

        room.discussion_saved = True
        room.log_path = path
        return path
