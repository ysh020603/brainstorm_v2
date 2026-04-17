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
    agent_configs: list[dict] = field(default_factory=list)
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


def _build_agents(agent_configs: list[dict], human_seats: list[int]) -> list:
    pool = _load_pool()
    model_keys = list(pool.keys())

    agents = []
    for i, cfg in enumerate(agent_configs):
        agent_id = i + 1
        if agent_id in human_seats:
            agents.append(AgentHuman(
                agent_id=agent_id,
                role_background=cfg.get("role", "人类专家"),
            ))
        else:
            model_key = cfg.get("model_key")
            if model_key and model_key in pool:
                agent = build_agent_from_config(agent_id, model_key, pool)
                if cfg.get("role"):
                    agent.role_background = cfg["role"]
            elif model_keys:
                agent = build_agent_from_config(agent_id, model_keys[0], pool)
                if cfg.get("role"):
                    agent.role_background = cfg["role"]
            else:
                agent = AgentLLM(
                    agent_id=agent_id,
                    role_background=cfg.get("role", ""),
                    api_config={"api_key": "EMPTY", "base_url": "http://localhost:8000/v1"},
                    inference_config={"model": "unknown", "temperature": 0.7},
                )
            agents.append(agent)
    return agents


def create_room(
    mode: str,
    topic: str,
    max_rounds: int,
    agent_configs: list[dict],
    human_seats: list[int],
    leader_ids: list[int] | None = None,
) -> str:
    """创建房间，实例化 env 并初始化，返回房间号。"""
    with _rooms_lock:
        room_id = _generate_room_id()

        agents = _build_agents(agent_configs, human_seats)
        random.shuffle(agents)
        env_cls = ENV_MAP[mode]
        if mode == "leader_worker":
            env = env_cls(
                agents=agents,
                topic=topic,
                max_rounds=max_rounds,
                leader_ids=leader_ids or [],
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

        room = RoomState(
            env=env,
            mode=mode,
            topic=topic,
            human_seats=list(human_seats),
            agent_configs=agent_configs,
            leader_ids=leader_ids or [],
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
                "position": a.position,
                "agent_id": a.agent_id,
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
            log_data["round_rankings"] = rankings

        with open(path, "w", encoding="utf-8") as f:
            json.dump(log_data, f, ensure_ascii=False, indent=2)

        room.discussion_saved = True
        room.log_path = path
        return path
