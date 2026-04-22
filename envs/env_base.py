from __future__ import annotations

import json
import os
from copy import deepcopy
from datetime import datetime

from agents.agent_base import AgentBase, EnvState
from prompts.system_prompts import build_system_prompt
from prompts import instruct_prompts as IP


class EnvBase:
    """讨论环境基类，采用步进式 (step-by-step) 状态机驱动。

    构造函数负责根据 agents 列表的顺序动态分配 agent_id（从 1 开始递增），
    该 agent_id 同时作为展示序号和唯一标识，不再依赖构造 Agent 时的静态绑定。
    """

    mode: str = "base"

    def __init__(
        self,
        agents: list[AgentBase],
        topic: str,
        max_rounds: int,
        log_dir: str | None = None,
    ):
        self.agents = agents
        self.topic = topic
        self.max_rounds = max_rounds
        self.log_dir = log_dir

        self.global_history: list[dict] = []
        self.current_round: int = 1
        self.current_agent_index: int = 0
        self.round_order: list[AgentBase] = []
        self.state: EnvState = EnvState.WAITING_LLM

        for i, agent in enumerate(agents):
            agent.agent_id = i + 1
        self._agent_map: dict[int, AgentBase] = {a.agent_id: a for a in agents}

    # ------------------------------------------------------------------
    # 初始化
    # ------------------------------------------------------------------

    def init(self):
        """初始化讨论：为每个 Agent 设置 system prompt，准备第一轮发言顺序。"""
        for agent in self.agents:
            agent.system_prompt = build_system_prompt(
                total_agents=len(self.agents),
                topic=self.topic,
                role_background=agent.role_background,
            )
        self._prepare_round_order()
        self._update_state_for_current_agent()

    # ------------------------------------------------------------------
    # 状态机核心
    # ------------------------------------------------------------------

    def step(self) -> EnvState:
        """推进一步：让当前 Agent 发言。

        Returns:
            当前状态枚举。
        """
        if self.state == EnvState.FINISHED:
            return self.state

        agent = self._get_current_agent()
        messages = self.build_messages_for_agent(agent)

        if agent.is_human and not agent.has_pending_input():
            self.state = EnvState.WAITING_HUMAN
            return self.state

        response = agent.talk(messages)
        self._record_talk(agent, response)
        self._advance()
        return self.state

    # ------------------------------------------------------------------
    # messages 构建（管道：Filter → Timeline Group → Render → Assemble）
    # ------------------------------------------------------------------

    def build_messages_for_agent(self, agent: AgentBase) -> list[dict]:
        """为指定 Agent 构建完整的 OpenAI messages 列表。

        基于 agent 个人时间线（以自身发言为锚点）严格保证
        System → [User → Assistant]* → User 的因果交替结构。
        """
        messages: list[dict] = [{"role": "system", "content": agent.system_prompt}]

        groups = self._build_timeline_groups(agent.agent_id)

        if not groups:
            messages.append({
                "role": "user",
                "content": self.format_initial_prompt(agent.agent_id),
            })
            return messages

        for turn_num, group in enumerate(groups, 1):
            others = group["others"]
            mine = group["mine"]

            if others:
                messages.append({
                    "role": "user",
                    "content": self.format_round_prompt(turn_num, others, agent.agent_id),
                })
            else:
                messages.append({
                    "role": "user",
                    "content": self.format_initial_prompt(agent.agent_id),
                })

            if mine is not None:
                messages.append({"role": "assistant", "content": mine})

        return messages

    # ------------------------------------------------------------------
    # 时间线分组（子类可重写）
    # ------------------------------------------------------------------

    def _build_timeline_groups(self, agent_id: int) -> list[dict]:
        """基于 agent 个人时间线构建消息分组。

        沿 global_history 时间顺序扫描，以自身发言为锚点切分：
        锚点之间的所有可见他人发言聚合为一个 "others" 块。

        返回 [{"others": [entry...], "mine": str|None}, ...] 有序列表。
        子类可重写（如 BrainWrite 需要逐轮可见性重建）。
        """
        visible = self.get_visible_messages(agent_id)
        visible_keys = {
            (e["agent_id"], e["round"])
            for e in visible
            if e["agent_id"] != agent_id
        }

        groups: list[dict] = []
        current_others: list[dict] = []

        for entry in self.global_history:
            if entry["agent_id"] == agent_id:
                groups.append({"others": current_others, "mine": entry["content"]})
                current_others = []
            elif (entry["agent_id"], entry["round"]) in visible_keys:
                current_others.append(entry)

        if current_others:
            groups.append({"others": current_others, "mine": None})
        elif not groups:
            groups.append({"others": [], "mine": None})
        elif self.current_round <= self.max_rounds:
            groups.append({"others": [], "mine": None})

        return groups

    # ------------------------------------------------------------------
    # 渲染 Hook（子类重写以定制 Prompt 语义）
    # ------------------------------------------------------------------

    def format_round_prompt(self, turn_num: int, others_entries: list[dict], agent_id: int) -> str:
        """将一组他人发言渲染为一条 User Prompt。

        turn_num 为该 agent 的第几次发言轮（1-based），子类可据此定制措辞。
        """
        lines = []
        for entry in others_entries:
            speaker = self.get_agent_display_name(entry["agent_id"])
            lines.append(IP.SPEAKER_LINE.format(speaker=speaker, content=entry["content"]))
        body = "\n".join(lines)
        if turn_num == 1:
            return IP.ROUND_FIRST.format(body=body)
        return IP.ROUND_FOLLOW.format(body=body)

    def format_initial_prompt(self, agent_id: int) -> str:
        """当无他人可见发言时的默认 User 提示。子类可重写。"""
        return IP.INITIAL_PROMPT

    # ------------------------------------------------------------------
    # 可见性（子类重写）
    # ------------------------------------------------------------------

    def get_visible_messages(self, agent_id: int) -> list[dict]:
        """返回指定 Agent 可见的历史发言列表。子类重写此方法实现不同规则。"""
        return list(self.global_history)

    # ------------------------------------------------------------------
    # 发言顺序（子类重写）
    # ------------------------------------------------------------------

    def _prepare_round_order(self):
        """准备当前轮的发言顺序。子类可重写。"""
        self.round_order = list(self.agents)

    # ------------------------------------------------------------------
    # 内部辅助
    # ------------------------------------------------------------------

    def _get_current_agent(self) -> AgentBase:
        return self.round_order[self.current_agent_index]

    def _record_talk(self, agent: AgentBase, content: str):
        self.global_history.append({
            "round": self.current_round,
            "agent_id": agent.agent_id,
            "agent_name": agent.display_name,
            "config_key": agent.config_key,
            "content": content,
        })

    def _advance(self):
        """推进到下一个 Agent 或下一轮。"""
        self.current_agent_index += 1
        if self.current_agent_index >= len(self.round_order):
            self.current_round += 1
            self.current_agent_index = 0
            if self.current_round > self.max_rounds:
                self.state = EnvState.FINISHED
            else:
                self._prepare_round_order()
                self._update_state_for_current_agent()
        else:
            self._update_state_for_current_agent()

    def _update_state_for_current_agent(self):
        if not self.round_order:
            self.state = EnvState.FINISHED
            return
        agent = self._get_current_agent()
        self.state = EnvState.WAITING_HUMAN if agent.is_human else EnvState.WAITING_LLM

    def get_agent(self, agent_id: int) -> AgentBase:
        return self._agent_map[agent_id]

    def get_agent_name(self, agent_id: int) -> str:
        return self._agent_map[agent_id].display_name

    def get_agent_display_name(self, agent_id: int) -> str:
        return self._agent_map[agent_id].display_name

    # ------------------------------------------------------------------
    # 日志
    # ------------------------------------------------------------------

    def save_log(self, log_dir: str | None = None):
        """将讨论结果保存为 JSON 日志。"""
        target_dir = log_dir or self.log_dir
        if target_dir is None:
            raise ValueError("未指定日志目录")
        os.makedirs(target_dir, exist_ok=True)

        human_count = sum(1 for a in self.agents if a.is_human)
        ts = datetime.now().strftime("%Y%m%d%H%M")
        filename = f"{self.mode}_{len(self.agents)}_{human_count}_{ts}.json"
        path = os.path.join(target_dir, filename)

        final_messages = {}
        for agent in self.agents:
            msgs = self.build_messages_for_agent(agent)
            final_messages[str(agent.agent_id)] = msgs

        position_map = [
            {
                "position": a.agent_id,
                "config_key": a.config_key,
                "type": "human" if a.is_human else "llm",
                "model": getattr(a, "inference_config", {}).get("model", "human"),
            }
            for a in self.agents
        ]

        log_data = {
            "metadata": {
                "mode": self.mode,
                "topic": self.topic,
                "max_rounds": self.max_rounds,
                "total_agents": len(self.agents),
                "human_count": human_count,
                "timestamp": datetime.now().isoformat(),
                "agents": [a.get_agent_info() for a in self.agents],
                "position_map": position_map,
            },
            "global_history": self.global_history,
            "final_messages": final_messages,
        }

        if hasattr(self, "final_rankings") and self.final_rankings:
            log_data["final_rankings"] = self.final_rankings

        with open(path, "w", encoding="utf-8") as f:
            json.dump(log_data, f, ensure_ascii=False, indent=2)
        return path
