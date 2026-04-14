from __future__ import annotations

import json
import os
from copy import deepcopy
from datetime import datetime

from agents.agent_base import AgentBase, EnvState
from prompts.system_prompts import build_system_prompt


class EnvBase:
    """讨论环境基类，采用步进式 (step-by-step) 状态机驱动。"""

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

        self._agent_map: dict[int, AgentBase] = {a.agent_id: a for a in agents}

    # ------------------------------------------------------------------
    # 初始化
    # ------------------------------------------------------------------

    def init(self):
        """初始化讨论：为每个 Agent 设置 system prompt，准备第一轮发言顺序。"""
        for agent in self.agents:
            agent.system_prompt = build_system_prompt(
                mode=self.mode,
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
    # messages 构建（四阶段管道：Filter → Group → Render → Assemble）
    # ------------------------------------------------------------------

    def build_messages_for_agent(self, agent: AgentBase) -> list[dict]:
        """为指定 Agent 构建完整的 OpenAI messages 列表。

        严格保证 System → [User → Assistant]* → User 的交替结构。
        """
        messages: list[dict] = [{"role": "system", "content": agent.system_prompt}]

        grouped = self._group_history_by_round(agent.agent_id)

        if not grouped:
            messages.append({
                "role": "user",
                "content": self.format_initial_prompt(agent.agent_id),
            })
            return messages

        for round_num in sorted(grouped.keys()):
            others = grouped[round_num]["others"]
            mine = grouped[round_num]["mine"]

            if others:
                messages.append({
                    "role": "user",
                    "content": self.format_round_prompt(round_num, others, agent.agent_id),
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
    # 分组（子类可重写）
    # ------------------------------------------------------------------

    def _group_history_by_round(self, agent_id: int) -> dict[int, dict]:
        """将可见历史按轮次分组为 {round: {"others": [...], "mine": str|None}}。

        "others" 来自 get_visible_messages 中非自身条目；
        "mine" 直接从 global_history 中提取自身发言。
        子类可重写以实现不同的分组策略（如 BrainWrite 逐轮可见性重建）。
        """
        visible = self.get_visible_messages(agent_id)

        others_by_round: dict[int, list[dict]] = {}
        for entry in visible:
            if entry["agent_id"] != agent_id:
                others_by_round.setdefault(entry["round"], []).append(entry)

        mine_by_round: dict[int, str] = {}
        for entry in self.global_history:
            if entry["agent_id"] == agent_id:
                mine_by_round[entry["round"]] = entry["content"]

        all_rounds = sorted(set(list(others_by_round.keys()) + list(mine_by_round.keys())))
        result: dict[int, dict] = {}
        for r in all_rounds:
            result[r] = {
                "others": others_by_round.get(r, []),
                "mine": mine_by_round.get(r, None),
            }

        if self.current_round <= self.max_rounds and self.current_round not in result:
            result[self.current_round] = {
                "others": others_by_round.get(self.current_round, []),
                "mine": mine_by_round.get(self.current_round),
            }

        return result

    # ------------------------------------------------------------------
    # 渲染 Hook（子类重写以定制 Prompt 语义）
    # ------------------------------------------------------------------

    def format_round_prompt(self, round_num: int, others_entries: list[dict], agent_id: int) -> str:
        """将同轮次的他人发言渲染为一条 User Prompt。子类重写此方法注入环境语义。"""
        lines = []
        for entry in others_entries:
            speaker = self.get_agent_name(entry["agent_id"])
            if self.get_agent(entry["agent_id"]).is_human:
                speaker = f"人类专家 {speaker}"
            lines.append(f"- {speaker} 说：{entry['content']}")
        return f"在第 {round_num} 轮讨论中，你听到了以下发言：\n" + "\n".join(lines)

    def format_initial_prompt(self, agent_id: int) -> str:
        """当某轮次无他人可见发言时的默认 User 提示。子类可重写。"""
        return "现在请你率先发言，针对讨论主题分享你的观点和思考。"

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
            "agent_name": agent.name,
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
        return self._agent_map[agent_id].name

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

        log_data = {
            "metadata": {
                "mode": self.mode,
                "topic": self.topic,
                "max_rounds": self.max_rounds,
                "total_agents": len(self.agents),
                "human_count": human_count,
                "timestamp": datetime.now().isoformat(),
                "agents": [a.get_agent_info() for a in self.agents],
            },
            "global_history": self.global_history,
            "final_messages": final_messages,
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(log_data, f, ensure_ascii=False, indent=2)
        return path
