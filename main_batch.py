#!/usr/bin/env python3
"""纯 LLM 批量测试入口。

示例：
    python main_batch.py \
        --mode brainwrite \
        --rounds 3 \
        --topic "人工智能技术能怎样帮助解决三体问题？" \
        --agents '[
            {"name":"AI专家","role":"请扮演一位AI研究员...","api_key":"sk-xxx","base_url":"https://...","model":"gpt-4","temperature":0.7},
            {"name":"数学家","role":"请扮演一位数学教授...","api_key":"sk-yyy","base_url":"https://...","model":"glm-4","temperature":0.9}
        ]'
"""

import argparse
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents.agent_base import EnvState
from agents.agent_llm import AgentLLM
from envs.brainwrite import BrainWrite
from envs.round_robin import RoundRobin
from envs.random_env import RandomEnv
from envs.leader_worker import LeaderWorker

ENV_MAP = {
    "brainwrite": BrainWrite,
    "round_robin": RoundRobin,
    "random": RandomEnv,
    "leader_worker": LeaderWorker,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Brainstorm 批量测试")
    parser.add_argument("--mode", type=str, required=True,
                        choices=list(ENV_MAP.keys()),
                        help="讨论形式")
    parser.add_argument("--rounds", type=int, required=True,
                        help="最大轮数")
    parser.add_argument("--topic", type=str, required=True,
                        help="讨论主题")
    parser.add_argument("--agents", type=str, required=True,
                        help="Agent 配置 JSON 字符串")
    parser.add_argument("--leader_ids", type=str, default="[]",
                        help="Leader ID 列表 JSON（仅 leader_worker 模式）")
    parser.add_argument("--log_dir", type=str, default=None,
                        help="日志目录（默认: log/）")
    return parser.parse_args()


def build_agents(agents_json: list[dict]) -> list[AgentLLM]:
    agents = []
    for i, cfg in enumerate(agents_json):
        agent_id = i + 1
        api_config = {
            "api_key": cfg["api_key"],
            "base_url": cfg["base_url"],
        }
        inference_config = {"model": cfg["model"]}
        if "temperature" in cfg:
            inference_config["temperature"] = cfg["temperature"]
        if "top_p" in cfg:
            inference_config["top_p"] = cfg["top_p"]
        if "max_tokens" in cfg:
            inference_config["max_tokens"] = cfg["max_tokens"]

        agents.append(AgentLLM(
            agent_id=agent_id,
            name=cfg["name"],
            role_background=cfg["role"],
            api_config=api_config,
            inference_config=inference_config,
        ))
    return agents


def main():
    args = parse_args()

    agents_cfg = json.loads(args.agents)
    leader_ids = json.loads(args.leader_ids)
    agents = build_agents(agents_cfg)

    log_dir = args.log_dir or os.path.join(os.path.dirname(__file__), "log")

    env_cls = ENV_MAP[args.mode]
    if args.mode == "leader_worker":
        env = env_cls(
            agents=agents,
            topic=args.topic,
            max_rounds=args.rounds,
            leader_ids=leader_ids,
            log_dir=log_dir,
        )
    else:
        env = env_cls(
            agents=agents,
            topic=args.topic,
            max_rounds=args.rounds,
            log_dir=log_dir,
        )

    env.init()
    print(f"[开始讨论] 模式={args.mode}, 轮数={args.rounds}, Agent数={len(agents)}")
    print(f"[主题] {args.topic}")
    print("-" * 60)

    step_count = 0
    while env.state != EnvState.FINISHED:
        state = env.step()
        step_count += 1
        last = env.global_history[-1] if env.global_history else None
        if last and last == env.global_history[-1]:
            print(f"[第{last['round']}轮] {last['agent_name']}：")
            content_preview = last["content"][:200] if last["content"] else ""
            print(f"  {content_preview}{'...' if len(last.get('content', '')) > 200 else ''}")
            print()

    print("-" * 60)
    log_path = env.save_log()
    print(f"[完成] 共 {step_count} 步, 日志已保存至: {log_path}")


if __name__ == "__main__":
    main()
