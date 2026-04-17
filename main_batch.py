#!/usr/bin/env python3
"""纯 LLM 批量测试入口。

新用法（推荐）：
    python main_batch.py \
        --config config/llm_config.json \
        --models "qwen3_8b_local,qwen3_8b_local,qwen3_8b_local,qwen3_8b_local" \
        --mode brainwrite \
        --rounds 4 \
        --topic "人工智能技术能怎样帮助解决三体问题？"

旧用法（向后兼容）：
    python main_batch.py \
        --mode brainwrite \
        --rounds 3 \
        --topic "..." \
        --agents '[{"role":"...","api_key":"sk-xxx","base_url":"https://...","model":"gpt-4","temperature":0.7}]'
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
from tools.config_loader import load_llm_config, build_agent_from_config

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

    parser.add_argument("--config", type=str, default=None,
                        help="llm_config.json 路径")
    parser.add_argument("--models", type=str, default=None,
                        help="逗号分隔的 model key 列表，顺序即 position")

    parser.add_argument("--agents", type=str, default=None,
                        help="（旧）Agent 配置 JSON 字符串（向后兼容）")
    parser.add_argument("--leader_ids", type=str, default="[]",
                        help="Leader ID 列表 JSON（仅 leader_worker 模式）")
    parser.add_argument("--log_dir", type=str, default=None,
                        help="日志目录（默认: log/）")
    return parser.parse_args()


def build_agents_from_models(model_keys: list[str], pool: dict) -> list[AgentLLM]:
    """根据 model key 列表构建 Agent，顺序即 position。"""
    agents = []
    for i, key in enumerate(model_keys):
        agent = build_agent_from_config(agent_id=i + 1, model_key=key, pool=pool)
        agents.append(agent)
    return agents


def build_agents_legacy(agents_json: list[dict]) -> list[AgentLLM]:
    """旧版 --agents JSON 构建方式（向后兼容）。"""
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
            role_background=cfg.get("role", ""),
            api_config=api_config,
            inference_config=inference_config,
        ))
    return agents


def main():
    args = parse_args()
    leader_ids = json.loads(args.leader_ids)

    if args.config and args.models:
        pool = load_llm_config(args.config)
        model_keys = [k.strip() for k in args.models.split(",")]
        agents = build_agents_from_models(model_keys, pool)
    elif args.agents:
        agents_cfg = json.loads(args.agents)
        agents = build_agents_legacy(agents_cfg)
    else:
        print("错误：必须指定 --config + --models 或 --agents")
        sys.exit(1)

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
    for a in agents:
        model = getattr(a, "inference_config", {}).get("model", "unknown")
        print(f"  {a.display_name} -> {model}")
    print("-" * 60)

    step_count = 0
    while env.state != EnvState.FINISHED:
        env.step()
        step_count += 1
        last = env.global_history[-1] if env.global_history else None
        if last:
            print(f"[第{last['round']}轮] {last['agent_name']}：")
            content_preview = last["content"][:200] if last["content"] else ""
            print(f"  {content_preview}{'...' if len(last.get('content', '')) > 200 else ''}")
            print()

    print("-" * 60)
    log_path = env.save_log()
    print(f"[完成] 共 {step_count} 步, 日志已保存至: {log_path}")


if __name__ == "__main__":
    main()
