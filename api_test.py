#!/usr/bin/env python3
"""使用项目内 `tools.call_openai.call_openai` 对 llm_config 中各端点做连通性探测。

构建 `api_config` / `inference_config` 的方式与 `tools.config_loader.build_agent_from_config`
一致，但**不使用** `enable_identity`、`identity_prompt`（与配置中这两项取值无关）。

用法：
    python api_test.py
    python api_test.py --config /path/to/llm_config.json
    python api_test.py --model qwen2.5_14B
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tools.call_openai import call_openai
from tools.config_loader import load_llm_config


def _build_api_and_inference(cfg: dict) -> tuple[dict, dict]:
    """与 build_agent_from_config 中 API 相关部分一致，忽略身份相关字段。"""
    api_config = {
        "api_key": cfg["api_key"],
        "base_url": cfg["api_url"],
    }
    inference_config: dict = {
        "model": cfg["model_name"],
        "temperature": cfg["temperature"],
        "is_reasoning": cfg.get("is_reasoning", False),
    }
    if cfg.get("top_p") is not None:
        inference_config["top_p"] = cfg["top_p"]
    if cfg.get("max_tokens") is not None:
        inference_config["max_tokens"] = cfg["max_tokens"]
    return api_config, inference_config


def _ping_messages() -> list[dict]:
    return [{"role": "user", "content": "只回复一个字：好"}]


def test_one(cfg: dict) -> tuple[bool, str]:
    api_config, inference_config = _build_api_and_inference(cfg)
    try:
        text = call_openai(_ping_messages(), api_config, inference_config)
    except Exception as exc:  # noqa: BLE001 — 连通性脚本需要捕获所有后端错误
        return False, f"{type(exc).__name__}: {exc}"

    if not (text and text.strip()):
        return True, "(HTTP 成功但返回正文为空，仍视为连通)"
    preview = text.strip().replace("\n", " ")[:80]
    return True, preview


def main() -> int:
    parser = argparse.ArgumentParser(description="LLM 池 API 连通性检测")
    parser.add_argument(
        "--config",
        default=None,
        help="llm_config.json 路径，默认使用 tools.config_loader 的默认路径",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="仅测试指定 config_key；默认测试池中全部模型",
    )
    args = parser.parse_args()

    pool = load_llm_config(args.config)
    keys = [args.model] if args.model else list(pool.keys())

    if args.model and args.model not in pool:
        print(f"错误：池中不存在模型键 {args.model!r}", file=sys.stderr)
        return 2

    failed = 0
    for key in keys:
        ok, info = test_one(pool[key])
        status = "OK " if ok else "FAIL"
        print(f"[{status}] {key}: {info}")
        if not ok:
            failed += 1

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
