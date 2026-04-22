#!/usr/bin/env python3
"""LLM 池 API 连通性检测（增强版）。

相比 `api_test.py`：
- 不仅输出测试结果摘要，还输出**模型原始输出**（包含 `<think>...</think>` 等 CoT/思考片段，如果服务端返回）。
- 可选输出完整的原始响应 JSON，用于判断 `is_reasoning` / `extra_body` 等设置是否生效。

用法：
    python api_test_2.py
    python api_test_2.py --config /path/to/llm_config.json
    python api_test_2.py --model qwen2.5_14B
    python api_test_2.py --toggle-reasoning
    python api_test_2.py --no-dump-json
    python api_test_2.py --max-chars 20000
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

from openai import OpenAI

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tools.config_loader import load_llm_config


def _build_api_config(cfg: dict) -> dict:
    return {
        "api_key": cfg["api_key"],
        "base_url": cfg["api_url"],
    }


def _build_call_kwargs(cfg: dict, *, is_reasoning: bool) -> dict:
    call_kwargs: dict[str, Any] = {
        "model": cfg["model_name"],
        "temperature": cfg["temperature"],
    }
    if cfg.get("top_p") is not None:
        call_kwargs["top_p"] = cfg["top_p"]
    if cfg.get("max_tokens") is not None:
        call_kwargs["max_tokens"] = cfg["max_tokens"]

    if not is_reasoning:
        model_name = str(call_kwargs.get("model", ""))
        if "glm" in model_name.lower():
            call_kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
        else:
            call_kwargs["extra_body"] = {
                "chat_template_kwargs": {"enable_thinking": False}
            }

    return call_kwargs


def _ping_messages() -> list[dict[str, str]]:
    return [{"role": "user", "content": "只回复一个字：好"}]


def _safe_str(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    return str(x)


def _extract_message_fields(completion: Any) -> tuple[str, str, dict]:
    """
    返回：
    - raw_content: message.content 原样（可能含 <think>）
    - maybe_reasoning: 一些兼容字段（如果服务端/SDK提供）
    - message_dict: message 的 dict 形式（尽可能完整）
    """
    choice0 = completion.choices[0]
    msg = choice0.message

    raw_content = _safe_str(getattr(msg, "content", None))

    maybe_reasoning = ""
    for key in ("reasoning", "thinking", "thought", "cot"):
        if hasattr(msg, key):
            maybe_reasoning = _safe_str(getattr(msg, key))
            if maybe_reasoning:
                break

    if hasattr(msg, "model_dump"):
        message_dict = msg.model_dump()
    elif hasattr(msg, "to_dict"):
        message_dict = msg.to_dict()
    else:
        message_dict = {
            k: getattr(msg, k)
            for k in dir(msg)
            if not k.startswith("_") and k in {"role", "content", "reasoning"}
        }

    return raw_content, maybe_reasoning, message_dict


def _dump_completion(completion: Any) -> dict:
    if hasattr(completion, "model_dump"):
        return completion.model_dump()
    if hasattr(completion, "to_dict"):
        return completion.to_dict()
    return {"repr": repr(completion)}


def _truncate(text: str, max_chars: int | None) -> str:
    if not max_chars or max_chars <= 0:
        return text
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n...[truncated, total_chars={len(text)}]"


def _run_once(cfg: dict, *, is_reasoning: bool, dump_json: bool, max_chars: int | None) -> tuple[bool, str]:
    api_config = _build_api_config(cfg)
    call_kwargs = _build_call_kwargs(cfg, is_reasoning=is_reasoning)

    client = OpenAI(**api_config)
    try:
        completion = client.chat.completions.create(messages=_ping_messages(), **call_kwargs)
    except Exception as exc:  # noqa: BLE001 — 测试脚本需要捕获所有后端错误
        return False, f"{type(exc).__name__}: {exc}"

    raw_content, maybe_reasoning, message_dict = _extract_message_fields(completion)

    has_think_tag = ("<think>" in raw_content) or ("</think>" in raw_content)
    cleaned_content = raw_content
    if has_think_tag:
        # 仅用于对比展示，不用于业务逻辑判断
        cleaned_content = cleaned_content.replace("<think>", "").replace("</think>", "")

    summary = cleaned_content.strip().replace("\n", " ")[:80] if cleaned_content.strip() else "(正文为空)"

    print("")
    print("=" * 80)
    print(f"model={cfg.get('model_name')!r}  is_reasoning={is_reasoning}  has_<think>={has_think_tag}")
    print(f"call_kwargs(extra_body?)={json.dumps({k: call_kwargs.get(k) for k in call_kwargs if k=='extra_body'}, ensure_ascii=False)}")
    print("-" * 80)
    print("[preview(cleaned_content)]")
    print(summary)
    print("-" * 80)
    print("[raw message.content]")
    print(_truncate(raw_content, max_chars))
    if maybe_reasoning:
        print("-" * 80)
        print("[message.reasoning/thinking field]")
        print(_truncate(maybe_reasoning, max_chars))
    print("-" * 80)
    print("[message dict]")
    print(_truncate(json.dumps(message_dict, ensure_ascii=False, indent=2), max_chars))

    if dump_json:
        print("-" * 80)
        print("[full completion json]")
        print(_truncate(json.dumps(_dump_completion(completion), ensure_ascii=False, indent=2), max_chars))

    print("=" * 80)
    return True, summary


def main() -> int:
    parser = argparse.ArgumentParser(description="LLM 池 API 连通性检测（增强输出版）")
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
    parser.add_argument(
        "--toggle-reasoning",
        action="store_true",
        help="对每个模型额外再跑一次 is_reasoning 取反，用于对比开关是否生效",
    )
    parser.add_argument(
        "--no-dump-json",
        action="store_true",
        help="不输出 full completion json（输出会更短）",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=0,
        help="单段输出最多字符数；0 表示不截断（可能很长）",
    )
    args = parser.parse_args()

    pool = load_llm_config(args.config)
    keys = [args.model] if args.model else list(pool.keys())

    if args.model and args.model not in pool:
        print(f"错误：池中不存在模型键 {args.model!r}", file=sys.stderr)
        return 2

    failed = 0
    dump_json = not args.no_dump_json
    max_chars: int | None = args.max_chars if args.max_chars and args.max_chars > 0 else None

    for key in keys:
        cfg = pool[key]
        base_is_reasoning = bool(cfg.get("is_reasoning", False))

        ok, _ = _run_once(
            cfg,
            is_reasoning=base_is_reasoning,
            dump_json=dump_json,
            max_chars=max_chars,
        )
        if not ok:
            print(f"[FAIL] {key}")
            failed += 1
            continue
        print(f"[OK]   {key} (primary run)")

        if args.toggle_reasoning:
            ok2, _ = _run_once(
                cfg,
                is_reasoning=not base_is_reasoning,
                dump_json=dump_json,
                max_chars=max_chars,
            )
            if not ok2:
                print(f"[FAIL] {key} (toggle run)")
                failed += 1
            else:
                print(f"[OK]   {key} (toggle run)")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

