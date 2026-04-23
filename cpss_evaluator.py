"""
CPSS (Creative Product Semantic Scale) 自动化评估脚本

利用 LLM 对头脑风暴日志中的创意产品进行 55 维语义量表打分。
每个维度独立发起一次 LLM 请求，通过异步并发 + Semaphore 限流完成评估，
结果以 "cpss_evaluation" 键回写至原 JSON 文件。
"""

from __future__ import annotations

import asyncio
import glob
import json
import os
import re
from typing import Any

from openai import AsyncOpenAI

# ═══════════════════════════════════════════════════════════════════
# CPSS 55-item 词库（已按标准量表修正第 22/37/45 题）
# 每项: (题号, 左侧词=1分端, 右侧词=7分端, 回写key)
# ═══════════════════════════════════════════════════════════════════

CPSS_ITEMS: list[dict[str, Any]] = [
    {"id": 1,  "left": "Over Used",      "right": "Fresh",            "key": "Q01_OverUsed_Fresh"},
    {"id": 2,  "left": "Stale",          "right": "Startling",        "key": "Q02_Stale_Startling"},
    {"id": 3,  "left": "Illogical",      "right": "Logical",          "key": "Q03_Illogical_Logical"},
    {"id": 4,  "left": "Usual",          "right": "Unusual",          "key": "Q04_Usual_Unusual"},
    {"id": 5,  "left": "Inadequate",     "right": "Adequate",         "key": "Q05_Inadequate_Adequate"},
    {"id": 6,  "left": "Original",       "right": "Conventional",     "key": "Q06_Original_Conventional"},
    {"id": 7,  "left": "Trendy",         "right": "Outdated",         "key": "Q07_Trendy_Outdated"},
    {"id": 8,  "left": "Unique",         "right": "Ordinary",         "key": "Q08_Unique_Ordinary"},
    {"id": 9,  "left": "Functional",     "right": "Nonfunctional",    "key": "Q09_Functional_Nonfunctional"},
    {"id": 10, "left": "Useful",         "right": "Useless",          "key": "Q10_Useful_Useless"},
    {"id": 11, "left": "Irrelevant",     "right": "Relevant",         "key": "Q11_Irrelevant_Relevant"},
    {"id": 12, "left": "Trivial",        "right": "Important",        "key": "Q12_Trivial_Important"},
    {"id": 13, "left": "Novel",          "right": "Predictable",      "key": "Q13_Novel_Predictable"},
    {"id": 14, "left": "Surprising",     "right": "Commonplace",      "key": "Q14_Surprising_Commonplace"},
    {"id": 15, "left": "Germane",        "right": "Inappropriate",    "key": "Q15_Germane_Inappropriate"},
    {"id": 16, "left": "Resourceful",    "right": "Unresourceful",    "key": "Q16_Resourceful_Unresourceful"},
    {"id": 17, "left": "Inoperable",     "right": "Workable",         "key": "Q17_Inoperable_Workable"},
    {"id": 18, "left": "Tasteful",       "right": "Tasteless",        "key": "Q18_Tasteful_Tasteless"},
    {"id": 19, "left": "Organic",        "right": "Contrived",        "key": "Q19_Organic_Contrived"},
    {"id": 20, "left": "Well Made",      "right": "Poorly Made",      "key": "Q20_WellMade_PoorlyMade"},
    {"id": 21, "left": "Valuable",       "right": "Worthless",        "key": "Q21_Valuable_Worthless"},
    {"id": 22, "left": "Shocking",       "right": "Old-Fashioned",    "key": "Q22_Shocking_OldFashioned"},
    {"id": 23, "left": "Elaborate",      "right": "Simple",           "key": "Q23_Elaborate_Simple"},
    {"id": 24, "left": "Misunderstood",  "right": "Understood",       "key": "Q24_Misunderstood_Understood"},
    {"id": 25, "left": "Exciting",       "right": "Dull",             "key": "Q25_Exciting_Dull"},
    {"id": 26, "left": "Inspired",       "right": "Uninspired",       "key": "Q26_Inspired_Uninspired"},
    {"id": 27, "left": "Hostile",        "right": "Inviting",         "key": "Q27_Hostile_Inviting"},
    {"id": 28, "left": "Elegant",        "right": "Inelegant",        "key": "Q28_Elegant_Inelegant"},
    {"id": 29, "left": "Valid",          "right": "Invalid",          "key": "Q29_Valid_Invalid"},
    {"id": 30, "left": "Expressive",     "right": "Unexpressive",     "key": "Q30_Expressive_Unexpressive"},
    {"id": 31, "left": "Ambitious",      "right": "Unambitious",      "key": "Q31_Ambitious_Unambitious"},
    {"id": 32, "left": "Vital",          "right": "Unimportant",      "key": "Q32_Vital_Unimportant"},
    {"id": 33, "left": "Effective",      "right": "Ineffective",      "key": "Q33_Effective_Ineffective"},
    {"id": 34, "left": "Progressive",    "right": "Regressive",       "key": "Q34_Progressive_Regressive"},
    {"id": 35, "left": "Imaginative",    "right": "Unimaginative",    "key": "Q35_Imaginative_Unimaginative"},
    {"id": 36, "left": "Avant-Garde",    "right": "Old-Guard",        "key": "Q36_AvantGarde_OldGuard"},
    {"id": 37, "left": "Radical",        "right": "Old Hat",          "key": "Q37_Radical_OldHat"},
    {"id": 38, "left": "Unpolished",     "right": "Polished",         "key": "Q38_Unpolished_Polished"},
    {"id": 39, "left": "Complete",       "right": "Incomplete",       "key": "Q39_Complete_Incomplete"},
    {"id": 40, "left": "Cohesive",       "right": "Disjointed",       "key": "Q40_Cohesive_Disjointed"},
    {"id": 41, "left": "Needed",         "right": "Unneeded",         "key": "Q41_Needed_Unneeded"},
    {"id": 42, "left": "Meticulous",     "right": "Careless",         "key": "Q42_Meticulous_Careless"},
    {"id": 43, "left": "Revolutionary",  "right": "Pedestrian",       "key": "Q43_Revolutionary_Pedestrian"},
    {"id": 44, "left": "Pleasurable",    "right": "Unpleasant",       "key": "Q44_Pleasurable_Unpleasant"},
    {"id": 45, "left": "Crude",          "right": "Well-Crafted",     "key": "Q45_Crude_WellCrafted"},
    {"id": 46, "left": "Visionary",      "right": "Mundane",          "key": "Q46_Visionary_Mundane"},
    {"id": 47, "left": "Insightful",     "right": "Trite",            "key": "Q47_Insightful_Trite"},
    {"id": 48, "left": "Desire",         "right": "Undesirable",      "key": "Q48_Desire_Undesirable"},
    {"id": 49, "left": "Deliberate",     "right": "Random",           "key": "Q49_Deliberate_Random"},
    {"id": 50, "left": "Appealing",      "right": "Unappealing",      "key": "Q50_Appealing_Unappealing"},
    {"id": 51, "left": "Detailed",       "right": "Sketchy",          "key": "Q51_Detailed_Sketchy"},
    {"id": 52, "left": "Feasible",       "right": "Unfeasible",       "key": "Q52_Feasible_Unfeasible"},
    {"id": 53, "left": "Meaningful",     "right": "Meaningless",      "key": "Q53_Meaningful_Meaningless"},
    {"id": 54, "left": "Flexible",       "right": "Inflexible",       "key": "Q54_Flexible_Inflexible"},
    {"id": 55, "left": "Overused",       "right": "New",              "key": "Q55_Overused_New"},
]

# ═══════════════════════════════════════════════════════════════════
# Prompt 模板
# ═══════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = (
    "You are an expert evaluator assessing creative products and ideas "
    "using the Creative Product Semantic Scale (CPSS).\n"
    "Your task is to evaluate a specific idea based on a single 7-point "
    "bipolar semantic scale.\n\n"
    "RULES:\n"
    "1. You will be provided with a \"Topic\", the \"Proposed Idea\", "
    "and a \"Semantic Scale\" ranging from 1 to 7 with contrasting "
    "adjectives at each end.\n"
    "2. 4 represents a neutral midpoint.\n"
    "3. You must choose exactly one integer between 1 and 7 that best "
    "represents your assessment of the idea on this specific scale.\n"
    "4. CRITICAL INSTRUCTION: You MUST output ONLY a single integer "
    "(e.g., 1, 4, 7). Do NOT provide any reasoning, explanation, "
    "punctuation, or additional text. Your entire response must be "
    "just the number."
)

USER_PROMPT_TEMPLATE = (
    "Topic: {topic}\n\n"
    "Proposed Idea:\n{idea_content}\n\n"
    "Semantic Scale to Evaluate:\n"
    "1 - {left_word}\n"
    "2 - Leaning towards {left_word}\n"
    "3 - Slightly {left_word}\n"
    "4 - Neutral / Neither\n"
    "5 - Slightly {right_word}\n"
    "6 - Leaning towards {right_word}\n"
    "7 - {right_word}\n\n"
    "Based on the Proposed Idea, evaluate it on the scale from 1 to 7 above.\n"
    "Output strictly a single integer:"
)

# ═══════════════════════════════════════════════════════════════════
# 配置文件路径与常量
# ═══════════════════════════════════════════════════════════════════

_CPSS_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "config",
    "cpss_eval_config.json",
)

_DIGIT_RE = re.compile(r"[1-7]")


def load_cpss_config(config_path: str | None = None) -> dict:
    """加载 CPSS 评估专用配置文件，返回完整配置字典。"""
    path = config_path or _CPSS_CONFIG_PATH
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_api_and_inference_config(cfg: dict) -> tuple[dict, dict]:
    """从 cpss_eval_config 构建 AsyncOpenAI 客户端参数和推理参数。"""
    api_config = {
        "api_key": cfg["api_key"],
        "base_url": cfg["api_url"],
    }
    inference_config: dict[str, Any] = {
        "model": cfg["model_name"],
        "temperature": cfg.get("temperature", 0.1),
    }
    if cfg.get("is_reasoning"):
        inference_config["is_reasoning"] = True
    if cfg.get("top_p") is not None:
        inference_config["top_p"] = cfg["top_p"]
    if cfg.get("max_tokens") is not None:
        inference_config["max_tokens"] = cfg["max_tokens"]

    return api_config, inference_config


# ═══════════════════════════════════════════════════════════════════
# 核心：异步单题评估
# ═══════════════════════════════════════════════════════════════════

async def evaluate_single_question(
    client: AsyncOpenAI,
    inference_config: dict,
    item: dict,
    topic: str,
    idea_content: str,
    semaphore: asyncio.Semaphore,
    max_retries: int = 3,
) -> tuple[str, int | None]:
    """对单个 CPSS 维度发起 LLM 评估请求。

    Returns:
        (item_key, score)  score 为 1-7 的整数，失败则为 None。
    """
    user_prompt = USER_PROMPT_TEMPLATE.format(
        topic=topic,
        idea_content=idea_content,
        left_word=item["left"],
        right_word=item["right"],
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    call_kwargs = dict(inference_config)
    is_reasoning = call_kwargs.pop("is_reasoning", False)
    if not is_reasoning:
        model_name = call_kwargs.get("model", "")
        if "glm" in model_name.lower():
            call_kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
        else:
            call_kwargs["extra_body"] = {
                "chat_template_kwargs": {"enable_thinking": False}
            }

    for attempt in range(1, max_retries + 1):
        try:
            async with semaphore:
                completion = await client.chat.completions.create(
                    messages=messages,
                    **call_kwargs,
                )
            raw = completion.choices[0].message.content.strip()
            match = _DIGIT_RE.search(raw)
            if match:
                return item["key"], int(match.group())
            print(
                f"    [WARN] Q{item['id']} attempt {attempt}: "
                f"unexpected response '{raw}'"
            )
        except Exception as e:
            print(
                f"    [ERR]  Q{item['id']} attempt {attempt}: {e}"
            )

    print(f"    [FAIL] Q{item['id']} exhausted {max_retries} retries")
    return item["key"], None


# ═══════════════════════════════════════════════════════════════════
# 组合全部 Agent 发言为 idea_content
# ═══════════════════════════════════════════════════════════════════

def extract_idea_content(data: dict) -> str:
    """将 global_history 中所有发言拼接为完整的创意描述文本。"""
    parts: list[str] = []
    for turn in data.get("global_history", []):
        content = turn.get("content", "").strip()
        if content:
            agent_name = turn.get("agent_name", f"Agent {turn.get('agent_id', '?')}")
            parts.append(f"[{agent_name}]: {content}")
    return "\n\n".join(parts)


# ═══════════════════════════════════════════════════════════════════
# 单文件评估流程
# ═══════════════════════════════════════════════════════════════════

async def evaluate_file(
    filepath: str,
    client: AsyncOpenAI,
    inference_config: dict,
    semaphore: asyncio.Semaphore,
    max_retries: int = 3,
) -> bool:
    """对单个 JSON 日志文件执行 55 维 CPSS 评估并回写。

    Returns:
        True 表示成功，False 表示有题目评估失败。
    """
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    topic = data.get("metadata", {}).get("topic", "")
    if not topic:
        print(f"  [WARN] No topic in {filepath}, using empty string.")

    idea_content = extract_idea_content(data)
    if not idea_content:
        print(f"  [SKIP] No content in {filepath}")
        return False

    tasks = [
        evaluate_single_question(
            client=client,
            inference_config=inference_config,
            item=item,
            topic=topic,
            idea_content=idea_content,
            semaphore=semaphore,
            max_retries=max_retries,
        )
        for item in CPSS_ITEMS
    ]

    results = await asyncio.gather(*tasks)

    cpss_evaluation: dict[str, int | None] = {}
    all_ok = True
    for key, score in results:
        cpss_evaluation[key] = score
        if score is None:
            all_ok = False

    data["cpss_evaluation"] = cpss_evaluation

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return all_ok


# ═══════════════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════════════

async def async_main() -> None:
    cfg = load_cpss_config()

    api_config, inference_config = build_api_and_inference_config(cfg)
    concurrency = cfg.get("concurrency", 10)
    max_retries = cfg.get("max_retries", 3)
    target_dirs: list[str] = cfg.get("target_dirs", ["log"])

    client = AsyncOpenAI(**api_config)
    semaphore = asyncio.Semaphore(concurrency)

    all_files: list[str] = []
    for folder in target_dirs:
        if not os.path.isdir(folder):
            print(f"[WARN] Directory not found: {folder}, skipping.")
            continue
        files = sorted(glob.glob(os.path.join(folder, "*.json")))
        all_files.extend(files)

    if not all_files:
        print("[INFO] No JSON files found in specified directories.")
        return

    print(
        f"CPSS Evaluator — model={inference_config['model']}, "
        f"concurrency={concurrency}, "
        f"temperature={inference_config.get('temperature', 0.1)}, "
        f"retries={max_retries}"
    )
    print(f"Target dirs: {target_dirs}")
    print(f"Found {len(all_files)} JSON file(s) to evaluate.\n")

    success_count = 0
    fail_count = 0

    for i, filepath in enumerate(all_files, 1):
        filename = os.path.basename(filepath)
        print(f"[{i}/{len(all_files)}] {filename} ...", flush=True)
        try:
            ok = await evaluate_file(
                filepath=filepath,
                client=client,
                inference_config=inference_config,
                semaphore=semaphore,
                max_retries=max_retries,
            )
            if ok:
                print(f"  => OK (55/55 scores written)")
                success_count += 1
            else:
                print(f"  => PARTIAL (some scores are null)")
                fail_count += 1
        except Exception as e:
            print(f"  => FAILED ({e})")
            fail_count += 1

    print(f"\nDone. Success: {success_count}, Issues: {fail_count}")


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
