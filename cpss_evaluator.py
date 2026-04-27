"""
CPSS (Creative Product Semantic Scale) 自动化评估脚本

利用 LLM 对头脑风暴日志中的创意产品进行 55 维语义量表打分。
对每个 Agent 的创意内容分别独立打分，通过异步并发 + Semaphore 限流完成评估，
结果以 "cpss_evaluation_per_agent" 键回写至原 JSON 文件。
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
# Prompt 模板与正则
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

_THINK_PATTERN = re.compile(r"<think>.*?</think>", re.DOTALL)
_DIGIT_RE = re.compile(r"[1-7]")

def _clean_think_tags(text: str) -> str:
    """剔除 <think>...</think> 块，防止从思考过程中抓去到了错误的评分数字"""
    cleaned = _THINK_PATTERN.sub("", text)
    return cleaned.strip()

# ═══════════════════════════════════════════════════════════════════
# 配置文件路径与常量
# ═══════════════════════════════════════════════════════════════════

_CPSS_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "config",
    "cpss_eval_config.json",
)

def load_cpss_config(config_path: str | None = None) -> dict:
    path = config_path or _CPSS_CONFIG_PATH
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def build_api_and_inference_config(cfg: dict) -> tuple[dict, dict]:
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
            # 必须先清理 think 标签，再做正则提取，避免抓出推理过程中的中间数字
            cleaned = _clean_think_tags(raw)
            match = _DIGIT_RE.search(cleaned)
            
            if match:
                return item["key"], int(match.group())
            
            print(f"    [WARN] Q{item['id']} attempt {attempt}: unexpected response '{cleaned}' (raw: {raw})")
            
        except Exception as e:
            err_msg = str(e).lower()
            # 容错：有些 API 节点不支持 extra_body 会抛出 400 错误，捕获并下一次重试剥离它
            if "extra_body" in err_msg or "unrecognized" in err_msg or "400" in err_msg:
                if "extra_body" in call_kwargs:
                    call_kwargs.pop("extra_body")
                    print(f"    [WARN] API rejected extra_body, retrying without it...")
                    
            print(f"    [ERR]  Q{item['id']} attempt {attempt}: {e}")

    print(f"    [FAIL] Q{item['id']} exhausted {max_retries} retries")
    return item["key"], None

# ═══════════════════════════════════════════════════════════════════
# 分离提取每个 Agent 的发言
# ═══════════════════════════════════════════════════════════════════

def extract_agent_ideas(data: dict) -> dict[str, dict[str, Any]]:
    """
    将 global_history 中每个 Agent 的发言提取为独立文本，并携带必要的身份字段：
    agent_id / position / config_key。

    说明：
    - 日志里 `position_map` 可能存在多个相同 `config_key`（同模型多席位），因此 position
      不能用 config_key 反查；这里直接使用 `agent_id` 作为 position（与现有日志统计字段保持一致）。
    """
    agent_ideas: dict[str, list[str]] = {}
    agent_meta: dict[str, dict[str, Any]] = {}

    # 先从 metadata.agents 建一个 agent_id -> config_key 的映射（更稳）
    agent_id_to_config: dict[int, str] = {}
    for a in data.get("metadata", {}).get("agents", []) or []:
        try:
            aid = int(a.get("agent_id"))
        except Exception:
            continue
        ck = a.get("config_key")
        if isinstance(ck, str) and ck:
            agent_id_to_config[aid] = ck

    for turn in data.get("global_history", []):
        content = turn.get("content", "").strip()
        if not content:
            continue

        # 过滤非必要角色，如系统裁判等
        role = turn.get("role", "")
        if role in ["system", "moderator"]:
            continue

        agent_id = turn.get("agent_id", None)
        try:
            agent_id_int: int | None = int(agent_id) if agent_id is not None else None
        except Exception:
            agent_id_int = None

        agent_name = turn.get("agent_name", None)
        if not agent_name:
            agent_name = f"Agent {agent_id_int if agent_id_int is not None else '?'}"

        if agent_name not in agent_ideas:
            agent_ideas[agent_name] = []
        agent_ideas[agent_name].append(content)

        # 记录一次 meta（以 first-seen 为准）
        if agent_name not in agent_meta:
            config_key = turn.get("config_key")
            if (not isinstance(config_key, str) or not config_key) and agent_id_int is not None:
                config_key = agent_id_to_config.get(agent_id_int, None)

            agent_meta[agent_name] = {
                "agent_id": agent_id_int,
                "position": agent_id_int,  # position 与 agent_id 对齐（日志里常用这种口径）
                "config_key": config_key,
            }

    out: dict[str, dict[str, Any]] = {}
    for name, texts in agent_ideas.items():
        meta = agent_meta.get(name, {})
        out[name] = {
            "idea_content": "\n\n".join(texts),
            "agent_id": meta.get("agent_id"),
            "position": meta.get("position"),
            "config_key": meta.get("config_key"),
        }
    return out

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
    
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    topic = data.get("metadata", {}).get("topic", "")
    if not topic:
        print(f"  [WARN] No topic in {filepath}, using empty string.")

    # 这里改为获取所有 Agent 的独立想法集合
    agent_ideas_dict = extract_agent_ideas(data)
    if not agent_ideas_dict:
        print(f"  [SKIP] No agent content in {filepath}")
        return False

    # 创建新的根字典字段
    data["cpss_evaluation_per_agent"] = {}
    file_all_ok = True

    # 针对每个 Agent 发起 55 个维度的评测
    for agent_name, payload in agent_ideas_dict.items():
        idea_content = payload.get("idea_content", "")
        agent_id = payload.get("agent_id")
        position = payload.get("position")
        config_key = payload.get("config_key")

        print(f"    Evaluating Agent: {agent_name} ...")
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
        agent_all_ok = True
        
        for key, score in results:
            cpss_evaluation[key] = score
            if score is None:
                agent_all_ok = False
                file_all_ok = False
                
        # 保存特定 Agent 的分数
        data["cpss_evaluation_per_agent"][agent_name] = {
            "agent_id": agent_id,
            "position": position,
            "config_key": config_key,
            "scores": cpss_evaluation,
        }
        if agent_all_ok:
            print(f"      => {agent_name} OK (55/55)")
        else:
            print(f"      => {agent_name} PARTIAL (some scores are null)")

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return file_all_ok

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
                success_count += 1
            else:
                fail_count += 1
        except Exception as e:
            print(f"  => FAILED ({e})")
            fail_count += 1

    print(f"\nDone. Success files: {success_count}, Files with issues: {fail_count}")

def main() -> None:
    asyncio.run(async_main())

if __name__ == "__main__":
    main()