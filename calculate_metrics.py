"""
Brainstorm 对话日志指标计算脚本

批量处理指定文件夹下的所有 JSON 日志文件，为每条发言计算：
  - Distinct-n (n-gram 多样性)
  - Entropy-n  (n-gram 信息熵)
  - Sentence-BERT Similarity (与 Topic 的语义相关性)
  - Max BLEU (与可见上下文中历史发言的最高 BLEU 相似度)

计算结果注入到原 JSON 结构中（turn-level + agent-level 聚合）。
"""

import argparse
import glob
import json
import math
import os
import re
from collections import Counter, defaultdict

from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from sentence_transformers import SentenceTransformer, util

# ======================== 可配置参数 ========================

FOLDER_ADDRESS = "log"
N_GRAM_LIST = [1, 2]
SBERT_MODEL_NAME = "all-MiniLM-L6-v2"
OUTPUT_MODE = "overwrite"  # "overwrite" | "copy"

# ===========================================================


def tokenize(text: str) -> list[str]:
    """英文分词：提取单词并转小写，去除标点。"""
    return re.findall(r"\b\w+\b", text.lower())


def get_ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    """从 token 列表中提取所有 n-gram。"""
    if len(tokens) < n:
        return []
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def calc_distinct_n(tokens: list[str], n: int) -> float:
    """计算 Distinct-n 指标。"""
    ngrams = get_ngrams(tokens, n)
    if not ngrams:
        return 0.0
    return len(set(ngrams)) / len(ngrams)


def calc_entropy_n(tokens: list[str], n: int) -> float:
    """计算 n-gram 信息熵 (Shannon Entropy, base-2)。"""
    ngrams = get_ngrams(tokens, n)
    if not ngrams:
        return 0.0
    counts = Counter(ngrams)
    total = len(ngrams)
    entropy = 0.0
    for count in counts.values():
        p = count / total
        entropy -= p * math.log2(p)
    return entropy


def compute_turn_metrics(
    content: str,
    topic_embedding,
    sbert_model: SentenceTransformer,
    n_gram_list: list[int],
) -> dict:
    """计算单条发言的全部指标，返回 metric 字典。"""
    tokens = tokenize(content)
    metric = {}

    for n in n_gram_list:
        metric[f"distinct_{n}"] = round(calc_distinct_n(tokens, n), 4)
    for n in n_gram_list:
        metric[f"entropy_{n}"] = round(calc_entropy_n(tokens, n), 4)

    content_embedding = sbert_model.encode(content, convert_to_tensor=True)
    sim = util.cos_sim(content_embedding, topic_embedding).item()
    metric["sbert_sim_to_topic"] = round(sim, 4)

    return metric


_SPEAKER_LINE_RE = re.compile(r"- Agent \d+ (?:say|的草稿)[：:]\s*")
_TRAILING_INSTR_RE = re.compile(
    r"\n(?:Please respond based on|Be concise|请根据|请仔细阅读|请在此基础上).*$",
    re.DOTALL,
)


def _extract_utterances_from_user_content(content: str) -> list[str]:
    """从 user-role 消息中提取其他 Agent 的发言文本。"""
    markers = list(_SPEAKER_LINE_RE.finditer(content))
    if not markers:
        return []
    utterances = []
    for i, m in enumerate(markers):
        start = m.end()
        end = markers[i + 1].start() if i + 1 < len(markers) else len(content)
        utterance = content[start:end].strip()
        if utterance:
            utterances.append(utterance)
    if utterances:
        utterances[-1] = _TRAILING_INSTR_RE.sub("", utterances[-1]).strip()
    return [u for u in utterances if u]


def compute_max_bleu_scores(final_messages: dict) -> dict[int, list[float]]:
    """
    从 final_messages 重构每个 Agent 的可见上下文历史，
    计算每次发言与所有可见历史的 Max BLEU。

    Returns: {agent_id: [score_round_1, score_round_2, ...]}
    """
    smoother = SmoothingFunction().method1
    result: dict[int, list[float]] = {}

    for agent_id_str, messages in final_messages.items():
        agent_id = int(agent_id_str)
        observed_history: list[str] = []
        max_bleu_scores: list[float] = []

        for msg in messages:
            role = msg["role"]
            content = msg.get("content", "")

            if role == "user":
                utterances = _extract_utterances_from_user_content(content)
                observed_history.extend(utterances)

            elif role == "assistant":
                if not observed_history:
                    max_bleu_scores.append(0.0)
                else:
                    hypothesis = tokenize(content)
                    if not hypothesis:
                        max_bleu_scores.append(0.0)
                    else:
                        best = 0.0
                        for hist in observed_history:
                            reference = tokenize(hist)
                            if not reference:
                                continue
                            score = sentence_bleu(
                                [reference],
                                hypothesis,
                                smoothing_function=smoother,
                            )
                            best = max(best, score)
                        max_bleu_scores.append(round(best, 4))
                observed_history.append(content)

        result[agent_id] = max_bleu_scores

    return result


def compute_agent_metrics(
    global_history: list[dict],
    agents_info: list[dict],
    n_gram_list: list[int],
) -> list[dict]:
    """
    聚合每个 Agent 在整局中的平均指标。
    agents_info 来自 metadata.agents，用于获取 agent_id / config_key。
    """
    agent_metrics_accum: dict[int, list[dict]] = defaultdict(list)
    for turn in global_history:
        if "metric" in turn:
            agent_metrics_accum[turn["agent_id"]].append(turn["metric"])

    agent_id_to_info = {}
    for a in agents_info:
        agent_id_to_info[a["agent_id"]] = a

    position_map_lookup = {}
    for a in agents_info:
        position_map_lookup[a["agent_id"]] = a.get("agent_id", a["agent_id"])

    result = []
    for agent_id in sorted(agent_metrics_accum.keys()):
        metrics_list = agent_metrics_accum[agent_id]
        if not metrics_list:
            continue

        all_keys = metrics_list[0].keys()
        avg_metrics = {}
        for key in all_keys:
            values = [m[key] for m in metrics_list]
            avg_key = f"avg_{key}" if not key.startswith("avg_") else key
            avg_metrics[avg_key] = round(sum(values) / len(values), 4)

        info = agent_id_to_info.get(agent_id, {})
        result.append(
            {
                "agent_id": agent_id,
                "position": agent_id,
                "config_key": info.get("config_key", "unknown"),
                "avg_metrics": avg_metrics,
            }
        )

    return result


def process_single_file(
    filepath: str,
    sbert_model: SentenceTransformer,
    n_gram_list: list[int],
) -> dict:
    """处理单个 JSON 日志文件，注入指标后返回更新后的数据。"""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    topic = data.get("metadata", {}).get("topic", "")
    if not topic:
        print(f"  [WARN] No topic found in {filepath}, skipping SBERT similarity.")

    topic_embedding = sbert_model.encode(topic, convert_to_tensor=True)

    bleu_scores_map: dict[int, list[float]] = {}
    if "final_messages" in data:
        bleu_scores_map = compute_max_bleu_scores(data["final_messages"])

    turn_index_tracker: dict[int, int] = defaultdict(int)

    for turn in data["global_history"]:
        content = turn.get("content", "")
        turn["metric"] = compute_turn_metrics(
            content, topic_embedding, sbert_model, n_gram_list
        )

        aid = turn["agent_id"]
        idx = turn_index_tracker[aid]
        scores = bleu_scores_map.get(aid, [])
        turn["metric"]["max_bleu"] = scores[idx] if idx < len(scores) else 0.0
        turn_index_tracker[aid] += 1

    agents_info = data.get("metadata", {}).get("agents", [])
    data["agent_metrics"] = compute_agent_metrics(
        data["global_history"], agents_info, n_gram_list
    )

    return data


def main():
    parser = argparse.ArgumentParser(
        description="Brainstorm 日志指标计算脚本 (Distinct-n, Entropy-n, SBERT Similarity)"
    )
    parser.add_argument(
        "--dir",
        type=str,
        default=None,
        help="日志文件夹路径（覆盖脚本内的 FOLDER_ADDRESS）",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        choices=["overwrite", "copy"],
        help="输出模式：overwrite=覆盖原文件，copy=输出到新文件夹（覆盖脚本内的 OUTPUT_MODE）",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Sentence-BERT 模型名称（覆盖脚本内的 SBERT_MODEL_NAME）",
    )
    args = parser.parse_args()

    folder = args.dir if args.dir else FOLDER_ADDRESS
    output_mode = args.output if args.output else OUTPUT_MODE
    model_name = args.model if args.model else SBERT_MODEL_NAME

    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

    if not os.path.isdir(folder):
        print(f"[ERROR] Folder not found: {folder}")
        return

    json_files = sorted(glob.glob(os.path.join(folder, "*.json")))
    if not json_files:
        print(f"[INFO] No JSON files found in {folder}")
        return

    print(f"Loading Sentence-BERT model: {model_name} ...")
    sbert_model = SentenceTransformer(model_name)
    print(f"Model loaded. Processing {len(json_files)} files from '{folder}' ...\n")

    if output_mode == "copy":
        out_folder = folder.rstrip("/\\") + "_metrics_added"
        os.makedirs(out_folder, exist_ok=True)
    else:
        out_folder = folder

    for filepath in json_files:
        filename = os.path.basename(filepath)
        print(f"  Processing: {filename} ... ", end="", flush=True)
        try:
            updated = process_single_file(filepath, sbert_model, N_GRAM_LIST)
            out_path = os.path.join(out_folder, filename)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(updated, f, ensure_ascii=False, indent=2)
            print("OK")
        except Exception as e:
            print(f"FAILED ({e})")

    print(f"\nDone. Output folder: {out_folder}")


if __name__ == "__main__":
    main()
