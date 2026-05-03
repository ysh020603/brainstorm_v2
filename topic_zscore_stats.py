"""
按“讨论范式(mode)内、跨 topic”计算客观指标的 Z-score。

样本定义：
- 一个样本 = 某个实验 JSON 中某个 agent(config_key) 的一组指标（优先用 agent_metrics.avg_metrics）。

标准化定义：
- 对每个 mode、topic、metric，先计算该分组的均值与标准差；
- 对组内任一样本 x，计算 z = (x - mean) / std；
- std=0 时该 z 记为 None。
"""

import argparse
import csv
import glob
import json
import math
import os
import re
from collections import defaultdict
from statistics import mean, pstdev
from typing import Optional
import matplotlib.pyplot as plt


DEFAULT_LOG_ROOT = "/data2/brainstorm/brainstorm_v2/eval/ex1_4LLM"
_WORD_RE = re.compile(r"\b\w+\b")


def tokenize(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def _safe_float(value):
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def collect_json_files(log_root: str) -> list[str]:
    pattern = os.path.join(log_root, "**", "*.json")
    return sorted(glob.glob(pattern, recursive=True))


def parse_mode_topic_from_path(filepath: str) -> tuple[str, str]:
    # 目录结构：.../ex1_4LLM/<topic_dir>/<mode>/<file>.json
    mode = os.path.basename(os.path.dirname(filepath))
    topic_dir = os.path.basename(os.path.dirname(os.path.dirname(filepath)))
    return mode, topic_dir


def extract_basic_metrics(data: dict) -> dict:
    """
    若 agent_metrics 不存在，回退到 global_history 聚合基础统计量。
    返回: {agent_id: metric_dict}
    """
    by_agent = defaultdict(lambda: {"turn_count": 0, "char_count": 0, "token_count": 0})
    for turn in data.get("global_history", []):
        aid = turn.get("agent_id")
        content = turn.get("content", "")
        if aid is None or not isinstance(content, str):
            continue
        by_agent[aid]["turn_count"] += 1
        by_agent[aid]["char_count"] += len(content)
        by_agent[aid]["token_count"] += len(tokenize(content))

    for aid, m in by_agent.items():
        turns = m["turn_count"]
        m["avg_chars_per_turn"] = (m["char_count"] / turns) if turns else 0.0
        m["avg_tokens_per_turn"] = (m["token_count"] / turns) if turns else 0.0
    return by_agent


def extract_agent_metrics(data: dict) -> dict:
    """
    优先读取 data["agent_metrics"][*]["avg_metrics"]。
    若缺失则回退到 extract_basic_metrics。
    返回: {agent_id: {metric_name: float}}
    """
    result = defaultdict(dict)
    has_agent_metrics = False
    for item in data.get("agent_metrics", []):
        aid = item.get("agent_id")
        avg_metrics = item.get("avg_metrics", {})
        if aid is None or not isinstance(avg_metrics, dict):
            continue
        has_agent_metrics = True
        for k, v in avg_metrics.items():
            fv = _safe_float(v)
            if fv is not None:
                result[aid][k] = fv

    if has_agent_metrics:
        return result
    return extract_basic_metrics(data)


def extract_agent_config_map(data: dict) -> dict:
    id2cfg = {}
    for a in data.get("metadata", {}).get("agents", []):
        aid = a.get("agent_id")
        if aid is not None:
            id2cfg[aid] = a.get("config_key", "unknown")
    return id2cfg


def gather_samples(log_root: str) -> tuple[list[dict], list[str], int]:
    files = collect_json_files(log_root)
    samples = []
    metric_keys = set()
    bad_files = 0

    for fp in files:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            bad_files += 1
            continue

        mode_from_path, topic_from_path = parse_mode_topic_from_path(fp)
        # 以路径目录名为准，确保 leader_worker 与 leader_worker_22 被严格区分
        mode = mode_from_path
        topic = data.get("metadata", {}).get("topic", topic_from_path)
        agent_metrics = extract_agent_metrics(data)
        config_map = extract_agent_config_map(data)

        for aid in sorted(agent_metrics.keys()):
            row = {
                "mode": str(mode).strip(),
                "topic": str(topic).strip(),
                "topic_dir": topic_from_path,
                "file_path": fp,
                "file_name": os.path.basename(fp),
                "agent_id": aid,
                "config_key": config_map.get(aid, f"agent_{aid}"),
                "metrics": {},
                "z_metrics": {},
            }
            for mk, mv in agent_metrics[aid].items():
                fv = _safe_float(mv)
                if fv is None:
                    continue
                row["metrics"][mk] = fv
                metric_keys.add(mk)
            if row["metrics"]:
                samples.append(row)

    return samples, sorted(metric_keys), bad_files


def gather_round_metric_samples(log_root: str) -> tuple[list[dict], list[str]]:
    """
    提取 turn-level 指标：用于“轮数变化”可视化。
    每条样本对应 global_history 的一条 turn。
    """
    rows = []
    metric_keys = set()
    for fp in collect_json_files(log_root):
        try:
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue

        mode, topic_dir = parse_mode_topic_from_path(fp)
        topic = data.get("metadata", {}).get("topic", topic_dir)
        for turn in data.get("global_history", []):
            rd = turn.get("round")
            metric = turn.get("metric", {})
            if rd is None or not isinstance(metric, dict):
                continue
            for mk, mv in metric.items():
                fv = _safe_float(mv)
                if fv is None:
                    continue
                rows.append(
                    {
                        "mode": mode,
                        "topic": str(topic).strip(),
                        "topic_dir": topic_dir,
                        "round": int(rd),
                        "metric": mk,
                        "value": fv,
                    }
                )
                metric_keys.add(mk)
    return rows, sorted(metric_keys)


def compute_mode_topic_stats(samples: list[dict], metric_keys: list[str]) -> dict:
    # stats[mode][topic][metric] = {mean, std, count}
    values = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for s in samples:
        mode = s["mode"]
        topic = s["topic"]
        for mk in metric_keys:
            v = s["metrics"].get(mk)
            if v is not None:
                values[mode][topic][mk].append(v)

    stats = defaultdict(lambda: defaultdict(dict))
    for mode, topic_map in values.items():
        for topic, metric_map in topic_map.items():
            for mk, arr in metric_map.items():
                if not arr:
                    continue
                mu = mean(arr)
                sd = pstdev(arr) if len(arr) > 1 else 0.0
                stats[mode][topic][mk] = {"mean": mu, "std": sd, "count": len(arr)}
    return stats


def apply_zscore(samples: list[dict], stats: dict, metric_keys: list[str]) -> None:
    for s in samples:
        mode = s["mode"]
        topic = s["topic"]
        for mk in metric_keys:
            x = s["metrics"].get(mk)
            st = stats.get(mode, {}).get(topic, {}).get(mk)
            if x is None or not st:
                continue
            sd = st["std"]
            s["z_metrics"][mk] = None if sd == 0 else (x - st["mean"]) / sd


def _aggregate_mean(items: list[float]) -> Optional[float]:
    return mean(items) if items else None


def write_sample_csv(path: str, samples: list[dict], metric_keys: list[str]) -> None:
    fields = ["mode", "topic", "topic_dir", "file_name", "agent_id", "config_key"]
    fields += [f"raw::{k}" for k in metric_keys]
    fields += [f"z::{k}" for k in metric_keys]

    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for s in samples:
            row = {
                "mode": s["mode"],
                "topic": s["topic"],
                "topic_dir": s["topic_dir"],
                "file_name": s["file_name"],
                "agent_id": s["agent_id"],
                "config_key": s["config_key"],
            }
            for k in metric_keys:
                row[f"raw::{k}"] = s["metrics"].get(k)
                row[f"z::{k}"] = s["z_metrics"].get(k)
            w.writerow(row)


def write_model_mode_topic_summary(path: str, samples: list[dict], metric_keys: list[str]) -> None:
    # 每行：某模型在某 mode 的某 topic 的均值
    grouped_raw = defaultdict(lambda: defaultdict(list))
    grouped_z = defaultdict(lambda: defaultdict(list))
    # key: (mode, topic, config_key)
    for s in samples:
        key = (s["mode"], s["topic"], s["config_key"])
        for k in metric_keys:
            rv = s["metrics"].get(k)
            zv = s["z_metrics"].get(k)
            if rv is not None:
                grouped_raw[key][k].append(rv)
            if zv is not None:
                grouped_z[key][k].append(zv)

    fields = ["mode", "topic", "config_key", "sample_count"]
    fields += [f"mean_raw::{k}" for k in metric_keys]
    fields += [f"mean_z::{k}" for k in metric_keys]
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for mode, topic, cfg in sorted(grouped_raw.keys()):
            row = {
                "mode": mode,
                "topic": topic,
                "config_key": cfg,
                "sample_count": len(next(iter(grouped_raw[(mode, topic, cfg)].values()), [])),
            }
            for k in metric_keys:
                row[f"mean_raw::{k}"] = _aggregate_mean(grouped_raw[(mode, topic, cfg)][k])
                row[f"mean_z::{k}"] = _aggregate_mean(grouped_z[(mode, topic, cfg)][k])
            w.writerow(row)


def write_model_mode_summary(path: str, samples: list[dict], metric_keys: list[str]) -> None:
    # 每行：某模型在某 mode 下跨所有 topic 的均值（用于范式内总体排名）
    grouped_raw = defaultdict(lambda: defaultdict(list))
    grouped_z = defaultdict(lambda: defaultdict(list))
    # key: (mode, config_key)
    for s in samples:
        key = (s["mode"], s["config_key"])
        for k in metric_keys:
            rv = s["metrics"].get(k)
            zv = s["z_metrics"].get(k)
            if rv is not None:
                grouped_raw[key][k].append(rv)
            if zv is not None:
                grouped_z[key][k].append(zv)

    fields = ["mode", "config_key", "sample_count"]
    fields += [f"mean_raw::{k}" for k in metric_keys]
    fields += [f"mean_z::{k}" for k in metric_keys]
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for mode, cfg in sorted(grouped_raw.keys()):
            row = {
                "mode": mode,
                "config_key": cfg,
                "sample_count": len(next(iter(grouped_raw[(mode, cfg)].values()), [])),
            }
            for k in metric_keys:
                row[f"mean_raw::{k}"] = _aggregate_mean(grouped_raw[(mode, cfg)][k])
                row[f"mean_z::{k}"] = _aggregate_mean(grouped_z[(mode, cfg)][k])
            w.writerow(row)


def write_stats_json(path: str, stats: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)


def _mean_by_key(rows: list[dict], key_fields: tuple[str, ...]) -> dict:
    grouped = defaultdict(list)
    for r in rows:
        key = tuple(r[k] for k in key_fields)
        grouped[key].append(r["value"])
    out = {}
    for key, vals in grouped.items():
        out[key] = mean(vals)
    return out


def render_mode_topic_model_heatmaps(samples: list[dict], out_png: str) -> None:
    """
    可视化1：
    每个 mode 一个热力图，Y轴=model，X轴=topic，值=overall_z(各指标 z 的均值)。
    """
    rows = []
    for s in samples:
        z_vals = [z for z in s["z_metrics"].values() if z is not None]
        if not z_vals:
            continue
        rows.append(
            {
                "mode": s["mode"],
                "topic": s["topic"],
                "config_key": s["config_key"],
                "value": mean(z_vals),
            }
        )
    if not rows:
        return

    agg = _mean_by_key(rows, ("mode", "topic", "config_key"))
    modes = sorted({r["mode"] for r in rows})
    topics = sorted({r["topic"] for r in rows})
    models = sorted({r["config_key"] for r in rows})

    n = len(modes)
    fig, axes = plt.subplots(1, n, figsize=(4.8 * n, max(8, 0.45 * len(models))), squeeze=False)
    vlim = max(abs(v) for v in agg.values()) if agg else 1.0
    vlim = max(vlim, 1e-6)

    for i, mode in enumerate(modes):
        ax = axes[0][i]
        mat = []
        for model in models:
            row = []
            for topic in topics:
                row.append(agg.get((mode, topic, model), float("nan")))
            mat.append(row)

        im = ax.imshow(mat, cmap="RdBu_r", vmin=-vlim, vmax=vlim, aspect="auto")
        ax.set_title(mode)
        ax.set_xticks(range(len(topics)))
        ax.set_xticklabels(topics, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels(models, fontsize=8)
        ax.set_xlabel("Topic")
        if i == 0:
            ax.set_ylabel("Model")

    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.85)
    cbar.set_label("Mean overall Z-score")
    fig.suptitle("Model vs Topic Z-score Heatmaps by Discussion Mode", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def render_round_trend_figures(round_rows: list[dict], out_dir: str) -> list[str]:
    """
    可视化2：
    对每个 turn-level metric 生成一张图（6个topic子图，线条为不同mode）。
    X轴=round，Y轴=该metric均值。
    """
    if not round_rows:
        return []

    by_key = _mean_by_key(round_rows, ("metric", "topic", "mode", "round"))
    metrics = sorted({r["metric"] for r in round_rows})
    topics = sorted({r["topic"] for r in round_rows})
    modes = sorted({r["mode"] for r in round_rows})
    rounds = sorted({r["round"] for r in round_rows})
    out_files = []

    for metric in metrics:
        fig, axes = plt.subplots(2, 3, figsize=(18, 9), sharex=True)
        axes = axes.ravel()
        for idx, topic in enumerate(topics[:6]):
            ax = axes[idx]
            for mode in modes:
                ys = [by_key.get((metric, topic, mode, rd), float("nan")) for rd in rounds]
                ax.plot(rounds, ys, marker="o", linewidth=1.8, label=mode)
            ax.set_title(topic, fontsize=10)
            ax.set_xticks(rounds)
            ax.grid(alpha=0.25)
            if idx % 3 == 0:
                ax.set_ylabel(metric)
        for idx in range(len(topics[:6]), 6):
            axes[idx].axis("off")
        axes[0].legend(loc="best", fontsize=8)
        fig.suptitle(f"Round-wise trend for {metric}", fontsize=14)
        fig.tight_layout()
        out_path = os.path.join(out_dir, f"round_trend_{metric}.png")
        fig.savefig(out_path, dpi=220, bbox_inches="tight")
        plt.close(fig)
        out_files.append(out_path)

    return out_files


def main():
    parser = argparse.ArgumentParser(
        description="在每个讨论范式(mode)内，按 topic 计算客观指标 Z-score"
    )
    parser.add_argument("--log-root", type=str, default=DEFAULT_LOG_ROOT, help="eval 实验日志根目录")
    parser.add_argument("--out-dir", type=str, default="./zscore_output", help="输出目录")
    parser.add_argument(
        "--plot",
        action="store_true",
        help="额外生成两个可视化：热力图 + 按轮数趋势图",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.log_root):
        print(f"[ERROR] log root not found: {args.log_root}")
        return
    os.makedirs(args.out_dir, exist_ok=True)

    samples, metric_keys, bad_files = gather_samples(args.log_root)
    if not samples:
        print("[INFO] no valid samples found.")
        return

    stats = compute_mode_topic_stats(samples, metric_keys)
    apply_zscore(samples, stats, metric_keys)

    sample_csv = os.path.join(args.out_dir, "mode_topic_zscore_samples.csv")
    mode_topic_summary_csv = os.path.join(args.out_dir, "mode_topic_model_summary.csv")
    mode_summary_csv = os.path.join(args.out_dir, "mode_model_summary.csv")
    stats_json = os.path.join(args.out_dir, "mode_topic_metric_stats.json")

    write_sample_csv(sample_csv, samples, metric_keys)
    write_model_mode_topic_summary(mode_topic_summary_csv, samples, metric_keys)
    write_model_mode_summary(mode_summary_csv, samples, metric_keys)
    write_stats_json(stats_json, stats)

    file_count = len({s["file_path"] for s in samples})
    mode_count = len({s["mode"] for s in samples})
    topic_count = len({(s["mode"], s["topic"]) for s in samples})
    model_count = len({s["config_key"] for s in samples})
    print(f"[OK] files parsed: {file_count}, bad files: {bad_files}")
    print(f"[OK] modes: {mode_count}, mode-topic groups: {topic_count}, models: {model_count}")
    print(f"[OK] samples: {len(samples)}, metrics: {len(metric_keys)}")
    print(f"[OUT] {sample_csv}")
    print(f"[OUT] {mode_topic_summary_csv}")
    print(f"[OUT] {mode_summary_csv}")
    print(f"[OUT] {stats_json}")

    if args.plot:
        heatmap_png = os.path.join(args.out_dir, "mode_topic_model_zscore_heatmaps.png")
        render_mode_topic_model_heatmaps(samples, heatmap_png)
        print(f"[OUT] {heatmap_png}")

        round_rows, round_metrics = gather_round_metric_samples(args.log_root)
        trend_pngs = render_round_trend_figures(round_rows, args.out_dir)
        print(
            f"[OK] round metrics found: {len(round_metrics)}, trend figures: {len(trend_pngs)}"
        )
        for p in trend_pngs:
            print(f"[OUT] {p}")


if __name__ == "__main__":
    main()
