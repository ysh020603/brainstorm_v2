"""
大模型人工评估 Elo 分数与置信度计算脚本

功能：
1. 读取 ex1 (3port) 和 ex2 (final_rankings) 格式的人类打分日志。
2. 将全序排名转换为两两对战 (Pairwise Comparisons) 关系。
3. 计算每个大模型 (config_key) 的 Elo 得分。
4. 使用 Bootstrap (重采样) 技术给出 95% 置信区间 (Confidence Intervals)。
"""

import os
import json
import random
import math
from collections import defaultdict

# ======================== 可配置参数 ========================
EX1_DIR = "log_experiment/ex1_4LLM"
EX2_DIR = "log_ex2_1human3LLM"

BASE_ELO = 1200.0   # 初始 Elo 分数
K_FACTOR = 16.0     # Elo K 因子 (决定单局分数变动幅度)
BOOTSTRAP_ITERS = 10000  # 重采样次数
CI_LOWER = 0.025    # 置信区间下界 (2.5%)
CI_UPPER = 0.975    # 置信区间上界 (97.5%)
# ===========================================================


def expected_score(rating_a: float, rating_b: float) -> float:
    """计算选手 A 对战选手 B 的期望胜率"""
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def extract_matches_from_logs(ex1_dir: str, ex2_dir: str) -> tuple[list, int, int]:
    """
    遍历指定目录，解析 JSON 日志，提取所有有效的排名数组。
    返回的格式为：([[{'config_key': 'A', 'rank': 1}, ...], ...], ex1条数, ex2条数)
    每一个子列表代表一位人类给出的一次完整排名记录。
    """
    matches = []
    ex1_count = 0
    ex2_count = 0

    # 1. 解析 ex1_4LLM (3port 格式)
    if os.path.exists(ex1_dir):
        for root, _, files in os.walk(ex1_dir):
            for file in files:
                if file.endswith(".json"):
                    path = os.path.join(root, file)
                    try:
                        with open(path, "r", encoding="utf-8") as f:
                            data = json.load(f)
                            # 如果存在 3port 字段，遍历里面的每个用户
                            if "3port" in data and isinstance(data["3port"], dict):
                                for user_name, ranking_list in data["3port"].items():
                                    if ranking_list:
                                        matches.append(ranking_list)
                                        ex1_count += 1
                    except Exception as e:
                        print(f"读取文件出错 {path}: {e}")

    # 2. 解析 ex2_1human3LLM (final_rankings 格式)
    if os.path.exists(ex2_dir):
        for root, _, files in os.walk(ex2_dir):
            for file in files:
                if file.endswith(".json"):
                    path = os.path.join(root, file)
                    try:
                        with open(path, "r", encoding="utf-8") as f:
                            data = json.load(f)
                            if "final_rankings" in data and isinstance(data["final_rankings"], list):
                                if data["final_rankings"]:
                                    matches.append(data["final_rankings"])
                                    ex2_count += 1
                    except Exception as e:
                        print(f"读取文件出错 {path}: {e}")

    return matches, ex1_count, ex2_count


def update_elo_for_single_match(elo_dict: dict, ranking: list, k_factor: float):
    """
    处理单次评估中的两两对战。
    为避免排名靠前的模型在同一个评估记录内被吃掉过多红利（顺序偏见），
    我们先结算所有对局的预期和真实得分差，最后统一下发 Elo 增减。
    """
    deltas = defaultdict(float)
    
    n = len(ranking)
    for i in range(n):
        for j in range(i + 1, n):
            model_a = ranking[i].get("config_key")
            rank_a = ranking[i].get("rank")
            
            model_b = ranking[j].get("config_key")
            rank_b = ranking[j].get("rank")
            
            if not model_a or not model_b or rank_a is None or rank_b is None:
                continue
                
            rating_a = elo_dict[model_a]
            rating_b = elo_dict[model_b]
            
            # 计算双方的期望胜率
            ea = expected_score(rating_a, rating_b)
            eb = expected_score(rating_b, rating_a)
            
            # 根据名次决定真实胜率 (数字越小名次越高)
            if rank_a < rank_b:
                sa, sb = 1.0, 0.0  # A 赢
            elif rank_a > rank_b:
                sa, sb = 0.0, 1.0  # B 赢
            else:
                sa, sb = 0.5, 0.5  # 平局 (严谨全序下不应出现，仅为兜底)
                
            # 累积该局变更值
            deltas[model_a] += k_factor * (sa - ea)
            deltas[model_b] += k_factor * (sb - eb)
            
    # 统一应用得分变更
    for model, delta in deltas.items():
        elo_dict[model] += delta


def compute_elo_from_matches(matches: list, k_factor: float, base_rating: float) -> dict:
    """计算给定所有比赛记录的 Elo 值"""
    # 默认积分为 BASE_ELO
    elo_dict = defaultdict(lambda: base_rating)
    
    # 因为 Elo 计分对比赛的出场顺序敏感，我们在每次计算前对其进行随机打乱，
    # 消除读取文件时的顺序偏见。
    shuffled_matches = list(matches)
    random.shuffle(shuffled_matches)
    
    for match in shuffled_matches:
        update_elo_for_single_match(elo_dict, match, k_factor)
        
    return dict(elo_dict)


def main():
    print("开始加载数据...")
    matches, n_ex1, n_ex2 = extract_matches_from_logs(EX1_DIR, EX2_DIR)
    total_matches = len(matches)

    print(f"  [{EX1_DIR}] 加载 {n_ex1} 条排名记录（3port，按标注用户计）")
    print(f"  [{EX2_DIR}] 加载 {n_ex2} 条排名记录（final_rankings，按日志文件计）")

    if total_matches == 0:
        print("未在指定目录下找到有效的人类标注数据！请检查路径。")
        return

    print(f"合计有效排名记录：{total_matches} 条。\n")
    print(f"正在进行 {BOOTSTRAP_ITERS} 次重采样以计算置信度，请稍候...")
    
    # 记录每一个模型在多次重采样中的 Elo 值
    bootstrap_results = defaultdict(list)
    
    for i in range(BOOTSTRAP_ITERS):
        # 有放回抽样 (Sampling with replacement)
        resampled_matches = random.choices(matches, k=total_matches)
        
        # 计算该批次抽样的 Elo
        sample_elo = compute_elo_from_matches(resampled_matches, K_FACTOR, BASE_ELO)
        
        for model, elo_val in sample_elo.items():
            bootstrap_results[model].append(elo_val)
            
    # 汇总结果
    final_stats = []
    
    # 我们用重采样分布的中位数作为最稳定的综合表现值
    for model, elo_list in bootstrap_results.items():
        elo_list.sort()
        n_samples = len(elo_list)
        
        median_elo = elo_list[n_samples // 2]
        lower_bound = elo_list[int(n_samples * CI_LOWER)]
        upper_bound = elo_list[int(n_samples * CI_UPPER)]
        
        final_stats.append({
            "model": model,
            "median_elo": median_elo,
            "lower_95": lower_bound,
            "upper_95": upper_bound
        })
        
    # 按中位数 Elo 从高到低排序
    final_stats.sort(key=lambda x: x["median_elo"], reverse=True)
    
    # 打印排版表格
    print("-" * 75)
    print(f"{'模型配置 (Config Key)':<30} | {'稳定 Elo (中位数)':<15} | {'95% 置信区间'}")
    print("-" * 75)
    for stat in final_stats:
        model_name = stat["model"]
        elo = stat["median_elo"]
        lower = stat["lower_95"]
        upper = stat["upper_95"]
        
        # 截断超长的模型名字以便对齐
        if len(model_name) > 28:
            model_name = model_name[:25] + "..."
            
        print(f"{model_name:<30} | {elo:>10.2f}      | [{lower:.2f}, {upper:.2f}]")
    print("-" * 75)
    print("注：稳定 Elo 采用 Bootstrap 重采样中位数。由于数据规模原因，部分冷门模型可能区间较宽。")

if __name__ == "__main__":
    main()