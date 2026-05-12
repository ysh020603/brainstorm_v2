import json
import os
import re
from pathlib import Path
import argparse

# ======================== 配置项 ========================
# 待处理的日志文件夹路径
LOG_DIR = "/data2/brainstorm/brainstorm_v2/log_cpss_human"

# 【新增】由 human 设置，只处理一模一样的 key
TARGET_EVAL_KEY = "cpss_evaluation_per_agent_deepseek-v4-flash_few_shot" 

# 7点计分反向转换逻辑：8 - 原始分
# 根据“高分对应高创意”原则，以下题项（当 1=创意端，7=非创意端时）需要反向计分
REVERSE_ITEMS = [
    4, 6, 7, 8, 10, 13, 14, 16, 19, 21, 22, 25, 26, 31, 34, 35, 36, 37, 43, 46, 47, 51, 53, 54
]

# CPSS 11个亚量表与题号的标准映射
SUBSCALES_MAPPING = {
    "Original": [6, 8, 13, 16, 35, 43, 46],
    "Surprising": [2, 14, 22, 25, 36, 37],
    "Germinal": [1, 4, 7, 26, 31, 34, 55],
    "Valuable": [5, 12, 21, 32, 41, 48, 53],
    "Logical": [3, 11, 15, 29, 47, 52],
    "Useful": [9, 10, 17, 33, 54],
    "Organic": [19, 27, 44, 50],
    "Elegant": [18, 28],
    "Complex": [23, 51],
    "Understandable": [24, 30, 40],
    "Well-Crafted": [20, 38, 39, 42, 45, 49]
}
# =======================================================

def calculate_metrics(raw_scores):
    """根据原始分数计算全套 CPSS 指标"""
    processed_scores = {}
    
    # 1. 预处理：提取分数并执行反向计分
    for q_key, val in raw_scores.items():
        # 提取题号，例如 "Q06_Original_Conventional" -> 6
        match = re.match(r"Q(\d+)_", q_key)
        if not match:
            continue
        q_num = int(match.group(1))
        
        # 执行反向转换
        if q_num in REVERSE_ITEMS:
            processed_scores[q_num] = 8 - val
        else:
            processed_scores[q_num] = val

    # 2. 计算 11 个亚量表得分 (Subscales)
    subscale_results = {}
    for name, q_nums in SUBSCALES_MAPPING.items():
        scores = [processed_scores[n] for n in q_nums if n in processed_scores]
        if scores:
            subscale_results[name] = round(sum(scores) / len(scores), 4)
        else:
            subscale_results[name] = 0.0

    # 3. 计算 3 个维度得分 (Dimensions)
    novelty = (subscale_results["Original"] + 
               subscale_results["Surprising"] + 
               subscale_results["Germinal"]) / 3
    
    resolution = (subscale_results["Valuable"] + 
                  subscale_results["Logical"] + 
                  subscale_results["Useful"]) / 3
    
    elaboration = (subscale_results["Organic"] + 
                   subscale_results["Elegant"] + 
                   subscale_results["Complex"] + 
                   subscale_results["Understandable"] + 
                   subscale_results["Well-Crafted"]) / 5

    # 4. 计算总分 (11个亚量表的算术平均)
    total_score = sum(subscale_results.values()) / 11

    return {
        "subscales": subscale_results,
        "dimensions": {
            "Novelty": round(novelty, 4),
            "Resolution": round(resolution, 4),
            "Elaboration_and_Synthesis": round(elaboration, 4)
        },
        "total_creative_score": round(total_score, 4)
    }

def process_file(filepath, target_key):
    """处理单个 JSON 文件，精确匹配目标 key 并注入结果"""
    with open(filepath, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
        except Exception as e:
            print(f"解析失败 {filepath}: {e}")
            return False

    modified = False
    
    # 【核心修改】只进行精确匹配，不再使用 startswith 遍历
    if target_key in data:
        print(f"  - 发现目标评估数据: {target_key}")
        eval_data = data[target_key]
        
        # 遍历该评估下的每个 Agent (Agent 1, Agent 2...)
        for agent_label, agent_info in eval_data.items():
            if "scores" in agent_info:
                # 计算得分
                metrics = calculate_metrics(agent_info["scores"])
                # 在对应的 agent 下记录结果
                agent_info["calculated_metrics"] = metrics
                modified = True

    if modified:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    return False

def main():
    # 增加命令行参数支持，方便批量跑不同的人
    parser = argparse.ArgumentParser(description="Calculate CPSS scores for a specific human evaluator.")
    parser.add_argument("--key", type=str, default=TARGET_EVAL_KEY, 
                        help="精确匹配的 human eval key (例如: human_eval_per_agent_Yuhan)")
    parser.add_argument("--dir", type=str, default=LOG_DIR, 
                        help="包含日志文件的根目录")
    args = parser.parse_args()

    log_path = Path(args.dir)
    target_key = args.key

    if not log_path.exists():
        print(f"错误: 路径 {args.dir} 不存在")
        return

    print(f"开始扫描目录: {log_path}")
    print(f"当前目标匹配 Key: '{target_key}'")
    
    count = 0
    for json_file in log_path.rglob("*.json"):
        # print(f"检查文件: {json_file.name}...")  # 如果觉得输出太吵可以注释掉这行
        if process_file(json_file, target_key):
            count += 1

    print(f"\n处理完成！共在 {count} 个文件中计算并更新了 '{target_key}' 的评分。")

if __name__ == "__main__":
    main()