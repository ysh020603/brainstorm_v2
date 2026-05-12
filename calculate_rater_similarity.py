import json
import os
from pathlib import Path
import argparse
import pandas as pd

# ======================== 配置项 ========================
# 待处理的日志文件夹路径
LOG_DIR = "/data2/brainstorm/brainstorm_v2/log_cpss_human"

# 参与对比的打分者 keys
# 确保你提取的 key 都已经通过上一个脚本跑出了 "calculated_metrics"
TARGET_EVAL_KEYS = [
    "human_eval_per_agent_Yuhan",
    "cpss_evaluation_per_agent_deepseek-v4-flash", # 在此添加你需要对比的所有 key
    "cpss_evaluation_per_agent_deepseek-v4-flash_few_shot"
]

# 结果保存路径
OUTPUT_CSV = "rater_similarity_matrices.csv"
# =======================================================

def extract_data_from_files(log_path, target_keys):
    """
    遍历日志，提取同一个 Agent 被不同评估者打分的数据。
    返回一个包含多条记录的列表。
    """
    records = []
    
    for json_file in log_path.rglob("*.json"):
        with open(json_file, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
            except Exception:
                continue

        # 收集当前文件中所有被评估过的 Agent 的名字 (如 "Agent 1", "Agent 2")
        agents_in_file = set()
        for key in target_keys:
            if key in data and isinstance(data[key], dict):
                agents_in_file.update(data[key].keys())

        # 针对每个 Agent，收集各个打分者的评分
        for agent_label in agents_in_file:
            record = {"file_id": json_file.name, "agent": agent_label}
            has_valid_data = False
            
            for key in target_keys:
                if key in data and agent_label in data[key]:
                    agent_data = data[key][agent_label]
                    if "calculated_metrics" in agent_data:
                        metrics = agent_data["calculated_metrics"]
                        
                        # 记录各个维度的分数，键名加上打分者前缀以免冲突
                        short_name = key.replace("human_eval_per_agent_", "")
                        record[f"{short_name}_Novelty"] = metrics["dimensions"].get("Novelty", None)
                        record[f"{short_name}_Resolution"] = metrics["dimensions"].get("Resolution", None)
                        record[f"{short_name}_Elaboration"] = metrics["dimensions"].get("Elaboration_and_Synthesis", None)
                        record[f"{short_name}_Total"] = metrics.get("total_creative_score", None)
                        
                        has_valid_data = True
                        
            # 只有当至少提取到一个有效打分时，才加入汇总列表
            if has_valid_data:
                records.append(record)
                
    return records

def generate_and_save_correlation_matrices(df, target_keys, output_csv):
    """
    计算各个维度的皮尔逊相关系数矩阵，打印并输出到同一个 CSV 文件。
    """
    # 提取所有评估者的简写名字
    evaluators = [k.replace("human_eval_per_agent_", "") for k in target_keys]
    
    if len(evaluators) < 2:
        print("⚠️ 至少需要提供 2 个 key 才能计算相似度矩阵。")
        return

    dimensions = ["Novelty", "Resolution", "Elaboration", "Total"]
    
    # 打开 CSV 准备写入
    with open(output_csv, "w", encoding="utf-8") as f:
        f.write("Inter-Rater Similarity (Pearson Correlation)\n")
        f.write(f"Sample Size (N) = {len(df)} pairs\n\n")

    print(f"\n样本总数: {len(df)} 份交叉评估数据")
    print("=" * 50)

    for dim in dimensions:
        # 提取当前维度下的所有打分者的列
        dim_columns = [f"{evaluator}_{dim}" for evaluator in evaluators]
        
        # 截取子 DataFrame 并重命名列名（去掉维度后缀，只保留人名，方便显示）
        df_dim = df[dim_columns].rename(columns=lambda x: x.split('_')[0])
        
        # 计算皮尔逊相关系数矩阵
        corr_matrix = df_dim.corr(method='pearson')
        
        print(f"\n📊 维度: {dim} (相似度矩阵)")
        print("-" * 30)
        print(corr_matrix.round(4))
        
        # 追加写入到 CSV
        with open(output_csv, "a", encoding="utf-8") as f:
            f.write(f"--- Dimension: {dim} ---\n")
        corr_matrix.to_csv(output_csv, mode='a')
        
        # 写入空行以便分隔
        with open(output_csv, "a", encoding="utf-8") as f:
            f.write("\n\n")

def main():
    parser = argparse.ArgumentParser(description="Calculate Pearson correlation matrices between human evaluators.")
    parser.add_argument("--keys", type=str, nargs='+', default=TARGET_EVAL_KEYS, 
                        help="参与对比的 human eval keys 列表 (以空格分隔)")
    parser.add_argument("--dir", type=str, default=LOG_DIR, 
                        help="包含日志文件的根目录")
    parser.add_argument("--output", type=str, default=OUTPUT_CSV, 
                        help="输出的 CSV 文件名")
    args = parser.parse_args()

    log_path = Path(args.dir)
    if not log_path.exists():
        print(f"错误: 路径 {args.dir} 不存在")
        return

    print(f"开始扫描目录: {log_path}")
    print(f"参与比对的打分者: {args.keys}")
    
    # 1. 提取所有对齐的打分数据
    records = extract_data_from_files(log_path, args.keys)
    
    if not records:
        print("未能提取到有效数据。请确保至少有两个人在相同的文件中对相同的 Agent 进行了评估，并且已经运行过算分脚本。")
        return

    # 2. 转换为 Pandas DataFrame
    df = pd.DataFrame(records)
    
    # 3. 计算并输出相关性矩阵
    generate_and_save_correlation_matrices(df, args.keys, args.output)
    
    print("=" * 50)
    print(f"✅ 所有维度的相似度矩阵已保存至: {args.output}")

if __name__ == "__main__":
    main()