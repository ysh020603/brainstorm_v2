import os
import json
import glob
import numpy as np
import pandas as pd
from collections import defaultdict

def main():
    # 基础目录
    base_dir = "eval/ex1_4LLM"
    
    # 需要统计的五种 brainstorm 形式
    modes = [
        "leader_worker", 
        "brainwrite", 
        "leader_worker_22", 
        "random", 
        "round_robin"
    ]

    # 确保基础目录存在
    if not os.path.exists(base_dir):
        print(f"找不到目录: {base_dir}")
        return

    # 遍历所有的 topic 文件夹
    for topic in os.listdir(base_dir):
        topic_dir = os.path.join(base_dir, topic)
        if not os.path.isdir(topic_dir):
            continue
            
        print(f"正在处理 Topic: {topic}")
        
        # 嵌套字典用于收集数据： metrics_dict[指标名称][模型 config_key][脑暴模式 mode] = [数值1, 数值2, ...]
        metrics_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        
        # 遍历该 topic 下每种 brainstorm 形式
        for mode in modes:
            mode_dir = os.path.join(topic_dir, mode)
            if not os.path.isdir(mode_dir):
                continue
                
            # 获取该模式下的所有 json 文件
            json_files = glob.glob(os.path.join(mode_dir, "*.json"))
            for json_file in json_files:
                with open(json_file, 'r', encoding='utf-8') as f:
                    try:
                        data = json.load(f)
                        agent_metrics = data.get("agent_metrics", [])
                        
                        # 提取每个 agent/模型的指标
                        for agent in agent_metrics:
                            config_key = agent.get("config_key")
                            avg_metrics = agent.get("avg_metrics", {})
                            
                            for metric_name, val in avg_metrics.items():
                                metrics_dict[metric_name][config_key][mode].append(val)
                    except Exception as e:
                        print(f"读取文件时出错 {json_file}: {e}")
                        
        # 针对收集到的每个指标生成一个单独的 CSV
        for metric_name, llm_data in metrics_dict.items():
            # 获取当前指标下所有出现过的 LLM 模型
            all_llms = sorted(list(llm_data.keys()))
            
            df_data = []
            for llm in all_llms:
                row = {"LLM (config_key)": llm}
                for mode in modes:
                    vals = llm_data[llm].get(mode, [])
                    if vals:
                        mean_val = np.mean(vals)
                        # 使用样本方差 (ddof=1)，如果只有一个样本则方差记为 0.0
                        var_val = np.var(vals, ddof=1) if len(vals) > 1 else 0.0
                        # 格式化输出：均值 ± 方差 (保留4位小数)
                        row[mode] = f"{mean_val:.4f} ± {var_val:.4f}"
                    else:
                        row[mode] = "N/A" # 如果该模型在这个模式下没有数据
                df_data.append(row)
                
            # 转换为 DataFrame 并整理列顺序
            df = pd.DataFrame(df_data)
            cols = ["LLM (config_key)"] + [m for m in modes if m in df.columns]
            df = df[cols]
            
            # 输出 CSV 到当前 topic 的文件夹下
            csv_filename = f"{metric_name}_stats.csv"
            csv_path = os.path.join(topic_dir, csv_filename)
            df.to_csv(csv_path, index=False, encoding='utf-8-sig')
            
        print(f"  -> 已在 {topic_dir} 下生成 {len(metrics_dict)} 个指标的 CSV 文件。\n")

if __name__ == "__main__":
    main()