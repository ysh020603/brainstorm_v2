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

    if not os.path.exists(base_dir):
        print(f"找不到目录: {base_dir}")
        return

    # 新建用于存储 round 统计结果的文件夹
    out_dir = os.path.join(base_dir, "round_metrics_results")
    os.makedirs(out_dir, exist_ok=True)
    print(f"统计结果将保存在: {out_dir}\n")

    # 嵌套字典用于收集数据： 
    # metrics_dict[指标名称][轮次][模型 config_key][脑暴模式 mode] = [数值1, 数值2, ...]
    metrics_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    
    # 遍历所有的 topic 文件夹
    topic_count = 0
    for topic in os.listdir(base_dir):
        topic_dir = os.path.join(base_dir, topic)
        if not os.path.isdir(topic_dir) or topic == "overall_results" or topic == "round_metrics_results":
            continue
            
        topic_count += 1
        
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
                        # 读取新增的 global_history 字段
                        global_history = data.get("global_history", [])
                        
                        for item in global_history:
                            round_num = item.get("round")
                            config_key = item.get("config_key")
                            metrics = item.get("metric", {})
                            
                            # 过滤无效数据
                            if round_num is None or not config_key or not metrics:
                                continue
                                
                            # 记录该回合的各项指标
                            for metric_name, val in metrics.items():
                                if val is None:
                                    continue
                                try:
                                    val = float(val)
                                except (TypeError, ValueError):
                                    continue
                                metrics_dict[metric_name][round_num][config_key][mode].append(val)
                    except Exception as e:
                        print(f"读取文件时出错 {json_file}: {e}")

    print(f"成功扫描了 {topic_count} 个 Topic 文件夹。正在生成 CSV 文件...\n")

    # 针对收集到的每个指标和每个轮次，生成一个单独的 CSV
    file_count = 0
    for metric_name, rounds_data in metrics_dict.items():
        for round_num, llm_data in rounds_data.items():
            # 获取当前 (指标, 轮次) 下所有出现过的 LLM 模型
            all_llms = sorted(list(llm_data.keys()))
            
            df_data = []
            for llm in all_llms:
                row = {"LLM (config_key)": llm}
                for mode in modes:
                    raw_vals = llm_data[llm].get(mode, [])
                    vals = [v for v in raw_vals if v is not None]
                    if vals:
                        mean_val = np.mean(vals)
                        std_val = np.std(vals, ddof=1) if len(vals) > 1 else 0.0
                        row[mode] = f"{mean_val:.4f} ± {std_val:.4f}"
                    else:
                        row[mode] = "N/A"
                df_data.append(row)
                
            # 转换为 DataFrame 并整理列顺序（保证预期的 mode 顺序在前）
            df = pd.DataFrame(df_data)
            cols = ["LLM (config_key)"] + [m for m in modes if m in df.columns]
            df = df[cols]
            
            # 输出 CSV 到指定的集中文件夹
            csv_filename = f"{metric_name}_round_{round_num}_stats.csv"
            csv_path = os.path.join(out_dir, csv_filename)
            df.to_csv(csv_path, index=False, encoding='utf-8-sig')
            file_count += 1
            
    print(f"处理完成！共生成了 {file_count} 个 CSV 文件，请在 '{out_dir}' 目录中查看。")

if __name__ == "__main__":
    main()