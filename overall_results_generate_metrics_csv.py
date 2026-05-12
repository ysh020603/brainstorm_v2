import os
import pandas as pd
import numpy as np

def parse_mean_std(val):
    """从 'Mean ± Std' 格式的字符串中解析出数值"""
    if isinstance(val, str) and '±' in val:
        parts = val.split('±')
        return float(parts[0].strip()), float(parts[1].strip())
    return np.nan, np.nan

def main():
    base_dir = 'eval/ex1_4LLM'
    
    # 自动识别所有包含 _stats.csv 文件的 Topic 文件夹
    topics = []
    for d in os.listdir(base_dir):
        dir_path = os.path.join(base_dir, d)
        if os.path.isdir(dir_path):
            if any(f.endswith('_stats.csv') for f in os.listdir(dir_path)):
                topics.append(d)
                
    print(f"找到以下 {len(topics)} 个 Topic 文件夹: {topics}")

    # 所有需要处理的指标
    metrics = [
        'avg_distinct_1',
        'avg_distinct_2',
        'avg_entropy_1',
        'avg_entropy_2',
        'avg_max_bleu',
        'avg_sbert_sim_to_topic'
    ]

    # 创建输出整体结果的文件夹
    overall_dir = os.path.join(base_dir, 'overall_results')
    os.makedirs(overall_dir, exist_ok=True)

    for metric in metrics:
        all_data = []
        for topic in topics:
            csv_path = os.path.join(base_dir, topic, f"{metric}_stats.csv")
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                df['Topic'] = topic
                all_data.append(df)
        
        if not all_data:
            print(f"没有找到关于 {metric} 的数据，跳过。")
            continue
            
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # 获取所有环境列（剔除模型名和刚才添加的Topic列）
        env_cols = [col for col in combined_df.columns if col not in ['LLM (config_key)', 'Topic']]
        
        # 拆分出数值列进行计算
        df_calc = combined_df.copy()
        for env in env_cols:
            df_calc[f'{env}_mean'] = df_calc[env].apply(lambda x: parse_mean_std(x)[0])
            df_calc[f'{env}_std'] = df_calc[env].apply(lambda x: parse_mean_std(x)[1])
            
        grouped = df_calc.groupby('LLM (config_key)')
        
        results = []
        for llm, group in grouped:
            row = {'LLM (config_key)': llm}
            for env in env_cols:
                means = group[f'{env}_mean'].dropna().values
                stds = group[f'{env}_std'].dropna().values
                
                if len(means) == 0:
                    row[env] = np.nan
                    continue
                    
                # 计算总均值
                overall_mean = np.mean(means)
                
                # 计算合并方差 (组内方差均值 + 组间均值的方差)
                overall_var = np.mean(stds**2) + np.var(means)
                # 总标准差
                overall_std = np.sqrt(overall_var)
                
                # 重新组合成 'Mean ± Std' 的格式
                row[env] = f"{overall_mean:.4f} ± {overall_std:.4f}"
            results.append(row)
            
        # 保存单个 metric 的整体统合数据
        final_df = pd.DataFrame(results)
        out_path = os.path.join(overall_dir, f"{metric}_overall_stats.csv")
        final_df.to_csv(out_path, index=False)
        print(f"已生成整体计算结果: {out_path}")

if __name__ == "__main__":
    main()