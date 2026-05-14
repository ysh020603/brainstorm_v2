import os
import json
import glob
import subprocess

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def fix_state_and_resume():
    # 路径配置（相对项目根，与 batch_experiment_same_llm.py 一致）
    state_path = os.path.join(PROJECT_ROOT, "log_same_llm", "batch_state.json")
    expected_count = 4

    if not os.path.exists(state_path):
        print(f"❌ 找不到状态文件: {state_path}")
        return

    # 读取当前实验的运行状态
    with open(state_path, "r", encoding="utf-8") as f:
        state = json.load(f)

    runs = state.get("runs", {})
    modified = False

    print("🔍 正在检查并校准实验状态记录...")
    
    for key, data in runs.items():
        model = data.get("model", "")
        log_paths = data.get("log_paths", [])
        
        target_dir = None
        # 根据已有的文件路径推导所在文件夹
        if len(log_paths) > 0:
            target_dir = os.path.dirname(log_paths[0])
        else:
            continue
            
        if target_dir and os.path.exists(target_dir):
            # 获取实际存在的 json 文件数（去重、排序，避免状态里重复路径导致误判已满）
            actual_logs = sorted(
                set(glob.glob(os.path.join(target_dir, "*.json")))
            )
            count = len(actual_logs)
            
            # 如果硬盘上的实际文件数量小于预期
            if count < expected_count:
                old_completed = data.get("completed", 0)
                if old_completed != count:
                    print(f"🔧 修正数据 [{model}] 位于 {os.path.basename(target_dir)}:")
                    print(f"   - 原记录已完成: {old_completed} 个")
                    print(f"   - 实际查找到: {count} 个 => 修正完毕")
                    
                    # 修正进度并更新有效的文件路径
                    data["completed"] = count
                    data["log_paths"] = actual_logs
                    modified = True

    # 如果有修改，则写回 batch_state.json
    if modified:
        with open(state_path, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
        print("\n✅ batch_state.json 状态修正成功！")
        print("🚀 开始重新运行 batch_experiment_same_llm.py 补充剩余的实验...\n")
        
        # 重新拉起原始实验脚本进行断点续传
        subprocess.run(
            ["python", "batch_experiment_same_llm.py"],
            cwd=PROJECT_ROOT,
        )
    else:
        print("\n✅ 检查完毕：状态文件中的记录与实际文件数量完全一致。")
        print("💡 您可以直接执行 python batch_experiment_same_llm.py 来继续未完成的任务。")

if __name__ == "__main__":
    fix_state_and_resume()