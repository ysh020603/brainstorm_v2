import json
import os

def extract_few_shots(input_path: str, output_path: str):
    print(f"Reading from {input_path}...")
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 1. 提取讨论主题
    topic = data.get("metadata", {}).get("topic", "")

    # 2. 提取并拼接每个 Agent 的完整发言 (idea_content)
    agent_ideas = {}
    for turn in data.get("global_history", []):
        agent_name = turn.get("agent_name")
        content = turn.get("content", "").strip()
        role = turn.get("role", "")
        
        if not agent_name or not content or role in ["system", "moderator"]:
            continue
            
        if agent_name not in agent_ideas:
            agent_ideas[agent_name] = []
        agent_ideas[agent_name].append(content)
        
    for name in agent_ideas:
        agent_ideas[name] = "\n\n".join(agent_ideas[name])

    # 3. 提取人类评估分数 (这里取 Yuhan 的打分作为 ground truth)
    human_scores = data.get("human_eval_per_agent_Yuhan", {})

    # 我们选取 3 个 Agent 作为 3-shot
    target_agents = ["Agent 1", "Agent 2", "Agent 3"]

    few_shots_dict = {}
    
    # 获取所有的 55 题的 key (以 Agent 1 的 key 为基准)
    if "Agent 1" in human_scores:
        score_keys = human_scores["Agent 1"]["scores"].keys()
        for key in score_keys:
            few_shots_dict[key] = []
            for agent_name in target_agents:
                if agent_name in human_scores and agent_name in agent_ideas:
                    score = human_scores[agent_name]["scores"].get(key)
                    if score is not None:
                        few_shots_dict[key].append({
                            "topic": topic,
                            "idea_content": agent_ideas[agent_name],
                            "score": score
                        })

    # 4. 保存为 JSON
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(few_shots_dict, f, ensure_ascii=False, indent=2)
        
    print(f"Successfully extracted few-shots for {len(few_shots_dict)} questions.")
    print(f"Saved to: {output_path}")

if __name__ == "__main__":
    # 请修改为你的实际文件路径
    input_file = "/data2/brainstorm/brainstorm_v2/log_cpss_human/round_robin_4_0_202604272325.json" 
    output_file = "config/cpss_few_shots.json"
    
    extract_few_shots(input_file, output_file)