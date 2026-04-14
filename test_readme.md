# Brainstorm 头脑风暴系统 — 测试指南

本系统支持三种使用场景：**纯 LLM Agent 批量测试**、**单人类参与（单机）**、**双人类联机（局域网）**。

---

## 一、环境准备

```bash
# 安装依赖
pip install -r requirements.txt
```

确保 LLM API 服务可用（默认配置为 `http://172.18.39.164:8002/v1`，模型 `Qwen3-8B`）。如需更换，请修改对应文件中的 `API_CONFIG_POOL` 或 `run_batch.sh` 中的环境变量。

---

## 二、纯 LLM Agent 批量测试（无人类参与）

所有 Agent 均由 LLM 扮演，自动运行到讨论结束。适合批量跑数据、对比不同讨论模式。

### 方式 A：使用批量脚本（推荐）

```bash
bash run_batch.sh
```

该脚本会自动遍历 `brainwrite`、`round_robin`、`random`、`leader_worker` 四种模式，每种跑 4 轮，日志保存到 `log/` 目录。

可通过环境变量覆盖默认配置：

```bash
API_KEY="your-key" BASE_URL="https://api.example.com/v1" MODEL="gpt-4" bash run_batch.sh
```

### 方式 B：手动运行单次测试

```bash
python main_batch.py \
  --mode brainwrite \
  --rounds 4 \
  --topic "人工智能技术能怎样帮助解决三体问题？" \
  --agents '[
    {"name":"AI专家","role":"人工智能研究员","api_key":"EMPTY","base_url":"http://172.18.39.164:8002/v1","model":"Qwen3-8B","temperature":0.7},
    {"name":"数学家","role":"数学教授","api_key":"EMPTY","base_url":"http://172.18.39.164:8002/v1","model":"Qwen3-8B","temperature":0.8},
    {"name":"生物学家","role":"生物学家","api_key":"EMPTY","base_url":"http://172.18.39.164:8002/v1","model":"Qwen3-8B","temperature":0.7},
    {"name":"管理学家","role":"管理顾问","api_key":"EMPTY","base_url":"http://172.18.39.164:8002/v1","model":"Qwen3-8B","temperature":0.8}
  ]'
```

Leader-Worker 模式需额外指定 `--leader_ids "[1]"`。

### 日志位置

- 目录：`log/`
- 命名格式：`{模式}_{agent数}_{human数}_{时间戳}.json`
- 示例：`brainwrite_4_0_202604141620.json`（0 表示无人类参与）

---

## 三、单人类参与测试（单机 Streamlit）

一个真人 + 多个 LLM Agent 在同一台机器上讨论。

### 启动

```bash
streamlit run app.py
```

浏览器访问 `http://localhost:8501`。

### 操作步骤

1. **左侧边栏配置**：
   - 选择讨论形式（脑力书写 / 轮流发言 / 随机发言 / 领导-组员）
   - 选择或输入话题
   - 设置 Agent 总数（3-5）和讨论轮数
   - 选择哪个 Agent 由人类操控（下拉菜单，只能选 1 个）
   - 为每个 Agent 设置名称和角色背景
2. **点击「开始讨论」**：系统初始化环境，LLM Agent 自动发言到人类回合
3. **人类发言**：轮到你时，页面显示你的专属视角和输入框，输入观点后点击提交
4. **循环**：提交后 LLM 继续自动推进，直到再次轮到你或讨论结束
5. **最终排名**：讨论结束后，对所有非人类 Agent 进行排名（1 = 最佳）
6. **保存**：排名提交后自动保存日志

### 日志位置

- 目录：`log_human/`
- 示例：`brainwrite_4_1_202604142054.json`（1 表示 1 个人类参与）

---

## 四、双人类联机测试（局域网 Multiplayer）

两个（或更多）真人通过局域网在不同设备上参与同一场讨论。

### 启动

```bash
streamlit run app_multiplayer.py --server.address 0.0.0.0
```

### 测试流程

#### 第一步：玩家 A 创建房间

1. 玩家 A 在本机浏览器打开 `http://localhost:8501`
2. 在「创建房间」标签页中：
   - 选择讨论模式和话题
   - 设置 Agent 总数和轮数
   - **在「人类参与者」中选择至少 2 个 Agent**（例如 Agent 1 和 Agent 3）
   - 配置每个 Agent 的名称和角色
   - 点击「创建房间」
3. 系统生成 4 位数房间号（例如 `8821`）
4. 选择自己要扮演的 Agent（例如认领 Agent 1）
5. 进入等待页面，把房间号告诉队友

#### 第二步：玩家 B 加入房间

1. 玩家 B 在自己的设备上访问 `http://<玩家A的内网IP>:8501`
   - 例如 `http://192.168.1.100:8501`
   - 手机也可以，只要连同一个局域网
2. 在「加入房间」标签页输入房间号 `8821`，点击「加入」
3. 选择并认领剩余的人类座位（例如 Agent 3）
4. 双方到齐，讨论自动开始

#### 第三步：讨论进行

- **轮到自己**：页面显示你的专属视角 + 输入框，输入后提交
- **轮到队友或 AI**：页面显示等待提示，每 3 秒自动刷新检测状态变化
- 提交发言后，系统自动驱动后续 LLM Agent 发言，直到遇到下一个人类
- 如此往复直到所有轮次结束

#### 第四步：排名

- 讨论结束后，**每个人类玩家各自**对所有非自己的 Agent 排名（包括队友和 AI）
- 排名互不可见，独立提交
- 等待所有人提交完成后进入结束页面

#### 第五步：完成

- 日志自动保存，页面显示保存路径
- 点击「开始新讨论」可重新回到大厅

### 日志位置

- 目录：`log_human_2/`
- 示例：`brainwrite_4_2_202604142214.json`（2 表示 2 个人类参与）
- `round_rankings` 格式示例：

```json
{
  "round_rankings": {
    "1": [
      {"agent_id": 2, "agent_name": "专家2", "rank": 1},
      {"agent_id": 3, "agent_name": "专家3", "rank": 2},
      {"agent_id": 4, "agent_name": "专家4", "rank": 3}
    ],
    "3": [
      {"agent_id": 1, "agent_name": "专家1", "rank": 2},
      {"agent_id": 2, "agent_name": "专家2", "rank": 1},
      {"agent_id": 4, "agent_name": "专家4", "rank": 3}
    ]
  }
}
```

键为提交排名的人类 Agent ID，值为该玩家对其他所有 Agent 的排序。

---

## 五、三种场景对比速查

| 项目 | 纯 LLM 批量 | 单人类 (单机) | 双人类 (联机) |
|------|-------------|--------------|--------------|
| 入口文件 | `main_batch.py` / `run_batch.sh` | `app.py` | `app_multiplayer.py` |
| 人类数量 | 0 | 1 | 2+ |
| 日志目录 | `log/` | `log_human/` | `log_human_2/` |
| 排名机制 | 无 | 1 人对所有 LLM 排名 | 每人对所有非自己 Agent 排名 |
| 网络要求 | 无 | 无 | 同一局域网 |
| 启动命令 | `bash run_batch.sh` | `streamlit run app.py` | `streamlit run app_multiplayer.py --server.address 0.0.0.0` |

---

## 六、常见问题

**Q: 联机版队友无法访问？**
确保两台设备在同一局域网，且防火墙允许 8501 端口。可用 `--server.port 端口号` 指定其他端口。

**Q: 页面不自动刷新？**
联机版依赖 `streamlit-autorefresh`，请确认已安装：`pip install streamlit-autorefresh`。

**Q: 两人同时提交会冲突吗？**
不会。系统使用 `threading.Lock` 确保同一时刻只有一个客户端驱动 LLM 推理，另一个客户端只读取状态。

**Q: 能否超过 2 个人类？**
可以。创建房间时在「人类参与者」中选择 3 个或更多 Agent 即可，每个真人认领一个座位。
