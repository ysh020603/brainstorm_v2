# Brainstorm — 多模式头脑风暴模拟系统

一个基于 LLM 的多 Agent 头脑风暴模拟平台，支持多种经典讨论结构，可灵活配置纯 AI 自动讨论、单人类参与、以及局域网多人联机协作。

## 系统概览

本系统将头脑风暴过程建模为**状态机驱动的多轮对话环境**。每个 Agent（LLM 或人类）按照特定讨论模式规定的顺序和可见性规则轮流发言，系统自动管理信息流转、上下文构建和轮次推进。

### 核心特性

- **4 种讨论模式**，覆盖主流头脑风暴方法论
- **人机混合**，支持 LLM Agent 与人类专家在同一场讨论中协作
- **局域网联机**，多人通过不同设备实时参与同一场讨论
- **结构化日志**，完整记录讨论历史、每位 Agent 的上下文视角和最终排名
- **可见性隔离**，不同模式下 Agent 看到的信息严格按规则过滤
- **全自动盲抽**，Human Evaluation 模式下 LLM Agent 从配置池中随机抽取，确保实验公正

## 讨论模式

### BrainWrite（脑力书写）

环形传递机制：每轮每位参与者在一张"纸条"上写下想法，然后纸条按固定方向传递给下一位。每人看到的是前人传来的草稿链，而非全局讨论。

- 第 R 轮，Agent_i 看到的是 Agent_((i-k) mod n) 在第 k 轮（k=1..R-1）的发言
- 适合产生多样化、不受群体思维干扰的创意

### Round Robin（轮流发言）

经典圆桌讨论：所有人按固定顺序依次发言，每人可见之前所有轮的全部发言以及当前轮排在自己前面的发言。

### Random（随机发言）

与 Round Robin 类似，但每轮发言顺序随机打乱，保证每人每轮发言一次。引入顺序随机性，减少位置偏见。

### Leader-Worker（领导-组员）

分层结构：Leader 先发言定调，Worker 根据 Leader 指导给出方案。双向信息隔离——Leader 只看到 Worker 的汇报，Worker 只看到 Leader 的指导，Worker 之间互不可见。

## 项目结构

```
brainstorm_v2/
├── agents/                     # Agent 模块
│   ├── agent_base.py           # AgentBase 基类 + EnvState 状态枚举
│   ├── agent_llm.py            # LLM Agent（调用 OpenAI 兼容 API）
│   └── agent_human.py          # 人类 Agent（等待外部输入）
├── envs/                       # 讨论环境模块
│   ├── env_base.py             # EnvBase 基类（状态机、消息构建、日志）
│   ├── brainwrite.py           # BrainWrite 环形传递环境
│   ├── round_robin.py          # Round Robin 轮流发言环境
│   ├── random_env.py           # Random 随机发言环境
│   └── leader_worker.py        # Leader-Worker 分层环境
├── prompts/                    # Prompt 模板
│   ├── system_prompts.py       # System Prompt 构建器
│   └── topics.py               # 预设话题与专家角色
├── tools/
│   ├── config_loader.py        # LLM 配置加载与 Agent 工厂
│   └── call_openai.py          # OpenAI 兼容 API 调用封装
├── app.py                      # 单机 Streamlit 前端（1 个人类）
├── app_multiplayer.py          # 局域网联机 Streamlit 前端（多人类）
├── room_manager.py             # 联机房间全局状态管理器
├── main_batch.py               # 纯 LLM 批量运行入口
├── run_batch.sh                # 批量测试脚本
├── requirements.txt            # Python 依赖
├── log/                        # 纯 LLM 批量测试日志
├── log_human/                  # 单人类测试日志
└── log_human_2/                # 多人联机测试日志
```

## 架构设计

### Agent 身份标识（动态重排序列）

Agent 的唯一标识（`agent_id`）**不在构造时静态绑定**，而是采用"先选取，后动态分配"的机制：

1. **组装候选池**：根据实验设置，将所有被选中的 LLM Agent 和 Human Agent 放入同一个列表。
2. **全局重排（Shuffle）**：对该列表进行随机打乱。
3. **动态分配序号**：由 `EnvBase` 构造函数根据重排后列表的索引顺序，赋予每个 Agent 一个从 1 开始递增的 `agent_id`。
4. **唯一标识**：该 `agent_id` 既代表 Agent 在 UI 上的展示顺序（Agent 1, Agent 2...），也是实验中该 Agent 的唯一标识，用于后续日志记录和打分。

### config_key 字段

每个 Agent 携带一个 `config_key` 字段，用于标识其配置来源：

- **LLM Agent**：对应 `llm_agents_pool` 字典中的外层键名（如 `"model_a"`, `"qwen3_8b_local"`），可据此追溯底层调用的实际模型和参数配置。
- **Human Agent**：固定为 `"human"`。

该字段在 Agent 初始化日志、对话历史（`global_history`）以及最终打分日志中均有记录。

### 状态机

环境通过 `EnvState` 驱动流转：

```
WAITING_LLM  →  Agent 发言  →  WAITING_LLM（下一个 LLM）
     ↓                              ↓
WAITING_HUMAN（遇到人类）      轮次结束 → 下一轮 / FINISHED
```

核心方法 `env.step()` 每次推进一位 Agent 发言。遇到人类 Agent 时暂停，等待外部调用 `submit_input()` 注入输入后再继续。

### 消息构建管线

```
get_visible_messages()       按模式规则过滤可见历史
        ↓
_build_timeline_groups()     以自身发言为锚点分组
        ↓
format_round_prompt()        渲染为自然语言 User Prompt
        ↓
build_messages_for_agent()   组装完整 OpenAI messages 列表
```

每种讨论模式通过重写可见性和 Prompt 渲染方法实现差异化行为，环境基类处理通用的状态机逻辑和日志保存。

### 排名提交校验（Strict Total Order）

讨论结束后，人类参与者需对其他 Agent 进行排名评价。系统在排名提交时执行**严格全序校验**，确保数据合法性：

**校验规则**：提交的排名值必须恰好构成 `{1, 2, ..., N}`（N 为被评价对象数量），不允许存在重复值（即不允许并列排名）。

**单机模式（`app.py`）**：
- 拦截点位于"提交排名"按钮的回调函数最前端
- 校验未通过时：立即中断提交流程，保持 UI 当前所有选项和已填数据不变，弹出错误提示要求重新排序
- 校验通过时：正常写入数据并触发后续状态流转

**联机模式（`app_multiplayer.py` + `room_manager.py`）**：
- 前端在提交前先执行客户端校验拦截
- 后端 `room_manager.submit_ranking()` 在写入房间状态前再次执行服务端校验
- 校验未通过时：拒绝写入数据，该玩家保持"未提交"状态，向其返回错误信息；同房间内其他已提交的玩家不受影响
- 校验通过时：数据写入 `RoomState`，标记该玩家为"已提交"，随后执行全局状态检查（判断是否所有玩家都已提交）

### 联机架构

联机版通过 Python 模块级全局字典实现跨 Streamlit Session 的状态共享：

- `room_manager.py` 维护 `{房间号: RoomState}` 全局字典
- 每个 RoomState 持有一个共享的 env 实例
- 多个浏览器 Session 通过相同房间号访问同一个 env
- `threading.Lock` 保证并发安全（防止重复触发 LLM 推理）
- `streamlit-autorefresh` 实现客户端定时轮询状态变化

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 纯 LLM 批量测试

```bash
bash run_batch.sh
```

自动遍历四种模式，无需人类参与。所有 Agent 均为 LLM，由 `main_batch.py` 驱动，无交互界面。日志保存到 `log/`。

**运行机制**：
1. `run_batch.sh` 遍历预定义的模式列表，依次调用 `main_batch.py`
2. `main_batch.py` 从 `llm_agents_pool` 中抽取全部 LLM Agent，构建环境并自动运行到讨论结束
3. 全程无需人工干预，适合大规模自动化实验

### 单人类交互（Human Evaluation / Single Human）

```bash
streamlit run app.py
```

在浏览器中配置讨论参数：

1. 选择讨论形式（BrainWrite / Round Robin / Random / Leader-Worker）
2. 选择或输入讨论话题
3. 设定 LLM Agent 数量（系统从配置池中**自动随机盲抽**，无需手动选择模型）
4. 设定讨论轮数
5. 点击"开始讨论"

**运行机制**：
1. **初始化**：系统从 `llm_agents_pool` 中随机抽取指定数量的 LLM Agent，与 1 个人类 Agent 一起随机打乱顺序，动态分配 Agent 编号
2. **讨论推进**：`auto_advance_llm()` 连续推进 LLM 发言，遇到人类回合时暂停等待输入。人类提交发言后继续推进 LLM，直到所有轮次结束
3. **排名阶段**：讨论结束后，人类对所有其他 Agent 进行排名评价。排名采用联动 `selectbox`，自动避免名次冲突
4. **排名校验**：提交时执行**严格全序校验**——排名值必须恰好构成 `{1, 2, ..., N}` 且无重复。校验失败时保持 UI 状态不变，弹出错误提示，不写入数据
5. **日志保存**：排名通过后，完整日志（含讨论历史、每位 Agent 的上下文视角、最终排名）保存到 `log_human/`

### 局域网多人联机（Multiplayer）

```bash
streamlit run app_multiplayer.py --server.address 0.0.0.0
```

**创建与加入**：
1. 房主创建房间：配置讨论模式、话题、人类参与者数量和 LLM 数量（LLM 自动盲抽）
2. 创建后获得 4 位数字房间号，分享给队友
3. 其他玩家通过房间号加入，认领角色座位

**运行机制**：
1. **等待阶段**：所有人类玩家认领座位后，系统自动执行 LLM 初始推进，进入讨论阶段
2. **讨论推进**：与单机模式类似，遇到人类回合时等待对应玩家输入。其他玩家通过 `streamlit-autorefresh` 自动轮询看到状态更新。当前发言者以外的玩家显示等待界面
3. **排名阶段**：讨论结束后，每位人类玩家独立对其他所有 Agent 排名。排名采用联动 `selectbox` 防止名次冲突
4. **双重校验**：
   - **前端校验**：提交前在客户端执行严格全序校验，拦截非法数据
   - **后端校验**：`room_manager.submit_ranking()` 在写入房间状态前再次执行服务端校验。校验失败时拒绝写入，该玩家保持"未提交"状态，不影响其他已提交的玩家
5. **日志保存**：所有人类玩家提交排名后，日志保存到 `log_human_2/`，`final_rankings` 为字典形式（键为人类 `agent_id`）

## 日志格式

所有日志为 JSON 文件，结构统一：

```json
{
  "metadata": {
    "mode": "brainwrite",
    "topic": "讨论主题",
    "max_rounds": 4,
    "total_agents": 4,
    "human_count": 1,
    "timestamp": "2026-04-17T15:07:00",
    "agents": [
      {
        "agent_id": 1,
        "config_key": "model_a",
        "type": "llm",
        "role_background": "创新设计师...",
        "model": "gpt-4o",
        "temperature": 0.7,
        "top_p": null,
        "max_tokens": null
      },
      {
        "agent_id": 2,
        "config_key": "human",
        "type": "human",
        "role_background": "人类专家"
      }
    ],
    "position_map": [
      {"position": 1, "config_key": "model_a", "type": "llm", "model": "gpt-4o"},
      {"position": 2, "config_key": "human", "type": "human", "model": "human"},
      {"position": 3, "config_key": "model_b_reasoning", "type": "llm", "model": "deepseek-r1"},
      {"position": 4, "config_key": "model_a", "type": "llm", "model": "gpt-4o"}
    ]
  },
  "global_history": [
    {
      "round": 1,
      "agent_id": 1,
      "agent_name": "Agent 1",
      "config_key": "model_a",
      "content": "..."
    }
  ],
  "final_messages": {
    "1": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
  },
  "final_rankings": [
    {"position": 4, "config_key": "model_a", "rank": 1},
    {"position": 1, "config_key": "model_a", "rank": 2},
    {"position": 3, "config_key": "model_b_reasoning", "rank": 3}
  ]
}
```

### 字段说明

| 字段 | 说明 |
|------|------|
| `metadata` | 运行配置、参与者信息（含 `config_key`）、时间戳 |
| `metadata.agents` | 每位 Agent 的详细信息，`config_key` 标识配置来源 |
| `metadata.position_map` | 按位置排列的 Agent 映射，包含 `config_key` |
| `global_history` | 按时间顺序的全部发言记录，每条含 `config_key` |
| `final_messages` | 每位 Agent 视角的完整对话上下文（含 System Prompt） |
| `final_rankings` | 人类参与者对其他 Agent 的排名评价（使用 `position` + `config_key` 标识） |

**排名结构说明**：

- 单机模式（`app.py`）：`final_rankings` 为排名数组
- 联机模式（`app_multiplayer.py`）：`final_rankings` 为字典，键为提交排名的人类 `agent_id`，值为对应的排名数组

文件命名：`{模式}_{Agent数}_{人类数}_{时间戳}.json`

## 配置文件

`config/llm_config.json` 定义可用的 LLM 模型池：

```json
{
  "llm_agents_pool": {
    "model_a": {
      "api_url": "https://api.example.com/v1",
      "api_key": "your-api-key",
      "model_name": "gpt-4o",
      "temperature": 0.7,
      "top_p": null,
      "max_tokens": null,
      "is_reasoning": false,
      "enable_identity": false,
      "identity_prompt": ""
    },
    "model_b_reasoning": {
      "api_url": "https://api.example.com/v1",
      "api_key": "your-api-key",
      "model_name": "deepseek-r1",
      "temperature": 0.7,
      "is_reasoning": true,
      "enable_identity": true,
      "identity_prompt": "You are a creative designer..."
    }
  }
}
```

- 外层键名（如 `"model_a"`）即为 `config_key`，用于在日志中追溯模型配置
- `model_name` 为实际调用的底层模型标识
- Human Evaluation 模式下，系统从该池中自动随机抽取所需数量的模型配置

## 依赖

- Python 3.9+
- `openai` >= 1.0.0 — LLM API 调用
- `streamlit` >= 1.30.0 — Web 前端框架
- `streamlit-autorefresh` >= 1.0.0 — 联机版客户端自动刷新

LLM 后端需提供 OpenAI 兼容的 Chat Completions API。
