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

### 四种实验 / 评估形态

| 实验形态 | 入口 | 目的 | 人类数量 |
|----------|------|------|----------|
| **纯 LLM 实验** | `run_batch.sh` / `main_batch.py` | 测试不同 LLM 在头脑风暴中的表现 | 0 |
| **Single Human 实验** | `app.py` | 1 个人类与多个不同 LLM 讨论，收集人类对各 LLM 的偏好排名 | 1 |
| **Multiplayer 实验** | `app_multiplayer.py` | 多个人类与多个 LLM 共同讨论，直观衡量 Human 与 LLM 的能力差距 | 2–4 |
| **第三方标注评估** | `app_eval_ui.py` | 对已有实验日志进行脱敏展示和人工排序标注（离线评估） | — |

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
├── config/
│   └── llm_config.json         # LLM 模型池配置（API 地址、密钥、推理参数）
├── app.py                      # Single Human 实验前端（1 人类 + N 个 LLM）
├── app_multiplayer.py          # Multiplayer 实验前端（多人类 + N 个 LLM）
├── app_eval_ui.py              # 第三方标注评估系统（脱敏展示 + 人工排序）
├── room_manager.py             # 联机房间全局状态管理器
├── main_batch.py               # 纯 LLM 批量实验入口
├── run_batch.sh                # 纯 LLM 批量实验脚本
├── api_test.py                 # LLM API 连通性检测工具
├── requirements.txt            # Python 依赖
├── log/                        # 纯 LLM 实验日志
├── log_human/                  # Single Human 实验日志
├── log_human_2/                # Multiplayer 实验日志
└── user_log/                   # 第三方标注人员的历史记录
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

- **LLM Agent**：对应 `llm_agents_pool` 字典中的外层键名（如 `"DeepSeek-V3.2"`, `"Qwen3-32B-thinking"`），可据此追溯底层调用的实际模型和参数配置。
- **Human Agent**：固定为 `"human"`。

该字段在 Agent 初始化日志、对话历史（`global_history`）以及最终打分日志中均有记录。

### 模型选取机制

系统根据不同的实验形态采用不同的模型选取策略：

**纯 LLM 实验**（`main_batch.py`）：
- 通过 `--models` 参数**显式指定**每个位置使用的 `config_key`，顺序即 position
- 支持同一模型占据多个位置（如 `"qwen3_8b_local,qwen3_8b_local,qwen3_8b_local,qwen3_8b_local"`）
- Agent 列表**不做 shuffle**，位置顺序即参数顺序

**Single Human / Multiplayer 实验**（`app.py` / `app_multiplayer.py`）：
- 用户仅指定 LLM 数量，系统从 `llm_agents_pool` 中**自动随机盲抽**
- 盲抽规则：数量 ≤ 池大小时**无放回**随机抽取；数量 > 池大小时先全选再**有放回**补齐
- 抽取后所有 Agent（含人类）随机 shuffle，动态分配 `agent_id`
- 人类参与者在实验过程中**无法得知**每个 LLM Agent 背后的具体模型，确保评价的公正性

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

### API 连通性检测

在运行任何实验之前，建议先使用 `api_test.py` 验证 `llm_config.json` 中配置的所有 LLM API 端点是否可用：

```bash
# 测试配置池中的全部模型
python api_test.py

# 指定配置文件路径
python api_test.py --config config/llm_config.json

# 仅测试某个特定模型
python api_test.py --model DeepSeek-V3.2
```

**工作原理**：`api_test.py` 按照与 `config_loader.build_agent_from_config` 完全一致的方式构建 `api_config` / `inference_config`，向每个端点发送一条简短的探测消息（`"只回复一个字：好"`），并报告连通状态：

```
[OK  ] qwen2.5_14B: 好
[OK  ] DeepSeek-V3.2: 好
[FAIL] GLM-4.7: ConnectionError: ...
```

- `OK` 表示 API 端点可达且返回了有效响应
- `FAIL` 表示连接失败或返回错误，需检查 API 地址、密钥或模型服务是否在线

> **注意**：该工具不使用配置中的 `enable_identity` / `identity_prompt` 字段，仅测试 API 层面的连通性。

### 实验一：纯 LLM 实验

**目的**：测试不同 LLM 在头脑风暴任务中的表现，全自动运行，无需人类参与。

```bash
bash run_batch.sh
```

**运行机制**：

1. **配置**：在 `run_batch.sh` 顶部的配置区域设定参数：
   - `CONFIG`：指定 `llm_config.json` 路径
   - `MODELS`：逗号分隔的 `config_key` 列表，顺序即各 Agent 的 position（如 `"qwen3_8b_local,qwen3_8b_local,qwen3_8b_local,qwen3_8b_local"` 表示 4 个位置都使用同一模型）
   - `TOPIC`：讨论主题

2. **执行**：脚本遍历 `brainwrite`、`round_robin`、`random`、`leader_worker` 四种模式，对每种模式调用 `main_batch.py`

3. **模型选取**：通过 `--models` 参数**显式指定**每个位置使用哪个 `config_key`。`main_batch.py` 从 `llm_config.json` 中读取对应的 API 地址、密钥和推理参数，构建 `AgentLLM` 实例。Agent 列表不做 shuffle，位置顺序即参数顺序

4. **讨论推进**：`env.step()` 循环执行直到 `EnvState.FINISHED`，全程无交互

5. **结果保存**：日志保存到 `log/` 目录，文件命名格式 `{模式}_{Agent数}_{人类数}_{时间戳}.json`

也可以直接调用 `main_batch.py` 进行单次实验：

```bash
python main_batch.py \
  --config config/llm_config.json \
  --models "DeepSeek-V3.2,Qwen3-32B-thinking,GLM-4.7,Kimi-k2.5" \
  --mode brainwrite \
  --rounds 4 \
  --topic "人工智能技术能怎样帮助解决三体问题？"
```

### 实验二：Single Human 实验

**目的**：让 1 个人类与多个不同的 LLM Agent 进行头脑风暴讨论，讨论结束后收集人类对各 LLM 的**偏好排名**（即人类更喜欢哪个 LLM 的表现），用于评估不同 LLM 在人类眼中的讨论能力。

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

1. **模型选取与初始化**：
   - 系统从 `llm_config.json` 的 `llm_agents_pool` 中随机盲抽指定数量的 LLM Agent
   - 盲抽规则：数量 ≤ 池大小时**无放回**随机抽取（保证模型多样性）；数量 > 池大小时先全选再**有放回**补齐
   - 抽取的 LLM Agent 与 1 个人类 Agent 一起随机 shuffle 打乱顺序
   - 按打乱后的顺序动态分配 Agent 编号（Agent 1, Agent 2, ...），人类**无法得知**每个 Agent 背后的具体模型

2. **讨论推进**：
   - `auto_advance_llm()` 连续推进 LLM 发言，遇到人类回合时暂停，在 UI 上展示当前讨论上下文
   - 人类在文本框中输入观点并提交后，系统继续推进 LLM 发言
   - 重复上述过程直到所有轮次结束

3. **排名阶段**：
   - 讨论结束后，人类对所有其他 Agent（均为 LLM）进行排名评价（1 = 最佳）
   - 排名采用联动 `selectbox`，修改一个 Agent 的名次时，冲突的 Agent 自动交换到原名次
   - 提交时执行**严格全序校验**——排名值必须恰好构成 `{1, 2, ..., N}` 且无重复；校验失败时保持 UI 状态不变，弹出错误提示，不写入数据

4. **结果保存**：
   - 排名通过后，完整日志保存到 `log_human/` 目录
   - 日志中 `final_rankings` 为排名数组，每项包含 `position`（Agent 编号）、`config_key`（模型标识）和 `rank`（排名）
   - 通过 `config_key` 可追溯每个 Agent 对应的实际模型，从而分析人类对不同 LLM 的偏好

### 实验三：Multiplayer 实验

**目的**：让多个人类与多个 LLM Agent 共同参与头脑风暴讨论。讨论结束后每位人类独立对所有其他参与者（包括其他人类和 LLM）排名，用于**直观衡量 Human 与 LLM 之间的能力差距**——人类在不知道对方身份的情况下，是否能分辨出 LLM 和人类，以及各自的排名位置。

```bash
streamlit run app_multiplayer.py --server.address 0.0.0.0
```

**创建与加入**：

1. 房主在浏览器中创建房间：配置讨论模式、话题、人类参与者数量（2–4）和 LLM 数量
2. 创建后获得 4 位数字房间号，分享给同一局域网内的队友
3. 其他玩家在浏览器中输入房间号加入，浏览可用角色并认领座位

**运行机制**：

1. **模型选取与初始化**：
   - 与 Single Human 实验相同，LLM 从配置池中自动盲抽
   - 多个人类 Agent + 抽取的 LLM Agent 一起随机 shuffle 后动态分配编号
   - 任何参与者都**无法得知**其他 Agent 的真实身份（人类还是 LLM）

2. **等待阶段**：
   - 所有人类玩家通过房间号加入并认领座位
   - 全员就位后，系统自动执行 LLM 初始推进（在 `room.llm_lock` 保护下仅执行一次），进入讨论阶段

3. **讨论推进**：
   - 遇到人类回合时，等待对应玩家输入；其他玩家通过 `streamlit-autorefresh` 自动轮询看到状态更新
   - 当前非发言者显示等待界面，避免干扰
   - 人类提交发言后，系统在 `llm_lock` 保护下继续推进 LLM 发言

4. **排名阶段**：
   - 讨论结束后，每位人类玩家**独立**对所有其他 Agent（含其他人类和 LLM）排名
   - 排名采用联动 `selectbox` 防止名次冲突
   - **双重校验**：
     - **前端校验**：提交前在客户端执行严格全序校验，拦截非法数据
     - **后端校验**：`room_manager.submit_ranking()` 在写入房间状态前再次执行服务端校验。校验失败时拒绝写入，该玩家保持"未提交"状态，不影响其他已提交的玩家

5. **结果保存**：
   - 所有人类玩家提交排名后，日志保存到 `log_human_2/` 目录
   - `final_rankings` 为字典形式，键为提交排名的人类 `agent_id`，值为该人类给出的排名数组
   - 通过交叉对比多位人类的排名结果和 `config_key` 标识，可分析人类与 LLM 之间的能力差距

### 第三方标注评估

**目的**：由未参与实验的第三方标注人员，对已有实验日志进行脱敏阅读和质量排序，获取独立的人工评价数据。

```bash
streamlit run app_eval_ui.py
```

**工作流程**：

1. **用户登录**：
   - 启动后进入登录界面，输入标注人员姓名（如 `Senhao`）并确认
   - 系统在 `user_log/` 目录下加载或创建该用户的历史标注记录（`<user_name>_history.json`）

2. **待标注文件过滤**：
   - 系统遍历后台配置的多个日志目录（默认为 `log/`、`log_human/`、`log_human_2/`），收集所有 JSON 文件
   - 自动剔除当前用户已标注过的文件，在侧边栏下拉列表中仅展示未标注文件
   - 可在 `app_eval_ui.py` 顶部的 `TARGET_LOG_DIRS` 列表中配置需要扫描的日志目录

3. **脱敏展示（Anonymization）**：
   - 选中待标注文件后，系统读取 JSON 内容并提取参与者列表
   - 动态生成随机映射表：将原始 Agent 编号（如 Agent 1–4）随机重映射为新的编号，确保标注人员无法通过位置推断模型身份
   - 使用确定性种子（文件路径 + 用户名），保证同一用户对同一文件的映射始终一致
   - 展示讨论记录时，所有发言者名称和消息正文中出现的 Agent 引用均按映射表替换

4. **排序标注**：
   - 讨论记录下方提供排序表单，为每个 Agent 分配名次（1 = 最佳）
   - 提交时执行**严格全序校验**——排名值必须恰好构成 `{1, 2, ..., N}` 且无重复

5. **结果持久化**：
   - 标注结果写入原 JSON 文件的 `3port` 字段（结构为 `{用户名: [{position, config_key, rank}, ...]}`）
   - 写入时使用文件锁（`filelock`）防止多人同时标注时数据覆盖
   - 同时将已标注文件路径追加到用户的历史记录中
   - 提交后自动刷新页面，已标注文件从待标注列表中消失

**`3port` 字段格式**：

```json
{
  "3port": {
    "Senhao": [
      {"position": 4, "config_key": "Qwen3-32B-nothinking", "rank": 1},
      {"position": 1, "config_key": "qwen2.5_14B", "rank": 2},
      {"position": 3, "config_key": "Qwen3-32B-thinking", "rank": 3},
      {"position": 2, "config_key": "human", "rank": 4}
    ],
    "Alice": [
      {"position": 3, "config_key": "Qwen3-32B-thinking", "rank": 1},
      {"position": 4, "config_key": "Qwen3-32B-nothinking", "rank": 2},
      {"position": 1, "config_key": "qwen2.5_14B", "rank": 3},
      {"position": 2, "config_key": "human", "rank": 4}
    ]
  }
}
```

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
        "config_key": "DeepSeek-V3.2",
        "type": "llm",
        "role_background": "创新设计师...",
        "model": "deepseek-chat",
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
      {"position": 1, "config_key": "DeepSeek-V3.2", "type": "llm", "model": "deepseek-chat"},
      {"position": 2, "config_key": "human", "type": "human", "model": "human"},
      {"position": 3, "config_key": "Qwen3-32B-thinking", "type": "llm", "model": "Qwen3-32B"},
      {"position": 4, "config_key": "GLM-4.7", "type": "llm", "model": "glm-4.7"}
    ]
  },
  "global_history": [
    {
      "round": 1,
      "agent_id": 1,
      "agent_name": "Agent 1",
      "config_key": "DeepSeek-V3.2",
      "content": "..."
    }
  ],
  "final_messages": {
    "1": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
  },
  "final_rankings": [
    {"position": 4, "config_key": "GLM-4.7", "rank": 1},
    {"position": 1, "config_key": "DeepSeek-V3.2", "rank": 2},
    {"position": 3, "config_key": "Qwen3-32B-thinking", "rank": 3}
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

- 纯 LLM 实验：无 `final_rankings` 字段（无人类参与排名）
- Single Human 实验（`app.py`）：`final_rankings` 为排名数组
- Multiplayer 实验（`app_multiplayer.py`）：`final_rankings` 为字典，键为提交排名的人类 `agent_id`，值为对应的排名数组

**日志目录与文件命名**：

| 实验形态 | 日志目录 | 文件命名 |
|----------|----------|----------|
| 纯 LLM | `log/` | `{模式}_{Agent数}_0_{时间戳}.json` |
| Single Human | `log_human/` | `{模式}_{Agent数}_1_{时间戳}.json` |
| Multiplayer | `log_human_2/` | `{模式}_{Agent数}_{人类数}_{时间戳}.json` |

## 配置文件

`config/llm_config.json` 定义可用的 LLM 模型池：

```json
{
  "llm_agents_pool": {
    "DeepSeek-V3.2": {
      "api_url": "https://api.deepseek.com/v1",
      "api_key": "your-api-key",
      "model_name": "deepseek-chat",
      "temperature": 0.7,
      "top_p": null,
      "max_tokens": null,
      "is_reasoning": false,
      "enable_identity": false,
      "identity_prompt": ""
    },
    "Qwen3-32B-thinking": {
      "api_url": "http://your-server:8815/v1",
      "api_key": "EMPTY",
      "model_name": "Qwen3-32B",
      "temperature": 0.7,
      "is_reasoning": true,
      "enable_identity": false,
      "identity_prompt": ""
    }
  }
}
```

### 配置字段说明

| 字段 | 说明 |
|------|------|
| 外层键名 | 即 `config_key`（如 `"DeepSeek-V3.2"`），用于在日志和排名中追溯模型 |
| `api_url` | OpenAI 兼容 API 的 base URL |
| `api_key` | API 密钥（本地部署可设为 `"EMPTY"`） |
| `model_name` | 实际调用的模型标识 |
| `temperature` | 采样温度 |
| `top_p` / `max_tokens` | 可选推理参数，设为 `null` 则使用模型默认值 |
| `is_reasoning` | 是否为混合推理模型开启推理（如 Qwen3-32B），影响 API 调用方式 |
| `enable_identity` | 是否启用自定义身份 Prompt |
| `identity_prompt` | 自定义身份描述（仅 `enable_identity: true` 时生效） |

## Evaluation Metrics (发言质量与相关性评测指标)

为了评估多智能体头脑风暴(Brainstorm)任务中 LLM 发言的多样性、信息量以及切题程度，我们引入了以下三个核心指标：

* **Distinct-n (词汇多样性)**: 衡量局部词汇的丰富度。计算公式为单次发言中 unique n-grams 的数量与总 n-grams 数量的比值。值越高，说明发言越少出现"车轱辘话"重复现象。（默认计算 n=1, 2）。
* **Entropy-n (信息熵)**: 基于香农熵评估发言的均匀分布程度与信息量。如果模型反复使用固定短语，概率分布集中，会导致该指标急剧下降。
* **Sentence-BERT Similarity (主题相关性/防跑题)**: 衡量单次发言在深层语义上是否紧扣初始的主题 (Topic)。利用 Sentence-BERT (如 `all-MiniLM-L6-v2`) 提取发言与 Topic 的句向量，并计算两者的余弦相似度。得分越接近 1，说明越切题。
* **Max BLEU (与已有发言的最高相似度/重复度)**: 衡量 LLM 当前发言是否在"抄袭"或过度模仿上下文中已有的内容。计算方式为：将当前发言视作候选文本（Hypothesis），将该 Agent 可观测到的所有历史发言逐一视作参考文本（Reference），计算句子级 BLEU 相似度并取最大值。这是一个**惩罚性指标**——值越高（接近 1），说明模型发言与历史记录高度雷同（"鹦鹉学舌"）；值越低，说明模型能够提出新颖观点。

## Metric Script Usage (指标计算脚本使用方式)

在完成 Brainstorm 对话数据收集后，您可以运行后处理脚本自动为日志计算上述指标。该脚本假设发言内容为**英文**。

### 环境依赖

```bash
pip install nltk sentence-transformers
```

### 运行脚本

在代码中指定日志文件夹地址，或通过命令行运行：

```bash
python calculate_metrics.py --dir /path/to/your/json/logs/
```

*(注：如果不使用命令行参数，可直接在 `calculate_metrics.py` 中修改 `FOLDER_ADDRESS` 变量。如果要拓展计算 Distinct-3，可以在脚本顶部的 `N_GRAM_LIST` 变量中添加 `3`。)*

### 可用参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--dir` | 日志文件夹路径 | 脚本内 `FOLDER_ADDRESS`（`log_human`） |
| `--output` | 输出模式：`overwrite` 覆盖原文件，`copy` 输出到新文件夹 | `copy` |
| `--model` | Sentence-BERT 模型名称 | `all-MiniLM-L6-v2` |

### 结果格式

脚本运行后，会在 JSON 文件中追加：

1. `global_history` 下的每一条对话记录都会新增一个 `metric` 字段，记录单次发言得分。
2. JSON 顶层会新增 `agent_metrics` 字段，记录该局游戏中各个 Agent (`config_key` / `position`) 在所有轮次中的各项指标**平均值**。

## 依赖

- Python 3.9+
- `openai` >= 1.0.0 — LLM API 调用
- `streamlit` >= 1.30.0 — Web 前端框架
- `streamlit-autorefresh` >= 1.0.0 — 联机版客户端自动刷新
- `filelock` >= 3.12.0 — 第三方标注系统的文件锁（防止并发写入冲突）
- `nltk` >= 3.8.0 — 英文分词工具（指标计算脚本）
- `sentence-transformers` >= 2.2.0 — Sentence-BERT 语义相似度（指标计算脚本）

LLM 后端需提供 OpenAI 兼容的 Chat Completions API。
