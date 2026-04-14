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

自动遍历四种模式，日志保存到 `log/`。

### 单人类交互

```bash
streamlit run app.py
```

在浏览器中配置讨论参数，选择一个 Agent 由自己操控，与 LLM Agent 协作讨论。

### 局域网多人联机

```bash
streamlit run app_multiplayer.py --server.address 0.0.0.0
```

玩家 A 创建房间并分享 4 位房间号，玩家 B 在同一局域网内通过内网 IP 访问并加入。

## 日志格式

所有日志为 JSON 文件，结构统一：

```json
{
  "metadata": {
    "mode": "brainwrite",
    "topic": "讨论主题",
    "max_rounds": 4,
    "total_agents": 4,
    "human_count": 2,
    "timestamp": "2026-04-14T22:13:05",
    "agents": [...]
  },
  "global_history": [
    {"round": 1, "agent_id": 1, "agent_name": "专家1", "content": "..."}
  ],
  "final_messages": {
    "1": [{"role": "system", "content": "..."}, ...]
  },
  "round_rankings": {
    "1": [{"agent_id": 2, "agent_name": "专家2", "rank": 1}, ...]
  }
}
```

| 字段 | 说明 |
|------|------|
| `metadata` | 运行配置、参与者信息、时间戳 |
| `global_history` | 按时间顺序的全部发言记录 |
| `final_messages` | 每位 Agent 视角的完整对话上下文（含 System Prompt） |
| `round_rankings` | 人类参与者对其他 Agent 的排名评价 |

文件命名：`{模式}_{Agent数}_{人类数}_{时间戳}.json`

## 依赖

- Python 3.9+
- `openai` >= 1.0.0 — LLM API 调用
- `streamlit` >= 1.30.0 — Web 前端框架
- `streamlit-autorefresh` >= 1.0.0 — 联机版客户端自动刷新

LLM 后端需提供 OpenAI 兼容的 Chat Completions API。
