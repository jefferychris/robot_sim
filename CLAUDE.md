# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目性质

这是一个**最小可运行、LLM 无关、机器人无关的多 agent 系统骨架**,运行在 rabo 平台的控制器容器里(`python3 -u main.py` 无参启动)。核心思路:用 LLM function calling 驱动具体动作,LLM 循环写一次、所有 agent 共用。详细教程见 `README.md`。

## 三层架构

```
main.py                # 入口: importlib 动态加载 agents/<DEFAULT_AGENT>,调其 run()
core/                  # 通用框架,与具体 agent/机器人/LLM 无关
  base_agent.py        # ★ BaseAgent.run() 工具调用循环 + safe_parse_args JSON 兜底
drivers/               # 多 agent 可复用的硬件/感知封装(默认 camera.py)
agents/                # 各个具体 agent(子包,__init__.py 暴露 run())
  example_agent/       # 最小示例:报时/算术/笔记 (无硬件,模板起点)
  camera_demo/         # 当前默认,订阅 RGB/深度相机并保存 PNG
  arm_hand_demo/       # 纯驱动演示(左臂伸直+左手大拇指),非 LLM agent
```

**核心约束**:`core/` 不准 import 任何具体 agent / 机器人 SDK / ROS(框架要 ROS-agnostic);`drivers/` 只放"未来其它 agent 也想复用"的代码,某一 agent 私有的常量放 `agents/<name>/config.py`。

## 当前默认 agent

`main.py` 的 `DEFAULT_AGENT = "camera_demo"`(订阅头部与左右手末端 RGB/深度相机并保存为 PNG 到 `camera_frames/`)。其它两个 agent:`example_agent`(框架最小示例)、`arm_hand_demo`(纯硬件演示)。

## 开发约定 (硬约定,改代码前先看)

### 文件归属决策树

| 改什么 | 改哪 |
| --- | --- |
| 给 agent 加新工具 | `agents/<agent>/tools.py`(schema + handler + 调度一站式) |
| 调 prompt / 行为 | `agents/<agent>/prompts.py` |
| 改 LLM 循环行为 (轮数/温度/JSON 容错) | `core/base_agent.py` |
| 换 LLM 模型 / API endpoint | `agents/<agent>/config.py` |
| 改硬件 ID / 物理偏移 / 私有常量 | `agents/<agent>/config.py` |
| 换/加传感器封装 | `drivers/<name>.py` |
| 加新 agent | `cp -r agents/example_agent agents/my_agent`,改 4 件:config / prompts / tools / agent,然后把 `main.py` 的 `DEFAULT_AGENT` 指过去(或删掉原 example_agent) |

### 工具 handler 签名(硬约定)

```python
def _my_tool(agent, args: dict) -> str:
    ...
```

- 第一个参数永远是 agent 实例,第二个永远是 dict,返回永远是 **str**(LLLM 当 tool 消息读)
- **捕获异常并 return 错误信息,不要 raise**(让 LLM 看到错误才能重试)
- 私有函数加下划线前缀 `_my_tool`
- 改 `_HANDLERS` 字典加映射,**别忘了**,否则 LLM 调到不存在的 handler

### Agent 子类协议(硬约定)

继承 `BaseAgent`(可同时多继承 `rclpy.Node`),在 `__init__` 末尾必须有:

```python
self.llm = OpenAI(base_url=..., api_key=...)
self.model = "..."
self.system_prompt = SYSTEM_PROMPT
self.tools = TOOLS
self.setup_messages()
```

并实现 `execute_tool(self, name: str, args: dict) -> str`(通常直接 `return tools_module.execute_tool(self, name, args)`)。外部调用 `agent.run(user_text)` → 字符串回复。

### run() 不能用 stdin REPL

平台容器没有交互式 stdin,`agents/<name>/__init__.py` 的 `run()` 必须常驻并接好输入源(常见两种:H5 控制面板 chat 控件 `rabo_dev_kit.RemoteControl` + `rclpy.spin`,或订阅 ROS2 话题再 spin)。`example_agent` 默认是占位实现(只打日志就返回),**别**写成 `input("you> ")` —— 立刻 EOFError。

## LLM 配置

- 默认走 rabo 平台大模型 API(`https://ai.rabo.cc/p/qwen`,qwen3.6-flash/plus,OpenAI 兼容),**必须用支持 function calling 的模型**
- Key 走环境变量 `RABO_LLM_KEY`(在「个人中心 → API Keys」申请**内部使用**类型,平台自动注入;或在场景环境变量覆盖)。**不要把 key 写进 config.py**
- ⚠️ `rabo_dev_kit.Chat` **不支持工具调用**,不能驱动本框架
- 换其它服务(OpenAI/DeepSeek/Kimi/Ollama/vLLM):改 `agents/<agent>/config.py` 的 `LLM_BASE_URL` / `LLM_MODEL` / 环境变量名

## 常用命令

平台自动按 `requirements.txt` 装依赖并跑 `python3 -u main.py`。本地手动调试:

```bash
python main.py                       # 启动 DEFAULT_AGENT(当前是 camera_demo)
python main.py example_agent         # 本地显式指定,平台用不到
python main.py <your_agent>          # 启动其它 agent
```

**没有测试/lint 命令**(模板默认空)。改动后直接运行对应 agent 验证。`requirements.txt` 是依赖清单:加任何 import 都要把包名列进去,否则平台环境装不上。

## 接入机器人本体

`drivers/` 默认空。在 `drivers/<name>.py` 写硬件 SDK 封装,在 `__init__.py` 暴露 `from .<name> import ...`。典型组合:

```python
from rabo_robocap import UR5, RobotiqHandEGripper, LinkerArmA7, LinkerHandO6Left
self.arm = UR5(robot_id="<sdf-model-name>", mode="sim")  # robot_id 填场景 SDF 里该 model 的 @name
```

详见 `README.md` 「教程四」。

## 已知坑

- **tool_call.arguments 非法 JSON**:Qwen/DeepSeek 偶发输出前导零、Python 字面量、算术表达式。`core/base_agent.py` 的 `safe_parse_args` 走 `ast` 兜底自动修复;如果还 400,可以在那里加更激进的策略
- **持久化 messages 失败**:`self.messages` 混了普通 dict(用户/工具消息)和 SDK 对象 `ChatCompletionMessage`(`tool_calls` 元素是 `ChatCompletionMessageToolCall`)。直接 `json.dumps` 会失败 —— 用 `msg.model_dump()` 转 dict 再入历史,反序列化后 `tool_call_id` 必须与对应 assistant 消息严格配对
- **handler 返回 None**:LLM 会困惑,**永远 return 字符串**(哪怕是空操作也要 return `"跳过"`)
- **未知工具**:LLM 调到一个 `_HANDLERS` 里没注册的工具名 → `execute_tool` 返回 `"未知工具: {name}"` 让 LLM 知道;别在 `TOOLS` 里加没在 `_HANDLERS` 注册的 schema

## 典型新 agent 流程

1. `cp -r agents/example_agent agents/my_agent`
2. 改 `agents/my_agent/agent.py` 类名 + 加硬件初始化
3. 改 `agents/my_agent/config.py` 的 `LLM_*` / 硬件常量(LLM key 用 `os.getenv("RABO_LLM_KEY")`)
4. 改 `agents/my_agent/prompts.py` 的 `SYSTEM_PROMPT`
5. 改 `agents/my_agent/tools.py`:按 `TOOLS` schema + `_xxx(agent, args)` handler + `_HANDLERS` 注册 三步走
6. 改 `agents/my_agent/__init__.py` 的 `run()` 接输入源
7. 改 `main.py` 的 `DEFAULT_AGENT = "my_agent"`