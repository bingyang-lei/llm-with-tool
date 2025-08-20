## 📌 项目介绍

本项目演示了如何使用 **DeepSeek 的 Chat API**（通过 `openai` 库）实现 **LLM 调用外部函数（tools / function calling）** 的完整流程。

通过本例，你可以学习到：

* 如何定义外部函数（如天气查询、时间查询）并注册到模型。
* 模型如何根据用户输入自动决定是否调用函数。
* 如何在函数调用后，将结果返回给模型，生成最终结构化输出（如 JSON）。
* 如何维护对话上下文 `conversation`，使模型保持连续的推理和选择。

这份代码非常适合作为 **LLM function calling 的入门学习案例**。

---

## 🚀 运行方式

1. 安装依赖：

   ```bash
   pip install openai
   ```

2. 设置环境变量（或直接写在代码里）：

   ```bash
   export OPENAI_API_KEY="your-deepseek-api-key"
   ```

3. 运行脚本：

   ```bash
   python example.py
   ```

---

## ⚙️ 核心代码说明

### 1. **定义外部函数**

我们定义了两个函数：

* `get_weather(location, unit="celsius")` → 返回指定城市的天气
* `get_time(city)` → 返回某城市当前的模拟时间

函数需要在 **tools** 中注册，指定 `name`、`description` 和 `parameters`，方便模型理解如何调用。

```python
weather_function = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "获取指定城市的天气",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
            },
            "required": ["location"]
        }
    }
}
```

---

### 2. **对话上下文 (conversation)**

* **system** → 设定模型身份，例如 "你是一个智能问答助手，只提供结构化 JSON 数据"。
* **user** → 用户输入。
* **assistant** → 模型的回复（可能包含函数调用请求）。
* **tool** → 工具执行后的结果，带上 `tool_call_id`。

示例：

```python
conversation = [
    {"role": "system", "content": "你是一个智能问答助手，只提供结构化 JSON 数据"},
    {"role": "user", "content": "北京的天气怎么样？"}
]
```

当模型调用函数后，必须 **补全一条 `role=tool` 的消息**，告诉模型函数执行结果：

```python
conversation.append({
    "role": "tool",
    "tool_call_id": tool_call.id,
    "content": json.dumps(result, ensure_ascii=False)
})
```

否则会报错：

```
Messages with role 'tool' must be a response to a preceding message with 'tool_calls'
```

---

### 3. **调用 API**

使用 DeepSeek 时，仍然可以用 `openai` 的客户端：

```python
from openai import OpenAI

client = OpenAI(
    api_key=os.environ["OPENAI_API_KEY"],
    base_url="https://api.deepseek.com"
)
```

调用聊天接口：

```python
response = client.chat.completions.create(
    model="deepseek-chat",
    messages=conversation,
    tools=[weather_function, time_function],
    tool_choice="auto",  # 让模型自动选择
    response_format={"type": "json_object"}  # 强制输出 JSON
)
```

---

### 4. **多轮调用逻辑**

完整流程：

1. 用户提问，追加到 `conversation`。
2. 模型输出 → 判断是否有 `tool_calls`。

   * ✅ 有 → 解析参数，执行对应函数。
   * ❌ 无 → 模型直接回答。
3. 将函数结果作为 `tool` 角色写入 `conversation`。
4. 再次请求 API，让模型结合结果生成最终答复。

---

## 📖 学到的要点

* `conversation` 是 LLM 记忆和调用函数的关键，缺一不可。
* 工具调用必须配合 `tool_call_id`，否则报错。
* 可以用 `tool_choice="auto"` 让模型自由选择，也可以指定某个函数。
* `response_format={"type": "json_object"}` 可以强制输出 JSON，方便解析。

---

## 🎯 示例输出

```bash
用户输入: 北京的天气怎么样？
函数调用参数: {'location': '北京'}
函数返回结果: {'temperature': 30, 'condition': '晴', 'unit': '°C'}
模型最终输出（JSON）: {"location": "北京", "temperature": 30, "condition": "晴", "unit": "°C"}
```

