import os
import json
from openai import OpenAI

# 设置 DeepSeek API Key（替换成你自己的）
os.environ["OPENAI_API_KEY"] = "your-api-key"

client = OpenAI(
    api_key=os.environ["OPENAI_API_KEY"],
    base_url="https://api.deepseek.com"  # 使用 DeepSeek endpoint
)

# -- 函数 1：天气查询 --
def get_weather(location: str, unit: str = "celsius"):
    fake_weather = {
        "北京": {"temperature": 30, "condition": "晴"},
        "上海": {"temperature": 28, "condition": "多云"},
        "纽约": {"temperature": 85, "condition": "sunny"}  # 华氏单位
    }
    data = fake_weather.get(location, {"temperature": 20, "condition": "未知"})
    if unit == "fahrenheit" and location != "纽约":
        data["temperature"] = round(data["temperature"] * 9/5 + 32, 1)
        data["unit"] = "°F"
    else:
        data["unit"] = "°C" if unit == "celsius" else "°F"
    return data

# -- 函数 2：汇率查询 --
def get_exchange_rate(currency_from: str, currency_to: str):
    fake_rates = {
        ("USD", "CNY"): 7.25,
        ("CNY", "USD"): 0.138,
        ("EUR", "USD"): 1.09,
        ("USD", "EUR"): 0.92,
    }
    rate = fake_rates.get((currency_from, currency_to), None)
    if rate is None:
        return {"error": "不支持该汇率查询"}
    return {"rate": rate, "from": currency_from, "to": currency_to}

# -- 定义工具函数接口 --
weather_function = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "获取指定城市的天气",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["location"],
        },
    },
}

exchange_function = {
    "type": "function",
    "function": {
        "name": "get_exchange_rate",
        "description": "获取两种货币之间的汇率",
        "parameters": {
            "type": "object",
            "properties": {
                "currency_from": {"type": "string"},
                "currency_to": {"type": "string"},
            },
            "required": ["currency_from", "currency_to"],
        },
    },
}

# -- 设置对话上下文模板 --
conversation = [
    {"role": "system", "content": "你是一个智能问答助手，只能用 JSON 格式回答问题"}
]

# -- 用户测试输入列表 --
user_inputs = [
    "北京的天气怎么样？",
    "请给我上海的天气，用华氏度",
    "现在 100 美元能兑换多少人民币？",
    "欧元兑美元的汇率是多少？",
    "介绍一下你自己"  # 这里不会调用函数
]

for query in user_inputs:
    conversation.append({"role": "user", "content": query})

    # 第一步：模型判断是否调用函数
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=conversation,
        tools=[weather_function, exchange_function],  # 支持两个函数
        tool_choice="auto",
        response_format={"type": "json_object"}
    )

    msg = response.choices[0].message
    conversation.append(msg)

    if msg.tool_calls:
        for tool_call in msg.tool_calls:
            args = json.loads(tool_call.function.arguments)

            if tool_call.function.name == "get_weather":
                result = get_weather(**args)
            elif tool_call.function.name == "get_exchange_rate":
                result = get_exchange_rate(**args)
            else:
                result = {"error": "未知函数调用"}

            print(f"\n用户输入: {query}")
            print("函数调用:", tool_call.function.name)
            print("函数参数:", args)
            print("函数返回结果:", result)

            # 第二步：把结果回传给模型
            conversation.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result, ensure_ascii=False)
            })

            final_response = client.chat.completions.create(
                model="deepseek-chat",
                messages=conversation,
                response_format={"type": "json_object"}
            )

            print("模型最终输出（JSON）:", final_response.choices[0].message.content)
            conversation.append({"role": "assistant", "content": final_response.choices[0].message.content})
    else:
        print(f"\n用户输入: {query}")
        print("模型回答（未调用函数）:", msg.content)
        conversation.append({"role": "assistant", "content": msg.content})