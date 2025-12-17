import os
import asyncio

from click import command
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import create_react_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.messages import SystemMessage, HumanMessage
from langchain.chat_models import init_chat_model
from typing import Dict, List, Any
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            model="qwen-max",
            temperature=0.5,
            timeout=30,  # 添加超时配置（秒）
            max_retries=2  # 添加重试次数
        )
# 获取高德地图 API Key
AMAP_MAPS_API_KEY=os.getenv('AMAP_MAPS_API_KEY')

# 使用 langgraph 推荐方式定义大模型
# 企业都会使用本地私有化部署模型（数据安全）
# llm = init_chat_model(
#     model="deepseek-chat",
#     temperature=0,
#     model_provider="deepseek",
# )


# 解析消息列表
def parse_messages(messages: List[Any]) -> None:
    """
    解析消息列表，打印 HumanMessage、AIMessage 和 ToolMessage 的详细信息

    Args:
        messages: 包含消息的列表，每个消息是一个对象
    """
    print("=== 消息解析结果 ===")
    for idx, msg in enumerate(messages, 1):
        print(f"\n消息 {idx}:")
        # 获取消息类型
        msg_type = msg.__class__.__name__
        print(f"类型: {msg_type}")
        # 提取消息内容
        content = getattr(msg, 'content', '')
        print(f"内容: {content if content else '<空>'}")
        # 处理附加信息
        additional_kwargs = getattr(msg, 'additional_kwargs', {})
        if additional_kwargs:
            print("附加信息:")
            for key, value in additional_kwargs.items():
                if key == 'tool_calls' and value:
                    print("  工具调用:")
                    for tool_call in value:
                        print(f"    - ID: {tool_call['id']}")
                        print(f"      函数: {tool_call['function']['name']}")
                        print(f"      参数: {tool_call['function']['arguments']}")
                else:
                    print(f"  {key}: {value}")
        # 处理 ToolMessage 特有字段
        if msg_type == 'ToolMessage':
            tool_name = getattr(msg, 'name', '')
            tool_call_id = getattr(msg, 'tool_call_id', '')
            print(f"工具名称: {tool_name}")
            print(f"工具调用 ID: {tool_call_id}")
        # 处理 AIMessage 的工具调用和元数据
        if msg_type == 'AIMessage':
            tool_calls = getattr(msg, 'tool_calls', [])
            if tool_calls:
                print("工具调用:")
                for tool_call in tool_calls:
                    print(f"  - 名称: {tool_call['name']}")
                    print(f"    参数: {tool_call['args']}")
                    print(f"    ID: {tool_call['id']}")
            # 提取元数据
            metadata = getattr(msg, 'response_metadata', {})
            if metadata:
                print("元数据:")
                token_usage = metadata.get('token_usage', {})
                print(f"  令牌使用: {token_usage}")
                print(f"  模型名称: {metadata.get('model_name', '未知')}")
                print(f"  完成原因: {metadata.get('finish_reason', '未知')}")
        # 打印消息 ID
        msg_id = getattr(msg, 'id', '未知')
        print(f"消息 ID: {msg_id}")
        print("-" * 50)


# 保存状态图的可视化表示
def save_graph_visualization(graph, filename: str = "graph.png") -> None:
    """保存状态图的可视化表示。

    Args:
        graph: 状态图实例。
        filename: 保存文件路径。
    """
    # 尝试执行以下代码块
    try:
        # 以二进制写模式打开文件
        with open(filename, "wb") as f:
            # 将状态图转换为Mermaid格式的PNG并写入文件
            f.write(graph.get_graph().draw_mermaid_png())
        # 记录保存成功的日志
        print(f"Graph visualization saved as {filename}")
    # 捕获IO错误
    except IOError as e:
        # 记录警告日志
        print(f"Failed to save graph visualization: {e}")


# 定义并运行agent
async def run_agent():
    # 实例化 MCP Server客户端
    client = MultiServerMCPClient({
        # 高德地图 MCP Server
        # "amap-amap-sse": {
        #     "url": "https://mcp.amap.com/sse?key="+AMAP_MAPS_API_KEY,
        #     "transport": "sse",
        # },
        "amap-maps-streamableHTTP": {
            "url": "https://mcp.amap.com/mcp?key=" + AMAP_MAPS_API_KEY,
            "transport": "streamable_http",
        },
        # 自定义 MCP Server
        # "calculator": {
        #     "command": "python",
        #     "args": ["calculatorMCPServer.py"],
        #     "transport": "stdio"
        # },
        "calculator": {
            "url": "http://127.0.0.1:8000/mcp",
            "transport": "streamable_http"
        },
    #     "tavily-remote-mcp": {
    #         "command": "cmd",
    #         "args": ["/c",
    #                  "npx -y mcp-remote https://mcp.tavily.com/mcp/?tavilyApiKey=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"],
    #         "env": {},
    #         "transport": "streamable_http"
    # }
        "tavily-remote-mcp": {
            "url": "https://mcp.tavily.com/mcp/?tavilyApiKey=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",
            "transport": "streamable_http"
        }
    })

    # 从MCP Server中获取可提供使用的全部工具
    tools = await client.get_tools()
    # print(f"tools:{tools}\n")

    # 基于内存存储的short-term
    checkpointer = InMemorySaver()

    # 定义系统消息，指导如何使用工具
    system_message = SystemMessage(content=(
        "你是一个旅行规划助手，你可以使用高德地图工具获取出行和天气等信息，也可以使用tavily工具搜索网页信息帮助进行旅行规划，还可以使用计算器工具保障计算预算的正确性。"
    ))

    # 创建 ReAct风格的 agent
    agent = create_react_agent(
        model=llm,
        tools=tools,
        prompt=system_message,
        checkpointer=checkpointer
    )

    # 将定义的agent的graph进行可视化输出保存至本地
    # save_graph_visualization(agent)

    # 定义short-term需使用的thread_id
    config = {"configurable": {"thread_id": "1"}}

    # 1、非流式处理查询
    # 高德地图接口测试
    # agent_response = await agent.ainvoke({"messages": [HumanMessage(content="这个114.05571,22.52245经纬度对应的地方是哪里")]}, config)
    # agent_response = await agent.ainvoke({"messages": [HumanMessage(content="深圳红树林的经纬度坐标是多少")]}, config)
    # agent_response = await agent.ainvoke({"messages": [HumanMessage(content="112.10.22.229这个IP所在位置")]}, config)
    # agent_response = await agent.ainvoke({"messages": [HumanMessage(content="深圳的天气如何")]}, config)
    # agent_response = await agent.ainvoke({"messages": [HumanMessage(content="我要从深圳市南山区中兴大厦骑行到宝安区宝安体育馆，帮我规划下路径")]}, config)
    # agent_response = await agent.ainvoke({"messages": [HumanMessage(content="我要从深圳市南山区中兴大厦步行到宝安区宝安体育馆，帮我规划下路径")]}, config)
    # agent_response = await agent.ainvoke({"messages": [HumanMessage(content="我要从深圳市南山区中兴大厦驾车到宝安区宝安体育馆，帮我规划下路径")]}, config)
    # agent_response = await agent.ainvoke({"messages": [HumanMessage(content="我要从深圳市南山区中兴大厦坐公共交通到宝安区宝安体育馆，帮我规划下路径")]}, config)
    # agent_response = await agent.ainvoke({"messages": [HumanMessage(content="测量下从深圳市南山区中兴大厦到宝安区宝安体育馆驾车距离是多少")]}, config)
    # agent_response = await agent.ainvoke({"messages": [HumanMessage(content="深圳市南山区中石化的加油站有哪些，需要有POI的ID")]}, config)
    # agent_response = await agent.ainvoke({"messages": [HumanMessage(content="POI为B020016GPH的详细信息")]}, config)
    # agent_response = await agent.ainvoke({"messages": [HumanMessage(content="深圳市南山区周围10公里的中石化的加油站")]}, config)
    # 将返回的messages进行格式化输出
    # parse_messages(agent_response['messages'])
    # agent_response_content = agent_response["messages"][-1].content
    # print(f"agent_response:{agent_response_content}")


    # 2、流式处理查询
    # async for message_chunk, metadata in agent.astream(
    #         input={"messages": [HumanMessage(content="3乘以4等于多少")]},
    #         config=config,
    #         stream_mode="messages"
    # ):
    #     # 测试原始输出
    #     # print(f"Token:{message_chunk}\n")
    #     # print(f"Metadata:{metadata}\n\n")
    #
    #     # 跳过工具输出
    #     if metadata["langgraph_node"]=="tools":
    #         continue
    #
    #     # 输出最终结果
    #     if message_chunk.content:
    #         print(message_chunk.content, end="", flush=True)
    while True:
        user_input = input("\n🧠 请输入你的问题（输入 exit 退出）：").strip()
        if user_input.lower() in ["exit", "quit", "退出"]:
            print("👋 已退出。")
            break

        # 启动异步流式调用
        async for message_chunk, metadata in agent.astream(
                input={"messages": [HumanMessage(content=user_input)]},
                config=config,
                stream_mode="messages"
        ):
            # 跳过工具输出
            if metadata["langgraph_node"] == "tools":
                continue

            # 输出最终结果（不换行）
            if message_chunk.content:
                print(message_chunk.content, end="", flush=True)

        print()  # 每轮对话结束换行


if __name__ == "__main__":
    asyncio.run(run_agent())




