from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain import hub
from langchain_classic.agents import create_react_agent
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import asyncio
import os

load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=os.getenv("OPENAI_API_KEY"))

server_params = StdioServerParameters(
        command="npx",
        args=["firecrawl-mcp"],
        env={"FIRECRAWL_API_KEY": os.getenv("FIRECRAWL_API_KEY")},
    )

async def main():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await load_mcp_tools(session=session)
            prompt = hub.pull("hwchase17/react")
            agent = create_react_agent(model, tools, prompt=prompt)
            
            messages = [
                {"role": "system", "content": "You are a helpful assistant that can use tools."},
            ]
            
            print("Available tools:", *[tool.name for tool in tools])
            print("-"*50)
            
            while True:
                user_input = input("Enter your prompt: ")
                if user_input.lower() in ["exit", "quit", "bye"]:
                    print("Exiting...")
                    break
                messages.append({"role": "user", "content": user_input[:175000]})
                
                try:
                    response = await agent.ainvoke(messages)
                    ai_message = response["messages"][-1]["content"]
                    print("AI:", ai_message)
                except Exception as e:
                    print(f"Error: {e}")
                    continue
                
if __name__ == "__main__":
    asyncio.run(main())