import asyncio
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from mcp_use import MCPAgent, MCPClient

load_dotenv()  # Load .env at the very top

async def run_memory_chat():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("❌ GROQ_API_KEY is not set. Check your .env file.")

    config_file = "browser_mcp.json"
    print("initializing chat...")

    client = MCPClient.from_config_file(config_file)

    # ✅ Pass key here
    llm = ChatGroq(model="qwen-qwq-32b", api_key=api_key)

    agent = MCPAgent(
        llm=llm,
        client=client,
        max_steps=15
    )

    print("interactive mcp chat")
    print("type exit or quit to end the chat")
    print("type clear to clear the memory")
    print("===============================")

    try:
        while True:
            user_input = input("\nyou: ")
            if user_input.lower() in ["exit", "quit"]:
                print(" ending conversation :")
                break

            if user_input.lower() == "clear":
                agent.clear_conversation_history()
                print(" conversation history cleared")
                continue 

            print("\nassistant: ", end="", flush=True)

            try:
                response = await agent.run(user_input)
                print(response)
            except Exception as e:
                print(f"error: {e}")

    finally:
        if client: 
            await client.close_all_sessions()

if __name__ == "__main__":
    asyncio.run(run_memory_chat())
