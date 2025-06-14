import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from my_agent.agent import graph
import asyncio

# Load environment variables from .env file
load_dotenv()

# Check if the required environment variables are set
if not os.environ.get("ANTHROPIC_API_KEY"):
    print("Please set the ANTHROPIC_API_KEY in your .env file.")
    exit()

if not os.environ.get("TAVILY_API_KEY"):
    print("Please set the TAVILY_API_KEY in your .env file.")
    exit()

async def main():
    """
    A simple command-line interface to interact with the agent.
    """
    print("Welcome to the LangGraph agent! Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break

        inputs = {"messages": [HumanMessage(content=user_input)]}
        
        # The config is used to specify which model to use, "anthropic" or "openai"
        config = {"configurable": {"model_name": "anthropic"}}

        async for output in graph.astream(inputs, config=config):
            # stream() yields dictionaries with output from the graph
            for key, value in output.items():
                if key == "agent":
                    # The 'agent' node output contains the assistant's response
                    if value['messages'][-1].content:
                        print(f"Assistant: {value['messages'][-1].content}")
                elif key == "action":
                    # The 'action' node output contains tool usage information
                    print(f"Tool Result: {value['messages'][-1].content}")


if __name__ == "__main__":
    asyncio.run(main()) 