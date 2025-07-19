"""
Simple A2A (Agent-to-Agent) Communication Demo
Two agents that communicate with each other using autogen.
"""

import asyncio
import yaml
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_core import CancellationToken
from autogen_core.models import ChatCompletionClient

# Simple model configuration for OpenAI
model_config = {
    "provider": "openai",
    "config": {
        "model": "gpt-3.5-turbo",
        "api_key": "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"
    }
}

async def main():
    print("🤖 Starting Simple A2A (Agent-to-Agent) Communication Demo")
    print("=" * 60)
    
    # Create the model client
    model_client = ChatCompletionClient.load_component(model_config)
    
    # Create two agents
    agent_a = AssistantAgent(
        name="AgentA",
        system_message="You are Agent A. You are helpful and like to ask questions. Always respond briefly and then ask a follow-up question.",
        model_client=model_client,
    )
    
    agent_b = AssistantAgent(
        name="AgentB",
        system_message="You are Agent B. You are knowledgeable and like to provide information. Always respond briefly and then ask a follow-up question.",
        model_client=model_client,
    )
    
    print("✅ Agents created successfully!")
    print("\n🔄 Starting conversation...")
    print("-" * 40)
    
    # Start the conversation
    initial_message = "Hello Agent B! I'm Agent A. How are you today? What's your favorite topic to discuss?"
    
    # Agent A sends message to Agent B
    print(f"Agent A: {initial_message}")
    
    # Get Agent B's response
    async for response in agent_b.on_messages_stream(
        messages=[TextMessage(content=initial_message, source="AgentA")],
        cancellation_token=CancellationToken(),
    ):
        if hasattr(response, 'content'):
            print(f"Agent B: {response.content}")
            break
    
    print("\n✅ Demo completed successfully!")

if __name__ == "__main__":
    asyncio.run(main()) 