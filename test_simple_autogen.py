#!/usr/bin/env python3
"""
Simple test to verify AutoGen v0.4 + Ollama setup works
"""

import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_ext.models.ollama import OllamaChatCompletionClient

async def test_simple():
    print("🧪 Testing simple AutoGen + Ollama setup...")
    
    # Create a simple model client
    model_client = OllamaChatCompletionClient(
        model="llama3.1:8b",
    )
    
    # Create two simple agents
    agent1 = AssistantAgent(
        name="Agent1",
        system_message="You are Agent1. Just say hello and count to 3.",
        model_client=model_client
    )
    
    agent2 = AssistantAgent(
        name="Agent2", 
        system_message="You are Agent2. Respond briefly to what Agent1 says.",
        model_client=model_client
    )
    
    # Create team with short termination
    termination = MaxMessageTermination(4)
    team = RoundRobinGroupChat([agent1, agent2], termination_condition=termination)
    
    print("🚀 Running simple test...")
    result = await team.run(task="Hello, let's test this setup!")
    
    print("✅ Test complete!")
    print(f"Messages: {len(result.messages)}")
    for i, msg in enumerate(result.messages):
        print(f"{i+1}. {msg.source}: {msg.content[:100]}...")

if __name__ == "__main__":
    asyncio.run(test_simple())