#!/usr/bin/env python3
"""
Simple A2A (Agent-to-Agent) Communication Demo
Two agents that communicate with each other using AutoGen v0.4.
"""

import asyncio
import os
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient


async def main():
    print("🤖 Starting Simple A2A (Agent-to-Agent) Communication Demo")
    print("=" * 60)
    
    # Get API key from environment
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ Error: Please set OPENAI_API_KEY environment variable")
        print("   Example: export OPENAI_API_KEY='your-api-key-here'")
        return
    
    # Create the model client
    model_client = OpenAIChatCompletionClient(
        model="gpt-3.5-turbo",
        api_key=api_key
    )
    
    # Create two agents with different personalities
    agent_a = AssistantAgent(
        name="Alice",
        system_message="You are Alice, a curious and friendly AI assistant. You love asking questions and learning about new topics. Keep your responses concise but engaging.",
        model_client=model_client,
    )
    
    agent_b = AssistantAgent(
        name="Bob",
        system_message="You are Bob, a knowledgeable and helpful AI assistant. You enjoy sharing information and explaining concepts clearly. Keep your responses informative but brief.",
        model_client=model_client,
    )
    
    print("✅ Agents created successfully!")
    print("   - Alice: Curious and friendly, loves asking questions")
    print("   - Bob: Knowledgeable and helpful, enjoys explaining")
    print("\n🔄 Starting conversation...")
    print("-" * 40)
    
    # Create a team for the agents to communicate
    team = RoundRobinGroupChat([agent_a, agent_b], max_turns=6)
    
    # Start the conversation
    initial_message = "Hello! I'm Alice. I'm really curious about artificial intelligence. Can you tell me something interesting about AI?"
    
    # Run the conversation
    await Console(team.run_stream(task=initial_message))
    
    print("\n✅ Demo completed successfully!")
    print("💡 The agents communicated using AutoGen's RoundRobinGroupChat!")
    
    # Clean up
    await model_client.close()


if __name__ == "__main__":
    asyncio.run(main())