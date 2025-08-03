#!/usr/bin/env python3
"""
Ollama A2A Demo - Two agents using local Llama 3.1
"""

import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.models.ollama import OllamaChatCompletionClient


async def main():
    print("🤖 Starting Ollama A2A Demo with Llama 3.1")
    print("=" * 50)

    # Create Ollama client
    model_client = OllamaChatCompletionClient(
        model="llama3.1:8b",
        base_url="http://localhost:11434",
    )

    # Create two agents with different personalities
    alice = AssistantAgent(
        name="Alice",
        system_message="You are Alice, a curious and friendly AI assistant. You love asking questions and learning about new topics. Keep your responses concise but engaging. Always ask follow-up questions.",
        model_client=model_client,
    )

    bob = AssistantAgent(
        name="Bob",
        system_message="You are Bob, a knowledgeable and helpful AI assistant. You enjoy sharing information and explaining concepts clearly. Keep your responses informative but brief. Always provide helpful explanations.",
        model_client=model_client,
    )

    print("✅ Local Llama 3.1 agents created!")
    print("   - Alice: Curious and friendly, loves asking questions")
    print("   - Bob: Knowledgeable and helpful, enjoys explaining")
    print("\n🔄 Starting conversation...")
    print("-" * 40)

    # Create team with max 8 turns (4 exchanges each)
    team = RoundRobinGroupChat([alice, bob], max_turns=8)

    # Start conversation
    initial_message = "Hello! I'm Alice. I'm really curious about quantum computing. Can you explain what makes it so special compared to regular computers?"

    # Run the conversation
    await Console(team.run_stream(task=initial_message))

    print("\n✅ Ollama A2A Demo completed!")
    print("💡 The agents communicated using local Llama 3.1 - no API keys needed!")


if __name__ == "__main__":
    asyncio.run(main())
