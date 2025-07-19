#!/usr/bin/env python3
"""
Core A2A Demo with Custom Agents
Shows how to create custom agents and have them communicate using AutoGen Core.
"""

import asyncio
import os
from typing import List, Sequence

from autogen_core import (
    CancellationToken,
    DefaultTopicId,
    RoutedAgent,
    SingleThreadedAgentRuntime,
    message_handler,
    type_subscription,
)
from autogen_core.models import (
    AssistantMessage,
    ChatCompletionClient,
    SystemMessage,
    UserMessage,
)
from autogen_ext.models.openai import OpenAIChatCompletionClient
from dataclasses import dataclass
import yaml


@dataclass
class ChatMessage:
    """Simple message type for agent communication"""
    content: str
    sender: str


@type_subscription("chat_topic")
class CustomAgent(RoutedAgent):
    """A custom agent that can participate in A2A communication"""
    
    def __init__(self, name: str, personality: str, model_client: ChatCompletionClient):
        super().__init__(f"Custom agent {name}")
        self.name = name
        self.personality = personality
        self.model_client = model_client
        self.conversation_history: List[ChatMessage] = []
    
    @message_handler
    async def handle_chat_message(self, message: ChatMessage, ctx) -> None:
        """Handle incoming chat messages from other agents"""
        if message.sender == self.name:
            return  # Don't respond to our own messages
        
        # Add to history
        self.conversation_history.append(message)
        
        # Prepare context for the LLM
        system_msg = SystemMessage(content=f"You are {self.name}. {self.personality}")
        
        # Convert conversation history to LLM messages
        messages = [system_msg]
        for msg in self.conversation_history[-5:]:  # Keep last 5 messages
            if msg.sender == self.name:
                messages.append(AssistantMessage(content=msg.content, source=self.name))
            else:
                messages.append(UserMessage(content=msg.content, source=msg.sender))
        
        # Get response from LLM
        response = await self.model_client.create(messages)
        response_content = response.content
        
        # Create response message
        response_msg = ChatMessage(content=response_content, sender=self.name)
        self.conversation_history.append(response_msg)
        
        print(f"🗣️  {self.name}: {response_content}")
        
        # Publish response to other agents
        await self.publish_message(response_msg, topic_id=DefaultTopicId("chat_topic"))


async def main():
    print("🤖 Starting Core A2A Demo with Custom Agents")
    print("=" * 50)
    
    # Get API key from environment
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ Error: Please set OPENAI_API_KEY environment variable")
        print("   Example: export OPENAI_API_KEY='your-api-key-here'")
        return
    
    # Create model client
    model_client = OpenAIChatCompletionClient(
        model="gpt-3.5-turbo",
        api_key=api_key
    )
    
    # Create runtime
    runtime = SingleThreadedAgentRuntime()
    
    # Create custom agents
    alice = CustomAgent(
        name="Alice",
        personality="You are curious and love asking questions about science and technology. Keep responses brief but engaging.",
        model_client=model_client
    )
    
    bob = CustomAgent(
        name="Bob", 
        personality="You are knowledgeable and enjoy explaining complex topics in simple terms. Keep responses informative but concise.",
        model_client=model_client
    )
    
    # Register agents with runtime
    await CustomAgent.register(runtime, "Alice", lambda: alice)
    await CustomAgent.register(runtime, "Bob", lambda: bob)
    
    print("✅ Custom agents created and registered!")
    print("   - Alice: Curious, loves asking questions")
    print("   - Bob: Knowledgeable, enjoys explaining")
    print("\n🔄 Starting conversation...")
    print("-" * 40)
    
    # Start the runtime
    runtime.start()
    
    # Send initial message
    initial_message = ChatMessage(
        content="Hi! I'm Alice. I'm really fascinated by quantum computing. Can you explain what makes it so special?",
        sender="Alice"
    )
    
    print(f"💬 {initial_message.sender}: {initial_message.content}")
    
    # Publish initial message
    await runtime.publish_message(initial_message, topic_id=DefaultTopicId("chat_topic"))
    
    # Let the conversation run for a bit
    await asyncio.sleep(10)
    
    # Stop the runtime
    await runtime.stop()
    
    print("\n✅ Demo completed successfully!")
    print("💡 The custom agents communicated using AutoGen Core's message passing!")
    
    # Clean up
    await model_client.close()


if __name__ == "__main__":
    asyncio.run(main())