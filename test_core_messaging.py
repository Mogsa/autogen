#!/usr/bin/env python3
"""
Minimal test of autogen-core message passing
"""

import asyncio
from dataclasses import dataclass
from autogen_core import (
    RoutedAgent, 
    SingleThreadedAgentRuntime, 
    message_handler, 
    DefaultTopicId, 
    MessageContext, 
    default_subscription,
    AgentId
)

@dataclass
class TestMessage:
    content: str

@default_subscription
class ReceiverAgent(RoutedAgent):
    def __init__(self):
        super().__init__("Test Receiver")
        print("[Receiver] 🎯 Initialized")
    
    @message_handler
    async def handle_test_message(self, message: TestMessage, ctx: MessageContext) -> None:
        print(f"[Receiver] 📩 RECEIVED MESSAGE: {message.content}")

@default_subscription
class SenderAgent(RoutedAgent):
    def __init__(self):
        super().__init__("Test Sender")
        print("[Sender] 🚀 Initialized")
    
    async def send_test_message(self):
        print("[Sender] 📤 Sending test message...")
        message = TestMessage(content="Hello from sender!")
        await self.publish_message(message, DefaultTopicId("test"))
        print("[Sender] ✅ Message sent")

async def main():
    print("🧪 Testing AutoGen Core Message Passing")
    print("=" * 40)
    
    runtime = SingleThreadedAgentRuntime()
    
    # Create and register agents
    receiver = ReceiverAgent()
    sender = SenderAgent()
    
    await ReceiverAgent.register(runtime, "receiver", lambda: receiver)
    await SenderAgent.register(runtime, "sender", lambda: sender)
    
    runtime.start()
    
    # Send test message via runtime AFTER starting
    print("\n📡 Sending message to receiver...")
    message = TestMessage(content="Hello from runtime!")
    await runtime.send_message(message, AgentId("receiver", "default"))
    
    # Give time for message processing
    await asyncio.sleep(1)
    
    await runtime.stop_when_idle()
    print("\n✅ Test complete")

if __name__ == "__main__":
    asyncio.run(main())