#!/usr/bin/env python3
"""
Scaling Test: Why 2 agents work but 4+ agents fail
This test demonstrates the exact scaling issue.
"""

import asyncio
from dataclasses import dataclass
from autogen_core import (
    RoutedAgent, 
    SingleThreadedAgentRuntime, 
    message_handler, 
    DefaultTopicId, 
    MessageContext,
    default_subscription
)

@dataclass
class TestMessage:
    from_agent: str
    to_agent: str
    content: str

@default_subscription
class TestAgent(RoutedAgent):
    def __init__(self, agent_id: str):
        super().__init__(f"Agent {agent_id}")
        self.agent_id = agent_id
        self.messages_received = 0
        print(f"[{self.agent_id}] 🤖 Created")
    
    @message_handler
    async def handle_test_message(self, message: TestMessage, ctx: MessageContext) -> None:
        print(f"[{self.agent_id}] 🔔 HANDLER TRIGGERED! Message: {message.content}")
        
        if message.to_agent != self.agent_id:
            print(f"[{self.agent_id}] ⏭️ Skipping - not for me")
            return
            
        self.messages_received += 1
        print(f"[{self.agent_id}] ✅ Processed message #{self.messages_received}: {message.content}")

async def test_scaling(num_agents: int):
    """Test with different numbers of agents"""
    print(f"\n🧪 TESTING WITH {num_agents} AGENTS")
    print("=" * 40)
    
    runtime = SingleThreadedAgentRuntime()
    
    # Create agents
    agents = []
    for i in range(1, num_agents + 1):
        agent = TestAgent(f"Agent{i}")
        agents.append(agent)
        await TestAgent.register(runtime, f"agent{i}", lambda a=agent: a)
    
    print(f"✅ Registered {num_agents} agents")
    
    # Start runtime
    runtime.start()
    
    # Send messages to each agent
    print(f"\n📤 Sending {num_agents} messages...")
    for i, agent in enumerate(agents, 1):
        message = TestMessage(
            from_agent="System",
            to_agent=f"Agent{i}",
            content=f"Hello Agent{i}!"
        )
        print(f"📨 Sending to Agent{i}")
        await runtime.publish_message(message, DefaultTopicId())
        await asyncio.sleep(0.1)  # Small delay
    
    # Wait for processing
    print("\n⏳ Processing...")
    await asyncio.sleep(2)
    
    # Check results
    print(f"\n📊 RESULTS FOR {num_agents} AGENTS:")
    total_received = 0
    for agent in agents:
        print(f"  {agent.agent_id}: {agent.messages_received} messages received")
        total_received += agent.messages_received
    
    expected = num_agents
    print(f"\n🎯 Expected: {expected} messages")
    print(f"🎯 Actual: {total_received} messages")
    
    if total_received == expected:
        print("✅ SUCCESS!")
    else:
        print("❌ FAILED!")
    
    await runtime.stop_when_idle()
    return total_received == expected

async def main():
    """Test scaling from 2 to 6 agents"""
    print("🔬 AutoGen Core Scaling Test")
    print("Testing why 2 agents work but 4+ agents fail")
    
    results = {}
    
    # Test with 2, 3, 4, 5, 6 agents
    for num_agents in [2, 3, 4, 5, 6]:
        success = await test_scaling(num_agents)
        results[num_agents] = success
        await asyncio.sleep(1)  # Gap between tests
    
    print("\n" + "=" * 50)
    print("🎯 SCALING TEST SUMMARY")
    print("=" * 50)
    
    for num_agents, success in results.items():
        status = "✅ WORKS" if success else "❌ FAILS"
        print(f"{num_agents} agents: {status}")
    
    # Find the breaking point
    working = [n for n, s in results.items() if s]
    failing = [n for n, s in results.items() if not s]
    
    if working and failing:
        max_working = max(working)
        min_failing = min(failing)
        print(f"\n🔍 SCALING BREAKS between {max_working} and {min_failing} agents")
    else:
        print(f"\n🔍 All tests: {'PASSED' if all(results.values()) else 'FAILED'}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n🛑 Test interrupted")
    except Exception as e:
        print(f"\n\n❌ Test error: {e}")
        import traceback
        traceback.print_exc() 