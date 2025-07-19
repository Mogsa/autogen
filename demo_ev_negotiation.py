#!/usr/bin/env python3
"""
EV Charging Negotiation Demo
Two agents negotiate charging prices and terms:
- EVStationAgent: Represents charging station with pricing strategy
- EVOwnerAgent: Represents EV owner with budget and charging needs
"""

import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.models.ollama import OllamaChatCompletionClient


async def main():
    print("⚡ Starting EV Charging Negotiation Demo")
    print("=" * 50)

    # Create Ollama client
    model_client = OllamaChatCompletionClient(
        model="llama3.1:8b",
        base_url="http://localhost:11434",
    )

    # EV Charging Station Agent
    station_agent = AssistantAgent(
        name="ChargingStation",
        system_message="""You are an EV charging station operator. Your goal is to maximize profit while providing competitive service.

Your station details:
- Base rate: $0.35/kWh during peak hours (6am-10pm)
- Off-peak rate: $0.25/kWh (10pm-6am)
- Fast charging premium: +$0.10/kWh
- Capacity: 8 charging ports
- Current demand: Medium (5/8 ports occupied)

Negotiation strategy:
- Start with standard rates
- Offer discounts for off-peak hours or bulk charging
- Consider loyalty programs for regular customers
- Factor in current demand when pricing
- Be willing to negotiate but maintain profitability
- Always end with a clear offer: price per kWh, charging speed, time slot

Keep responses concise and business-focused.""",
        model_client=model_client,
    )

    # EV Owner Agent
    owner_agent = AssistantAgent(
        name="EVOwner",
        system_message="""You are an EV owner looking to charge your vehicle. Your goal is to get the best deal while meeting your charging needs.

Your situation:
- Battery level: 25% (need to charge to 80%)
- Required charging: ~45 kWh
- Budget: Prefer under $20 total
- Time flexibility: Can charge now or wait 2-3 hours
- Vehicle: Tesla Model 3 (supports fast charging)

Negotiation strategy:
- Start by stating your charging needs
- Ask about available options and pricing
- Negotiate for better rates, especially for off-peak
- Consider bulk charging discounts
- Be willing to adjust timing for better prices
- Make counteroffers when prices are too high
- Always respond with your position: accept, counteroffer, or walk away

Keep responses concise and focused on getting value.""",
        model_client=model_client,
    )

    print("✅ Negotiation agents created!")
    print("   - ChargingStation: Profit-focused operator")
    print("   - EVOwner: Cost-conscious driver")
    print("\n⚡ Starting negotiation...")
    print("-" * 40)

    # Create team for negotiation (limited turns for focused negotiation)
    team = RoundRobinGroupChat([owner_agent, station_agent], max_turns=12)

    # Start negotiation
    initial_message = """Hi! I'm looking to charge my Tesla Model 3. I need about 45 kWh to get from 25% to 80% battery. 

What are your current rates and availability? I'm hoping to keep the total cost under $20 if possible. I have some flexibility on timing if that helps with pricing."""

    # Run the negotiation
    await Console(team.run_stream(task=initial_message))

    print("\n✅ EV Charging Negotiation Demo completed!")
    print("💡 The agents negotiated pricing and terms using local Llama 3.1!")


if __name__ == "__main__":
    asyncio.run(main())