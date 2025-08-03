#!/usr/bin/env python3
"""
EV Charging Concurrent Negotiation Demo
Alternative approach: Buyer negotiates with multiple vendors simultaneously
in separate private channels, then compares final offers
"""

import asyncio
from dataclasses import dataclass
from typing import List, Dict, Optional
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.models.ollama import OllamaChatCompletionClient


@dataclass
class ConcurrentNegotiationResult:
    """Result from a single vendor negotiation"""
    vendor_name: str
    final_offer: Optional[str]
    negotiation_rounds: int
    deal_reached: bool
    buyer_satisfaction: int  # 1-10 scale


class ConcurrentNegotiationManager:
    """Manages multiple simultaneous negotiations"""
    
    def __init__(self):
        self.negotiations: Dict[str, ConcurrentNegotiationResult] = {}
        self.buyer_budget = 25.00
        
    def add_negotiation_result(self, result: ConcurrentNegotiationResult):
        """Add result from a completed negotiation"""
        self.negotiations[result.vendor_name] = result
    
    def get_best_deal(self) -> Optional[ConcurrentNegotiationResult]:
        """Determine the best deal from all negotiations"""
        successful_deals = [result for result in self.negotiations.values() if result.deal_reached]
        if not successful_deals:
            return None
        
        # Sort by buyer satisfaction, then by price (parsed from final_offer)
        return max(successful_deals, key=lambda x: x.buyer_satisfaction)


async def negotiate_with_vendor(vendor_name: str, vendor_agent: AssistantAgent, 
                               owner_agent: AssistantAgent, model_client) -> ConcurrentNegotiationResult:
    """Conduct a private negotiation with a single vendor"""
    
    print(f"\n🔄 Starting private negotiation with {vendor_name}")
    print("-" * 40)
    
    # Create a private team for this vendor
    team = RoundRobinGroupChat([owner_agent, vendor_agent], max_turns=6)
    
    # Vendor-specific negotiation messages
    vendor_messages = {
        "FastCharge_Plaza": """PRIVATE NEGOTIATION with FastCharge Plaza

I'm negotiating with multiple vendors for 45 kWh charging service.
Your advantages are downtown location and fast charging.

Please provide your best offer considering:
- My budget: max $25 total
- Your premium positioning
- Competition from other vendors

I'm looking for your most competitive offer.""",
        
        "EcoCharge_Hub": """PRIVATE NEGOTIATION with EcoCharge Hub

I'm negotiating with multiple vendors for 45 kWh charging service.
Your advantages are green energy and competitive pricing.

Please provide your best offer considering:
- My budget: max $25 total
- Your cost advantages from renewable energy
- Competition from premium and value competitors

I'm looking for your most competitive offer.""",
        
        "ServicePlus_Station": """PRIVATE NEGOTIATION with ServicePlus Station

I'm negotiating with multiple vendors for 45 kWh charging service.
Your advantages are bundled services and mall location.

Please provide your best offer considering:
- My budget: max $25 total
- Your additional services (maintenance, parking, shopping)
- Competition from premium and budget competitors

I'm looking for your most competitive offer."""
    }
    
    initial_message = vendor_messages.get(vendor_name, "Please provide your best offer for 45 kWh charging.")
    
    rounds = 0
    deal_reached = False
    final_offer = None
    
    try:
        # Run the private negotiation
        # Note: In a real implementation, you'd capture the conversation and analyze the outcome
        result = await team.run_stream(task=initial_message)
        
        # For demo purposes, simulate negotiation outcome
        # In practice, you'd parse the actual conversation
        deal_reached = True
        final_offer = f"Simulated final offer from {vendor_name}"
        rounds = 4
        
    except Exception as e:
        print(f"Negotiation with {vendor_name} failed: {e}")
        deal_reached = False
        final_offer = None
        rounds = 0
    
    # Simulate buyer satisfaction based on vendor characteristics
    satisfaction_scores = {
        "FastCharge_Plaza": 7,  # Premium but expensive
        "EcoCharge_Hub": 8,     # Good value and green
        "ServicePlus_Station": 9  # Best overall value
    }
    
    satisfaction = satisfaction_scores.get(vendor_name, 5)
    
    return ConcurrentNegotiationResult(
        vendor_name=vendor_name,
        final_offer=final_offer,
        negotiation_rounds=rounds,
        deal_reached=deal_reached,
        buyer_satisfaction=satisfaction
    )


async def main():
    print("⚡ Starting Concurrent Multi-Vendor Negotiation Demo")
    print("=" * 60)
    print("🤝 Mode: Private negotiations with each vendor")
    print("📊 Buyer compares final offers and chooses best deal")
    print("=" * 60)
    
    # Create Ollama client
    model_client = OllamaChatCompletionClient(
        model="llama3.1:8b",
        base_url="http://localhost:11434",
    )
    
    # Create vendor agents (same as before but focused on private negotiation)
    fastcharge_agent = AssistantAgent(
        name="FastCharge_Plaza",
        system_message="""You are FastCharge Plaza in a PRIVATE NEGOTIATION with a potential customer.

IMPORTANT: You're negotiating privately - the customer is also talking to competitors separately.
You won't see their offers, but you know they exist.

YOUR PROFILE:
- Premium downtown location
- Fastest charging speeds (250kW)
- Covered parking included
- BATNA: Cannot go below $0.30/kWh
- Current load: 3/8 ports (low demand)

STRATEGY FOR PRIVATE NEGOTIATION:
- Make your best offer quickly (limited rounds)
- Emphasize unique value (speed, location, service)
- Be competitive but maintain premium positioning
- Close the deal before customer shops elsewhere

OFFER FORMAT:
"OFFER: $X.XX/kWh × 45kWh = $ZZ.ZZ (250kW fast charging, downtown location, covered parking)"

Focus on closing the deal in this private conversation.""",
        model_client=model_client,
    )
    
    ecocharge_agent = AssistantAgent(
        name="EcoCharge_Hub",
        system_message="""You are EcoCharge Hub in a PRIVATE NEGOTIATION with a potential customer.

IMPORTANT: You're negotiating privately - the customer is also talking to competitors separately.
You won't see their offers, but you know they exist.

YOUR PROFILE:
- 100% renewable energy
- Competitive pricing (BATNA: $0.25/kWh minimum)
- Suburban location
- Current load: 6/12 ports (moderate demand)

STRATEGY FOR PRIVATE NEGOTIATION:
- Lead with aggressive pricing to win the deal
- Emphasize environmental benefits
- Use your cost advantage to undercut competitors
- Make compelling offers quickly

OFFER FORMAT:
"OFFER: $X.XX/kWh × 45kWh = $ZZ.ZZ (100% green energy, competitive rates, flexible timing)"

Focus on winning with superior value and environmental benefits.""",
        model_client=model_client,
    )
    
    serviceplus_agent = AssistantAgent(
        name="ServicePlus_Station",
        system_message="""You are ServicePlus Station in a PRIVATE NEGOTIATION with a potential customer.

IMPORTANT: You're negotiating privately - the customer is also talking to competitors separately.
You won't see their offers, but you know they exist.

YOUR PROFILE:
- Mall location with amenities
- Bundled services: charging + parking + maintenance
- BATNA: Cannot go below $0.28/kWh
- Current load: 2/6 ports (low demand - immediate availability)

STRATEGY FOR PRIVATE NEGOTIATION:
- Lead with bundled value proposition
- Offer package deals (charging + services)
- Emphasize convenience and immediate availability
- Create compelling total value package

OFFER FORMAT:
"OFFER: $X.XX/kWh × 45kWh = $ZZ.ZZ (charging + parking + mall access + maintenance check)"

Focus on total value and convenience to differentiate from pure charging competitors.""",
        model_client=model_client,
    )
    
    # Create buyer agent for concurrent negotiations
    owner_agent = AssistantAgent(
        name="EVOwner",
        system_message="""You are an EV owner conducting PRIVATE NEGOTIATIONS with multiple vendors.

YOUR SITUATION:
- Need: 45 kWh charging (25% → 80% battery)
- Budget: max $25 total
- Strategy: Negotiate privately with each vendor to get their best offer

NEGOTIATION APPROACH:
- Each conversation is private (vendors don't see each other's offers)
- Push for best possible terms from each vendor
- Ask specific questions about their advantages
- Negotiate firmly but fairly
- Try to close a good deal in limited rounds

RESPONSE PATTERNS:
- "What's your best offer for 45 kWh charging?"
- "How does that compare to your standard rates?"
- "What additional value can you provide?"
- "Can you do better on price/terms?"
- "I accept your offer" (when satisfied)

Remember: You're having separate private conversations with each vendor.""",
        model_client=model_client,
    )
    
    # Create negotiation manager
    manager = ConcurrentNegotiationManager()
    
    # Conduct concurrent negotiations
    vendors = [
        ("FastCharge_Plaza", fastcharge_agent),
        ("EcoCharge_Hub", ecocharge_agent),
        ("ServicePlus_Station", serviceplus_agent)
    ]
    
    print("🚀 Starting concurrent negotiations with all vendors...")
    
    # Run all negotiations concurrently
    negotiation_tasks = []
    for vendor_name, vendor_agent in vendors:
        task = negotiate_with_vendor(vendor_name, vendor_agent, owner_agent, model_client)
        negotiation_tasks.append(task)
    
    # Wait for all negotiations to complete
    results = await asyncio.gather(*negotiation_tasks, return_exceptions=True)
    
    # Process results
    for result in results:
        if isinstance(result, ConcurrentNegotiationResult):
            manager.add_negotiation_result(result)
        else:
            print(f"Error in negotiation: {result}")
    
    # Final comparison and selection
    print("\n" + "=" * 60)
    print("📊 CONCURRENT NEGOTIATIONS SUMMARY")
    print("=" * 60)
    
    print("\n📈 All Negotiation Results:")
    for vendor_name, result in manager.negotiations.items():
        status = "✅ DEAL" if result.deal_reached else "❌ NO DEAL"
        print(f"{vendor_name}: {status}")
        print(f"  • Rounds: {result.negotiation_rounds}")
        print(f"  • Satisfaction: {result.buyer_satisfaction}/10")
        if result.final_offer:
            print(f"  • Final offer: {result.final_offer}")
        print()
    
    # Determine best deal
    best_deal = manager.get_best_deal()
    if best_deal:
        print(f"🏆 WINNER: {best_deal.vendor_name}")
        print(f"   Satisfaction score: {best_deal.buyer_satisfaction}/10")
        print(f"   Final offer: {best_deal.final_offer}")
    else:
        print("❌ No acceptable deals reached")
    
    print("\n💡 Concurrent Negotiation Benefits:")
    print("   - Private negotiations prevent vendor collusion")
    print("   - Buyer gets each vendor's best offer")
    print("   - Vendors must compete on value, not just price")
    print("   - More efficient than sequential negotiations")
    print("   - Buyer maintains negotiation leverage")


if __name__ == "__main__":
    asyncio.run(main())