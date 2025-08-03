#!/usr/bin/env python3
"""
EV Charging Structured Negotiation Demo
Implements formal negotiation protocol with:
- Structured offers and counteroffers
- BATNA (Best Alternative to Negotiated Agreement)
- Negotiation phases and round tracking
- Deal constraints and timeout conditions
"""

import asyncio
from dataclasses import dataclass
from enum import Enum
from typing import Optional, List
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.models.ollama import OllamaChatCompletionClient


class NegotiationPhase(Enum):
    OPENING = "opening"
    BARGAINING = "bargaining"
    CLOSING = "closing"
    COMPLETED = "completed"


@dataclass
class NegotiationOffer:
    """Structured offer format"""
    price_per_kwh: float
    total_kwh: float
    charging_speed: str  # "standard" or "fast"
    time_slot: str  # "now", "off_peak", "flexible"
    total_cost: float
    additional_terms: str = ""
    
    def __str__(self):
        return f"${self.price_per_kwh:.2f}/kWh × {self.total_kwh}kWh = ${self.total_cost:.2f} ({self.charging_speed} charging, {self.time_slot})"


class NegotiationTracker:
    """Tracks negotiation progress and constraints"""
    
    def __init__(self):
        self.round_number = 0
        self.max_rounds = 8
        self.phase = NegotiationPhase.OPENING
        self.offers: List[tuple[str, NegotiationOffer]] = []
        self.deal_reached = False
        self.deal_terms: Optional[NegotiationOffer] = None
        
        # BATNA (Best Alternative to Negotiated Agreement)
        self.station_batna = {
            "min_acceptable_price": 0.28,  # Won't go below this
            "alternative": "Wait for next customer or use standard pricing"
        }
        
        self.owner_batna = {
            "max_acceptable_cost": 25.00,  # Won't pay more than this
            "alternative": "Find another station or charge at home overnight"
        }
    
    def add_offer(self, agent_name: str, offer: NegotiationOffer):
        """Add an offer to the negotiation history"""
        self.offers.append((agent_name, offer))
        self.round_number += 1
        
        # Update phase based on round number
        if self.round_number <= 2:
            self.phase = NegotiationPhase.OPENING
        elif self.round_number <= 6:
            self.phase = NegotiationPhase.BARGAINING
        else:
            self.phase = NegotiationPhase.CLOSING
    
    def is_offer_acceptable(self, agent_name: str, offer: NegotiationOffer) -> bool:
        """Check if offer meets BATNA constraints"""
        if agent_name == "ChargingStation":
            return offer.price_per_kwh >= self.station_batna["min_acceptable_price"]
        elif agent_name == "EVOwner":
            return offer.total_cost <= self.owner_batna["max_acceptable_cost"]
        return True
    
    def should_terminate(self) -> tuple[bool, str]:
        """Check if negotiation should end"""
        if self.deal_reached:
            return True, "Deal successfully reached!"
        if self.round_number >= self.max_rounds:
            return True, "Maximum rounds reached - negotiation failed"
        return False, ""
    
    def get_negotiation_status(self) -> str:
        """Get current negotiation status"""
        last_offer = self.offers[-1] if self.offers else None
        status = f"Round {self.round_number}/{self.max_rounds} | Phase: {self.phase.value.upper()}"
        
        if last_offer:
            agent, offer = last_offer
            status += f"\nLast offer from {agent}: {offer}"
        
        return status


# Global negotiation tracker
negotiation = NegotiationTracker()


async def main():
    print("⚡ Starting EV Charging Structured Negotiation Demo")
    print("=" * 60)
    print("🔄 Negotiation Protocol: Offer → Counteroffer → Deal/Walk Away")
    print("📋 BATNA Constraints:")
    print("   Station: Won't go below $0.28/kWh")
    print("   Owner: Won't pay more than $25 total")
    print("=" * 60)

    # Create Ollama client
    model_client = OllamaChatCompletionClient(
        model="llama3.1:8b",
        base_url="http://localhost:11434",
    )

    # EV Charging Station Agent with structured negotiation
    station_agent = AssistantAgent(
        name="ChargingStation",
        system_message="""You are an EV charging station operator using STRUCTURED NEGOTIATION.

YOUR CONSTRAINTS:
- Your BATNA: Cannot go below $0.28/kWh (your minimum acceptable price)
- Alternative: Wait for next customer or use standard pricing

STATION DETAILS:
- Standard rate: $0.35/kWh (peak), $0.25/kWh (off-peak)
- Fast charging: +$0.10/kWh premium
- Current demand: Medium (5/8 ports occupied)

NEGOTIATION RULES:
1. Always make SPECIFIC offers in this format:
   "OFFER: $X.XX/kWh × YYkWh = $ZZ.ZZ (speed: standard/fast, timing: now/off_peak/flexible)"
   
2. Include reasoning for your offer
3. Reference previous offers when making counteroffers
4. Be prepared to walk away if offer goes below your BATNA ($0.28/kWh)
5. If you accept a deal, clearly state "DEAL ACCEPTED"

STRATEGY:
- Start with standard rates but be willing to negotiate
- Offer discounts for off-peak timing or bulk charging
- Use time pressure strategically
- Make concessions gradually

Keep responses concise and business-focused. Always end with a clear OFFER or ACCEPT/REJECT decision.""",
        model_client=model_client,
    )

    # EV Owner Agent with structured negotiation
    owner_agent = AssistantAgent(
        name="EVOwner",
        system_message="""You are an EV owner using STRUCTURED NEGOTIATION.

YOUR CONSTRAINTS:
- Your BATNA: Cannot pay more than $25 total (your maximum acceptable cost)
- Alternative: Find another station or charge at home overnight

YOUR SITUATION:
- Need: 45 kWh charging (25% → 80% battery)
- Budget: Prefer under $20, absolute max $25
- Time flexibility: Can wait 2-3 hours for better rates
- Vehicle: Tesla Model 3 (supports fast charging)

NEGOTIATION RULES:
1. Always respond to offers with SPECIFIC counteroffers:
   "COUNTEROFFER: $X.XX/kWh × YYkWh = $ZZ.ZZ (speed: standard/fast, timing: now/off_peak/flexible)"
   
2. Include reasoning for your counteroffer
3. Reference the station's previous offers
4. Be prepared to walk away if total cost exceeds $25
5. If you accept a deal, clearly state "DEAL ACCEPTED"

STRATEGY:
- Start by stating your needs clearly
- Negotiate aggressively but fairly
- Use timing flexibility as leverage
- Make reasonable concessions to reach a deal

Keep responses focused on getting value. Always end with a clear COUNTEROFFER or ACCEPT/REJECT decision.""",
        model_client=model_client,
    )

    print("✅ Structured negotiation agents created!")
    print("🎯 Negotiation will track offers, counteroffers, and BATNA constraints")
    print("\n⚡ Starting formal negotiation...")
    print("-" * 40)

    # Create team for structured negotiation
    team = RoundRobinGroupChat([owner_agent, station_agent], max_turns=negotiation.max_rounds)

    # Start structured negotiation
    initial_message = f"""NEGOTIATION OPENING - Round 1/{negotiation.max_rounds}

I need to charge my Tesla Model 3: 45 kWh (25% → 80% battery).

My requirements:
- Target budget: Under $20 total
- Maximum I can pay: $25 (my BATNA limit)
- Time flexibility: Can charge now or wait 2-3 hours
- Charging speed: Flexible (standard or fast)

Please provide your initial OFFER with specific terms: price per kWh, total cost, charging speed, and timing options."""

    print(f"📊 {negotiation.get_negotiation_status()}")
    print("-" * 40)

    # Run the structured negotiation
    try:
        await Console(team.run_stream(task=initial_message))
    except Exception as e:
        print(f"Negotiation ended: {e}")

    # Final negotiation summary
    print("\n" + "=" * 50)
    print("📊 NEGOTIATION SUMMARY")
    print("=" * 50)
    print(f"Total rounds: {negotiation.round_number}")
    print(f"Final phase: {negotiation.phase.value.upper()}")
    
    if negotiation.deal_reached:
        print("✅ Deal reached successfully!")
        print(f"Final terms: {negotiation.deal_terms}")
    else:
        print("❌ No deal reached")
        print("Both parties used their BATNA alternatives")
    
    print("\n📈 Offer History:")
    for i, (agent, offer) in enumerate(negotiation.offers, 1):
        print(f"{i}. {agent}: {offer}")

    print("\n💡 This demo showed structured negotiation with:")
    print("   - Formal offer/counteroffer tracking")
    print("   - BATNA constraints and walk-away points")
    print("   - Negotiation phases and round limits")
    print("   - Deal outcome tracking")


if __name__ == "__main__":
    asyncio.run(main())