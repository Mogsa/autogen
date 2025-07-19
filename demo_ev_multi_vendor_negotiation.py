#!/usr/bin/env python3
"""
EV Charging Multi-Vendor Negotiation Demo
Extends the structured negotiation with multiple competing sellers and vendors:
- Multiple charging stations with different strategies
- Multiple service vendors (parking, maintenance, etc.)
- Auction-style and concurrent negotiation modes
- Market dynamics simulation
"""

import asyncio
from dataclasses import dataclass
from enum import Enum
from typing import Optional, List, Dict, Any
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.models.ollama import OllamaChatCompletionClient


class NegotiationPhase(Enum):
    OPENING = "opening"
    VENDOR_SELECTION = "vendor_selection"
    BARGAINING = "bargaining"
    CLOSING = "closing"
    COMPLETED = "completed"


class NegotiationMode(Enum):
    AUCTION = "auction"  # Sellers compete openly
    CONCURRENT = "concurrent"  # Separate simultaneous negotiations
    SEQUENTIAL = "sequential"  # One seller at a time


@dataclass
class NegotiationOffer:
    """Structured offer format"""
    price_per_kwh: float
    total_kwh: float
    charging_speed: str  # "standard" or "fast"
    time_slot: str  # "now", "off_peak", "flexible"
    total_cost: float
    additional_services: List[str] = None  # parking, maintenance, etc.
    additional_terms: str = ""
    vendor_id: str = ""
    
    def __post_init__(self):
        if self.additional_services is None:
            self.additional_services = []
    
    def __str__(self):
        services = f" + {', '.join(self.additional_services)}" if self.additional_services else ""
        return f"[{self.vendor_id}] ${self.price_per_kwh:.2f}/kWh × {self.total_kwh}kWh = ${self.total_cost:.2f} ({self.charging_speed} charging, {self.time_slot}){services}"


@dataclass
class VendorProfile:
    """Profile for each vendor/seller"""
    name: str
    vendor_type: str  # "charging_station", "service_provider", "hybrid"
    specialties: List[str]
    price_strategy: str  # "competitive", "premium", "value"
    batna_min_price: float
    location: str
    capacity: int
    current_load: int


class MultiVendorNegotiationTracker:
    """Tracks multi-vendor negotiation progress"""
    
    def __init__(self, mode: NegotiationMode = NegotiationMode.AUCTION):
        self.mode = mode
        self.round_number = 0
        self.max_rounds = 12  # Increased for multi-vendor
        self.phase = NegotiationPhase.OPENING
        self.offers: List[tuple[str, NegotiationOffer]] = []
        self.vendor_offers: Dict[str, List[NegotiationOffer]] = {}
        self.deal_reached = False
        self.winning_vendor = None
        self.deal_terms: Optional[NegotiationOffer] = None
        
        # Buyer BATNA
        self.buyer_batna = {
            "max_acceptable_cost": 25.00,
            "alternative": "Find another station or charge at home overnight"
        }
        
        # Vendor profiles
        self.vendors: Dict[str, VendorProfile] = {
            "FastCharge_Plaza": VendorProfile(
                name="FastCharge Plaza",
                vendor_type="charging_station",
                specialties=["fast_charging", "premium_location"],
                price_strategy="premium",
                batna_min_price=0.30,
                location="Downtown",
                capacity=8,
                current_load=3
            ),
            "EcoCharge_Hub": VendorProfile(
                name="EcoCharge Hub",
                vendor_type="charging_station",
                specialties=["green_energy", "competitive_pricing"],
                price_strategy="competitive",
                batna_min_price=0.25,
                location="Suburbs",
                capacity=12,
                current_load=6
            ),
            "ServicePlus_Station": VendorProfile(
                name="ServicePlus Station",
                vendor_type="hybrid",
                specialties=["maintenance", "parking", "charging"],
                price_strategy="value",
                batna_min_price=0.28,
                location="Mall",
                capacity=6,
                current_load=2
            )
        }
    
    def add_offer(self, vendor_name: str, offer: NegotiationOffer):
        """Add an offer to the negotiation history"""
        self.offers.append((vendor_name, offer))
        if vendor_name not in self.vendor_offers:
            self.vendor_offers[vendor_name] = []
        self.vendor_offers[vendor_name].append(offer)
        
        # Only increment round for buyer responses in auction mode
        if self.mode == NegotiationMode.AUCTION and vendor_name == "EVOwner":
            self.round_number += 1
        elif self.mode != NegotiationMode.AUCTION:
            self.round_number += 1
        
        # Update phase
        if self.round_number <= 2:
            self.phase = NegotiationPhase.OPENING
        elif self.round_number <= 4:
            self.phase = NegotiationPhase.VENDOR_SELECTION
        elif self.round_number <= 9:
            self.phase = NegotiationPhase.BARGAINING
        else:
            self.phase = NegotiationPhase.CLOSING
    
    def is_offer_acceptable(self, vendor_name: str, offer: NegotiationOffer) -> bool:
        """Check if offer meets BATNA constraints"""
        if vendor_name == "EVOwner":
            return offer.total_cost <= self.buyer_batna["max_acceptable_cost"]
        elif vendor_name in self.vendors:
            vendor = self.vendors[vendor_name]
            return offer.price_per_kwh >= vendor.batna_min_price
        return True
    
    def get_best_offer(self) -> Optional[tuple[str, NegotiationOffer]]:
        """Get the best offer from all vendors"""
        if not self.offers:
            return None
        
        vendor_offers = [(vendor, offer) for vendor, offer in self.offers if vendor != "EVOwner"]
        if not vendor_offers:
            return None
        
        # Sort by total cost (lowest first)
        best_offer = min(vendor_offers, key=lambda x: x[1].total_cost)
        return best_offer
    
    def should_terminate(self) -> tuple[bool, str]:
        """Check if negotiation should end"""
        if self.deal_reached:
            return True, f"Deal reached with {self.winning_vendor}!"
        if self.round_number >= self.max_rounds:
            return True, "Maximum rounds reached - negotiation failed"
        return False, ""
    
    def get_negotiation_status(self) -> str:
        """Get current negotiation status"""
        status = f"Round {self.round_number}/{self.max_rounds} | Phase: {self.phase.value.upper()} | Mode: {self.mode.value.upper()}"
        
        if self.mode == NegotiationMode.AUCTION:
            best_offer = self.get_best_offer()
            if best_offer:
                vendor, offer = best_offer
                status += f"\nCurrent best offer: {offer}"
        
        return status
    
    def get_vendor_summary(self) -> str:
        """Get summary of all vendors"""
        summary = "📊 VENDOR PROFILES:\n"
        for vendor_id, profile in self.vendors.items():
            load_pct = (profile.current_load / profile.capacity) * 100
            summary += f"• {profile.name} ({profile.vendor_type}): {profile.specialties} - {load_pct:.0f}% capacity\n"
        return summary


# Global negotiation tracker
negotiation = MultiVendorNegotiationTracker(mode=NegotiationMode.AUCTION)


async def main():
    print("⚡ Starting Multi-Vendor EV Charging Negotiation Demo")
    print("=" * 70)
    print("🏪 Multiple Vendors: FastCharge Plaza, EcoCharge Hub, ServicePlus Station")
    print("🎯 Negotiation Mode: Auction (vendors compete openly)")
    print("📋 Buyer BATNA: Won't pay more than $25 total")
    print("=" * 70)
    
    print(negotiation.get_vendor_summary())
    
    # Create Ollama client
    model_client = OllamaChatCompletionClient(
        model="llama3.1:8b",
        base_url="http://localhost:11434",
    )

    # Create multiple vendor agents
    fastcharge_agent = AssistantAgent(
        name="FastCharge_Plaza",
        system_message="""You are FastCharge Plaza, a PREMIUM charging station operator in a MULTI-VENDOR AUCTION.

YOUR PROFILE:
- Location: Downtown (premium location)
- Specialties: Fast charging, premium service
- Strategy: Premium pricing with superior service
- BATNA: Cannot go below $0.30/kWh
- Capacity: 8 ports, currently 3 occupied (37% load)

COMPETITIVE SITUATION:
- You're competing against EcoCharge Hub (budget competitor) and ServicePlus Station (value competitor)
- This is an AUCTION - all vendors see each other's offers
- Customer will choose the best overall value

YOUR ADVANTAGES:
- Fastest charging speeds (up to 250kW)
- Premium downtown location
- Superior customer service
- Covered parking included

STRATEGY:
- Lead with your premium value proposition
- Emphasize speed and convenience benefits
- Watch competitor offers and respond strategically
- Justify higher prices with superior service
- Make targeted counteroffers to beat competitors

OFFER FORMAT:
"OFFER: $X.XX/kWh × YYkWh = $ZZ.ZZ (fast charging, premium location, covered parking)"

Watch for competitor offers and respond competitively while maintaining your premium positioning.""",
        model_client=model_client,
    )

    ecocharge_agent = AssistantAgent(
        name="EcoCharge_Hub",
        system_message="""You are EcoCharge Hub, a GREEN & COMPETITIVE charging station in a MULTI-VENDOR AUCTION.

YOUR PROFILE:
- Location: Suburbs (green energy focused)
- Specialties: 100% renewable energy, competitive pricing
- Strategy: Competitive pricing with environmental benefits
- BATNA: Cannot go below $0.25/kWh
- Capacity: 12 ports, currently 6 occupied (50% load)

COMPETITIVE SITUATION:
- You're competing against FastCharge Plaza (premium) and ServicePlus Station (value)
- This is an AUCTION - all vendors see each other's offers
- Customer will choose the best overall value

YOUR ADVANTAGES:
- Lowest operating costs (renewable energy)
- Competitive pricing flexibility
- Environmental appeal
- High capacity with moderate load

STRATEGY:
- Lead with competitive pricing
- Emphasize environmental benefits
- Undercut premium competitors aggressively
- Use your cost advantage to win on price
- Offer flexible timing incentives

OFFER FORMAT:
"OFFER: $X.XX/kWh × YYkWh = $ZZ.ZZ (green energy, competitive rates, flexible timing)"

Be aggressive on pricing while highlighting your environmental advantages.""",
        model_client=model_client,
    )

    serviceplus_agent = AssistantAgent(
        name="ServicePlus_Station",
        system_message="""You are ServicePlus Station, a FULL-SERVICE charging & maintenance hub in a MULTI-VENDOR AUCTION.

YOUR PROFILE:
- Location: Mall (convenient shopping location)
- Specialties: Charging + maintenance + parking + shopping
- Strategy: Value bundling with additional services
- BATNA: Cannot go below $0.28/kWh
- Capacity: 6 ports, currently 2 occupied (33% load)

COMPETITIVE SITUATION:
- You're competing against FastCharge Plaza (premium) and EcoCharge Hub (budget)
- This is an AUCTION - all vendors see each other's offers
- Customer will choose the best overall value

YOUR ADVANTAGES:
- Bundled services (maintenance, parking, shopping)
- Mall location with amenities
- Low current load means immediate availability
- Comprehensive value proposition

STRATEGY:
- Lead with bundled value proposition
- Offer additional services to justify pricing
- Position between premium and budget competitors
- Emphasize convenience and comprehensive service
- Bundle discounts for multiple services

OFFER FORMAT:
"OFFER: $X.XX/kWh × YYkWh = $ZZ.ZZ (charging + parking + mall access, immediate availability)"

Focus on total value and convenience, not just charging price.""",
        model_client=model_client,
    )

    # EV Owner Agent (buyer)
    owner_agent = AssistantAgent(
        name="EVOwner",
        system_message="""You are an EV owner participating in a MULTI-VENDOR AUCTION for charging services.

YOUR SITUATION:
- Need: 45 kWh charging (25% → 80% battery)
- Budget: Prefer under $20, absolute max $25 (your BATNA)
- Alternative: Find another location or charge at home overnight
- Time: Can wait 2-3 hours for better rates or benefits

VENDORS COMPETING:
- FastCharge Plaza: Premium location, fast charging, higher prices
- EcoCharge Hub: Green energy, competitive pricing, suburban location
- ServicePlus Station: Full service, maintenance, mall location

AUCTION RULES:
1. All vendors will make offers - you evaluate ALL offers
2. Compare total value, not just price (speed, location, services)
3. Ask follow-up questions to clarify offers
4. Negotiate with the most promising vendors
5. Make your final decision based on best overall value

RESPONSE FORMAT:
- "EVALUATING OFFERS..." (when reviewing multiple offers)
- "QUESTIONS FOR [Vendor]: ..." (when seeking clarification)
- "COUNTEROFFER TO [Vendor]: ..." (when negotiating)
- "SELECTING [Vendor]: ..." (when making final choice)

STRATEGY:
- Let vendors compete first
- Ask strategic questions to get better offers
- Play vendors against each other appropriately
- Consider total value, not just lowest price
- Make final decision based on best overall deal

Remember: This is an auction where vendors compete for your business!""",
        model_client=model_client,
    )

    print("✅ Multi-vendor negotiation agents created!")
    print("🎯 Auction mode: Vendors compete openly for the customer")
    print("\n⚡ Starting multi-vendor negotiation...")
    print("-" * 50)

    # Create team with all vendors plus buyer
    team = RoundRobinGroupChat(
        [owner_agent, fastcharge_agent, ecocharge_agent, serviceplus_agent], 
        max_turns=negotiation.max_rounds
    )

    # Start multi-vendor negotiation
    initial_message = f"""MULTI-VENDOR AUCTION - Round 1/{negotiation.max_rounds}

I need to charge my Tesla Model 3: 45 kWh (25% → 80% battery).

AUCTION INVITATION TO ALL VENDORS:
Please submit your best OFFERS including:
- Pricing: $/kWh and total cost
- Charging speed and timing options
- Additional services or benefits
- Location advantages

My requirements:
- Target budget: Under $20 total
- Maximum budget: $25 (my walk-away point)
- Time flexibility: Can charge now or wait 2-3 hours
- Looking for best overall value (not just lowest price)

All vendors please make your opening offers - I'll evaluate them all before responding."""

    print(f"📊 {negotiation.get_negotiation_status()}")
    print("-" * 50)

    # Run the multi-vendor negotiation
    try:
        await Console(team.run_stream(task=initial_message))
    except Exception as e:
        print(f"Negotiation ended: {e}")

    # Final negotiation summary
    print("\n" + "=" * 60)
    print("📊 MULTI-VENDOR NEGOTIATION SUMMARY")
    print("=" * 60)
    print(f"Total rounds: {negotiation.round_number}")
    print(f"Final phase: {negotiation.phase.value.upper()}")
    print(f"Mode: {negotiation.mode.value.upper()}")
    
    if negotiation.deal_reached:
        print(f"✅ Deal reached with {negotiation.winning_vendor}!")
        print(f"Winning terms: {negotiation.deal_terms}")
    else:
        print("❌ No deal reached")
        best_offer = negotiation.get_best_offer()
        if best_offer:
            vendor, offer = best_offer
            print(f"Best offer was from {vendor}: {offer}")
        print("Buyer used BATNA alternative")
    
    print("\n📈 Vendor Competition Analysis:")
    for vendor_id, offers in negotiation.vendor_offers.items():
        print(f"\n{vendor_id}:")
        for i, offer in enumerate(offers, 1):
            print(f"  {i}. {offer}")
    
    print("\n💡 This demo demonstrated:")
    print("   - Multi-vendor competitive negotiation")
    print("   - Auction-style bidding process")
    print("   - Vendor specialization and differentiation")
    print("   - Value-based decision making (not just price)")
    print("   - Market dynamics simulation")


if __name__ == "__main__":
    asyncio.run(main())