#!/usr/bin/env python3
"""
Clean EV Charging Marketplace
No jank, no AutoGen complexity - just clean async negotiation that works.
"""

import asyncio
import json
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Callable
from enum import Enum
from autogen_ext.models.ollama import OllamaChatCompletionClient
from autogen_core.models import UserMessage

# =====================================================================================
# GLOBAL MARKET STATE - PREVENTS MULTIPLE DEALS
# =====================================================================================

class MarketState:
    """Global market state to prevent EVs from making multiple deals"""
    
    def __init__(self):
        self.ev_deals: Dict[str, str] = {}  # ev_id -> station_id
        self.station_deals: Dict[str, str] = {}  # station_id -> ev_id
        
    def can_make_deal(self, ev_id: str, station_id: str) -> bool:
        """Check if EV and station can make a deal"""
        return ev_id not in self.ev_deals and station_id not in self.station_deals
    
    def register_deal(self, ev_id: str, station_id: str) -> bool:
        """Register a deal between EV and station"""
        if self.can_make_deal(ev_id, station_id):
            self.ev_deals[ev_id] = station_id
            self.station_deals[station_id] = ev_id
            return True
        return False
    
    def has_deal(self, ev_id: str) -> bool:
        """Check if EV already has a deal"""
        return ev_id in self.ev_deals

# Global market state
market_state = MarketState()

# =====================================================================================
# CLEAN MESSAGE SYSTEM
# =====================================================================================

class MessageType(Enum):
    OFFER = "offer"
    COUNTER_OFFER = "counter_offer"
    ACCEPTANCE = "acceptance"
    REJECTION = "rejection"
    DEAL_COMPLETE = "deal_complete"

@dataclass
class Message:
    type: MessageType
    from_agent: str
    to_agent: str
    data: Dict
    timestamp: float
    message_id: str

@dataclass
class Offer:
    station_id: str
    price_per_kwh: float
    available_slots: int
    offer_id: str

@dataclass
class Deal:
    ev_id: str
    station_id: str
    price_per_kwh: float
    energy_kwh: float
    total_cost: float

# =====================================================================================
# CLEAN MESSAGE BUS - NO JANK
# =====================================================================================

class MessageBus:
    """Clean, simple message bus for agent communication"""
    
    def __init__(self):
        self.subscribers: Dict[str, List[Callable]] = {}
        self.message_history: List[Message] = []
        
    def subscribe(self, agent_id: str, handler):
        """Subscribe agent to receive messages"""
        if agent_id not in self.subscribers:
            self.subscribers[agent_id] = []
        self.subscribers[agent_id].append(handler)
        
    async def publish(self, message: Message):
        """Publish message to target agent"""
        self.message_history.append(message)
        
        # Direct delivery to target agent
        if message.to_agent in self.subscribers:
            for handler in self.subscribers[message.to_agent]:
                try:
                    await handler(message)
                except Exception as e:
                    print(f"❌ Handler error: {e}")
        
        # Also send to marketplace coordinator
        if "coordinator" in self.subscribers and message.to_agent != "coordinator":
            for handler in self.subscribers["coordinator"]:
                try:
                    await handler(message)
                except Exception as e:
                    pass  # Coordinator errors are non-critical

# =====================================================================================
# EV AGENT - CLEAN & SMART
# =====================================================================================

class EVAgent:
    """Clean EV agent that makes immediate smart decisions"""
    
    def __init__(self, ev_id: str, max_budget: float, energy_needed: float, message_bus: MessageBus):
        self.ev_id = ev_id
        self.max_budget = max_budget
        self.energy_needed = energy_needed
        self.message_bus = message_bus
        self.received_offers: Dict[str, Offer] = {}
        self.active_negotiations: Dict[str, int] = {}  # station_id -> counter_count
        self.has_deal = False
        self.max_counters = 2
        
        # LLM for smart decisions
        self.llm = OllamaChatCompletionClient(model="llama3.1:8b")
        
        # Subscribe to messages
        message_bus.subscribe(ev_id, self.handle_message)
        
        print(f"[{self.ev_id}] 🚗 Ready to negotiate - Budget: ${max_budget:.3f}/kWh, Need: {energy_needed:.1f} kWh")
    
    async def handle_message(self, message: Message):
        """Handle incoming messages cleanly"""
        # CRITICAL: Check global market deal status
        if market_state.has_deal(self.ev_id):
            print(f"[{self.ev_id}] ⏭️ Ignoring {message.type.value} from {message.from_agent} - already have deal with {market_state.ev_deals[self.ev_id]}")
            return
            
        if message.type == MessageType.OFFER:
            await self._handle_offer(message)
        elif message.type == MessageType.DEAL_COMPLETE:
            await self._handle_deal_complete(message)
    
    async def _handle_offer(self, message: Message):
        """Handle charging station offers with immediate decision"""
        # Double-check global market deal status
        if market_state.has_deal(self.ev_id):
            print(f"[{self.ev_id}] ⏭️ Offer from {message.from_agent} arrived too late - already have deal with {market_state.ev_deals[self.ev_id]}")
            return
            
        offer_data = message.data
        offer = Offer(**offer_data)
        
        print(f"[{self.ev_id}] 📩 Offer from {offer.station_id}: ${offer.price_per_kwh:.3f}/kWh")
        
        self.received_offers[offer.station_id] = offer
        
        # Make immediate smart decision
        decision = await self._make_smart_decision(offer)
        await self._execute_decision(decision, offer, message.from_agent)
    
    async def _make_smart_decision(self, offer: Offer) -> Dict:
        """Use LLM to make immediate smart negotiation decisions"""
        
        context = {
            "budget": self.max_budget,
            "energy_needed": self.energy_needed,
            "offer_price": offer.price_per_kwh,
            "station": offer.station_id,
            "previous_offers": {k: v.price_per_kwh for k, v in self.received_offers.items()},
            "negotiations": self.active_negotiations.get(offer.station_id, 0)
        }
        
        prompt = f"""You are {self.ev_id}, an EV owner making IMMEDIATE negotiation decisions.

SITUATION:
- Your max budget: ${self.max_budget:.3f}/kWh
- Energy needed: {self.energy_needed:.1f} kWh
- Current offer: ${offer.price_per_kwh:.3f}/kWh from {offer.station_id}
- Counter-offers made to {offer.station_id}: {self.active_negotiations.get(offer.station_id, 0)}/{self.max_counters}

OFFERS RECEIVED SO FAR:
{json.dumps({k: f"${v.price_per_kwh:.3f}/kWh" for k, v in self.received_offers.items()}, indent=2)}

DECISION TIME - Choose ONE action RIGHT NOW:

1. ACCEPT - Take this offer immediately
2. COUNTER - Make a counter-offer (if you have counter-offers left)
3. WAIT - Wait for better offers from other stations

Strategy: You want the lowest price but need to be realistic about market conditions.

RESPOND WITH VALID JSON ONLY:
{{"action": "ACCEPT", "reasoning": "Why accepting"}}
{{"action": "COUNTER", "price": 0.145, "reasoning": "Why counter-offering"}}
{{"action": "WAIT", "reasoning": "Why waiting"}}

JSON RESPONSE:"""

        try:
            response = await self.llm.create(
                messages=[UserMessage(content=prompt, source="user")]
            )
            
            decision_text = response.content
            if isinstance(decision_text, str):
                decision = json.loads(decision_text)
            else:
                decision = {"action": "WAIT", "reasoning": "Could not parse response"}
                
            print(f"[{self.ev_id}] 🧠 Decision: {decision['action']} - {decision['reasoning']}")
            return decision
            
        except Exception as e:
            print(f"[{self.ev_id}] ⚠️ LLM error: {e}, using fallback")
            # Simple fallback logic
            if offer.price_per_kwh <= self.max_budget:
                return {"action": "ACCEPT", "reasoning": "Within budget (fallback)"}
            elif self.active_negotiations.get(offer.station_id, 0) < self.max_counters:
                return {"action": "COUNTER", "price": self.max_budget * 0.95, "reasoning": "Counter-offer (fallback)"}
            else:
                return {"action": "WAIT", "reasoning": "Wait for better (fallback)"}
    
    async def _execute_decision(self, decision: Dict, offer: Offer, station_agent: str):
        """Execute the negotiation decision immediately"""
        # Final check before executing any action using global market state
        if market_state.has_deal(self.ev_id):
            print(f"[{self.ev_id}] ⏭️ Decision canceled - already have deal with {market_state.ev_deals[self.ev_id]}")
            return
            
        action = decision.get("action", "WAIT")
        
        if action == "ACCEPT":
            # REGISTER DEAL IN GLOBAL MARKET STATE IMMEDIATELY
            if market_state.register_deal(self.ev_id, offer.station_id):
                self.has_deal = True
                print(f"[{self.ev_id}] ✅ ACCEPTING {offer.station_id} at ${offer.price_per_kwh:.3f}/kWh")
                
                # Send acceptance
                acceptance_msg = Message(
                    type=MessageType.ACCEPTANCE,
                    from_agent=self.ev_id,
                    to_agent=station_agent,
                    data={
                        "offer_id": offer.offer_id,
                        "accepted_price": offer.price_per_kwh
                    },
                    timestamp=asyncio.get_event_loop().time(),
                    message_id=f"{self.ev_id}_accept_{offer.offer_id}"
                )
                
                await self.message_bus.publish(acceptance_msg)
            else:
                print(f"[{self.ev_id}] ❌ Cannot accept {offer.station_id} - market conflict detected")
            
        elif action == "COUNTER" and self.active_negotiations.get(offer.station_id, 0) < self.max_counters:
            counter_price = decision.get("price", self.max_budget * 0.9)
            print(f"[{self.ev_id}] 🔄 COUNTER-OFFER to {offer.station_id}: ${counter_price:.3f}/kWh")
            
            # Track negotiation
            self.active_negotiations[offer.station_id] = self.active_negotiations.get(offer.station_id, 0) + 1
            
            # Send counter-offer
            counter_msg = Message(
                type=MessageType.COUNTER_OFFER,
                from_agent=self.ev_id,
                to_agent=station_agent,
                data={
                    "original_offer_id": offer.offer_id,
                    "counter_price": counter_price,
                    "counter_id": f"{self.ev_id}_counter_{offer.station_id}_{self.active_negotiations[offer.station_id]}"
                },
                timestamp=asyncio.get_event_loop().time(),
                message_id=f"{self.ev_id}_counter_{offer.offer_id}"
            )
            
            await self.message_bus.publish(counter_msg)
            
        else:
            print(f"[{self.ev_id}] ⏳ WAITING for better offers...")
    
    async def _handle_deal_complete(self, message: Message):
        """Handle deal completion notification"""
        deal_data = message.data
        if deal_data.get("ev_id") == self.ev_id:
            total_cost = deal_data["total_cost"]
            print(f"[{self.ev_id}] 🎉 DEAL COMPLETE! Total cost: ${total_cost:.2f}")

# =====================================================================================
# CHARGING STATION AGENT - CLEAN & COMPETITIVE
# =====================================================================================

class ChargingStationAgent:
    """Clean charging station agent with competitive pricing"""
    
    def __init__(self, station_id: str, electricity_cost: float, min_margin: float, message_bus: MessageBus):
        self.station_id = station_id
        self.electricity_cost = electricity_cost
        self.min_margin = min_margin
        self.min_price = electricity_cost * (1 + min_margin)
        self.message_bus = message_bus
        self.available_slots = 3
        self.active_offers: Dict[str, float] = {}
        self.has_deal = False
        
        # LLM for competitive decisions
        self.llm = OllamaChatCompletionClient(model="llama3.1:8b")
        
        # Subscribe to messages
        message_bus.subscribe(station_id, self.handle_message)
        
        print(f"[{self.station_id}] ⚡ Ready - Cost: ${electricity_cost:.3f}, Min price: ${self.min_price:.3f}")
    
    async def make_immediate_offer(self):
        """Make competitive initial offer immediately"""
        if self.has_deal:
            return
            
        # Smart pricing decision
        initial_price = await self._determine_initial_price()
        offer_id = f"{self.station_id}_initial"
        
        print(f"[{self.station_id}] 📢 IMMEDIATE OFFER: ${initial_price:.3f}/kWh")
        
        # Create and broadcast offer to ALL EVs
        offer_data = {
            "station_id": self.station_id,
            "price_per_kwh": initial_price,
            "available_slots": self.available_slots,
            "offer_id": offer_id
        }
        
        self.active_offers[offer_id] = initial_price
        
        # Send to all EV agents (broadcast)
        for ev_id in ["EV1", "EV2", "EV3"]:  # Known EV agents
            offer_msg = Message(
                type=MessageType.OFFER,
                from_agent=self.station_id,
                to_agent=ev_id,
                data=offer_data,
                timestamp=asyncio.get_event_loop().time(),
                message_id=f"{self.station_id}_offer_{ev_id}"
            )
            
            await self.message_bus.publish(offer_msg)
    
    async def _determine_initial_price(self) -> float:
        """Use LLM to determine competitive initial pricing"""
        prompt = f"""You are {self.station_id}, a charging station entering a competitive marketplace.

YOUR COSTS:
- Electricity cost: ${self.electricity_cost:.3f}/kWh
- Required minimum margin: {self.min_margin*100:.0f}%
- Absolute minimum price: ${self.min_price:.3f}/kWh

STRATEGY: Set an initial competitive price that:
1. Attracts customers
2. Leaves room for negotiation
3. Ensures profitability
4. Positions competitively in market

RESPOND WITH VALID JSON ONLY:
{{"price": 0.125, "reasoning": "Brief explanation of pricing strategy"}}

JSON RESPONSE:"""

        try:
            response = await self.llm.create(
                messages=[UserMessage(content=prompt, source="user")]
            )
            
            decision_text = response.content
            if isinstance(decision_text, str):
                decision = json.loads(decision_text)
            else:
                decision = {"price": self.min_price * 1.15, "reasoning": "Fallback pricing"}
            
            price = max(decision.get("price", self.min_price * 1.15), self.min_price)
            print(f"[{self.station_id}] 💭 Pricing strategy: {decision.get('reasoning', 'N/A')}")
            return price
            
        except Exception as e:
            print(f"[{self.station_id}] ⚠️ LLM pricing error: {e}")
            return self.min_price * 1.15  # Safe fallback
    
    async def handle_message(self, message: Message):
        """Handle incoming messages from EVs"""
        # Check if station already has a deal using global market state
        if self.station_id in market_state.station_deals:
            print(f"[{self.station_id}] ⏭️ Ignoring {message.type.value} from {message.from_agent} - already have deal with {market_state.station_deals[self.station_id]}")
            return
            
        if message.type == MessageType.COUNTER_OFFER:
            await self._handle_counter_offer(message)
        elif message.type == MessageType.ACCEPTANCE:
            await self._handle_acceptance(message)
    
    async def _handle_counter_offer(self, message: Message):
        """Handle counter-offers from EVs"""
        counter_data = message.data
        counter_price = counter_data["counter_price"]
        
        print(f"[{self.station_id}] 🔍 Counter-offer from {message.from_agent}: ${counter_price:.3f}/kWh")
        
        # Smart decision on counter-offer
        decision = await self._evaluate_counter_offer(counter_price, message.from_agent)
        await self._respond_to_counter(decision, counter_data, message.from_agent)
    
    async def _evaluate_counter_offer(self, counter_price: float, ev_id: str) -> Dict:
        """Evaluate counter-offer using LLM"""
        profit = counter_price - self.electricity_cost
        margin = (profit / self.electricity_cost) * 100 if self.electricity_cost > 0 else 0
        
        prompt = f"""You are {self.station_id}. EV {ev_id} has made a counter-offer.

FINANCIAL ANALYSIS:
- Your electricity cost: ${self.electricity_cost:.3f}/kWh
- Minimum profitable price: ${self.min_price:.3f}/kWh
- Counter-offer: ${counter_price:.3f}/kWh
- Profit at counter-price: ${profit:.3f}/kWh ({margin:.1f}% margin)

DECISION OPTIONS:
1. ACCEPT - Take the counter-offer (if profitable)
2. COUNTER - Make a new competitive offer
3. REJECT - Decline if unprofitable

RESPOND WITH VALID JSON ONLY:
{{"action": "ACCEPT", "reasoning": "Why accepting"}}
{{"action": "COUNTER", "price": 0.120, "reasoning": "Why counter-offering"}}
{{"action": "REJECT", "reasoning": "Why rejecting"}}

JSON RESPONSE:"""

        try:
            response = await self.llm.create(
                messages=[UserMessage(content=prompt, source="user")]
            )
            
            decision_text = response.content
            if isinstance(decision_text, str):
                decision = json.loads(decision_text)
            else:
                decision = {"action": "REJECT", "reasoning": "Could not parse"}
            
            print(f"[{self.station_id}] 💭 Counter-offer decision: {decision['action']} - {decision['reasoning']}")
            return decision
            
        except Exception as e:
            print(f"[{self.station_id}] ⚠️ LLM error: {e}")
            # Fallback logic
            if counter_price >= self.min_price:
                return {"action": "ACCEPT", "reasoning": "Profitable (fallback)"}
            else:
                return {"action": "REJECT", "reasoning": "Below minimum (fallback)"}
    
    async def _respond_to_counter(self, decision: Dict, counter_data: Dict, ev_id: str):
        """Respond to counter-offer based on LLM decision"""
        action = decision.get("action", "REJECT")
        
        if action == "ACCEPT":
            counter_price = counter_data["counter_price"]
            print(f"[{self.station_id}] ✅ ACCEPTING counter-offer from {ev_id}: ${counter_price:.3f}/kWh")
            
            # Send acceptance
            acceptance_msg = Message(
                type=MessageType.ACCEPTANCE,
                from_agent=self.station_id,
                to_agent=ev_id,
                data={
                    "counter_id": counter_data["counter_id"],
                    "accepted_price": counter_price
                },
                timestamp=asyncio.get_event_loop().time(),
                message_id=f"{self.station_id}_accept_{counter_data['counter_id']}"
            )
            
            await self.message_bus.publish(acceptance_msg)
            self.has_deal = True
            
        elif action == "COUNTER":
            new_price = decision.get("price", max(self.min_price, counter_data["counter_price"] * 1.05))
            new_price = max(new_price, self.min_price)
            
            print(f"[{self.station_id}] 🔄 NEW OFFER to {ev_id}: ${new_price:.3f}/kWh")
            
            # Send new offer
            new_offer_id = f"{self.station_id}_counter_{ev_id}"
            offer_msg = Message(
                type=MessageType.OFFER,
                from_agent=self.station_id,
                to_agent=ev_id,
                data={
                    "station_id": self.station_id,
                    "price_per_kwh": new_price,
                    "available_slots": self.available_slots,
                    "offer_id": new_offer_id
                },
                timestamp=asyncio.get_event_loop().time(),
                message_id=f"{self.station_id}_new_offer_{ev_id}"
            )
            
            await self.message_bus.publish(offer_msg)
            
        else:  # REJECT
            print(f"[{self.station_id}] ❌ REJECTING counter-offer from {ev_id}")
    
    async def _handle_acceptance(self, message: Message):
        """Handle acceptance from EV"""
        # Double-check global market deal status
        if self.station_id in market_state.station_deals:
            print(f"[{self.station_id}] ⏭️ Acceptance from {message.from_agent} arrived too late - already have deal with {market_state.station_deals[self.station_id]}")
            return
            
        acceptance_data = message.data
        price = acceptance_data["accepted_price"]
        
        # The deal should already be registered by the EV agent
        self.has_deal = True
        print(f"[{self.station_id}] 🎉 DEAL ACCEPTED by {message.from_agent} at ${price:.3f}/kWh")

# =====================================================================================
# MARKETPLACE COORDINATOR - CLEAN & SIMPLE
# =====================================================================================

class MarketplaceCoordinator:
    """Clean coordinator that tracks deals and announces completions"""
    
    def __init__(self, message_bus: MessageBus):
        self.message_bus = message_bus
        self.completed_deals: List[Deal] = []
        
        # Subscribe to all messages for coordination
        message_bus.subscribe("coordinator", self.handle_message)
        
        print("[Marketplace] 🏪 Coordinator ready")
    
    async def handle_message(self, message: Message):
        """Track all marketplace activity"""
        if message.type == MessageType.ACCEPTANCE:
            await self._finalize_deal(message)
    
    async def _finalize_deal(self, message: Message):
        """Finalize deal when acceptance occurs"""
        acceptance_data = message.data
        accepted_price = acceptance_data["accepted_price"]
        
        # Determine who's who
        if message.from_agent.startswith("EV"):
            ev_id = message.from_agent
            station_id = message.to_agent
        else:
            ev_id = message.to_agent  
            station_id = message.from_agent
        
        # Create deal record
        energy_kwh = 35.0  # Standard amount
        total_cost = accepted_price * energy_kwh
        
        deal = Deal(
            ev_id=ev_id,
            station_id=station_id,
            price_per_kwh=accepted_price,
            energy_kwh=energy_kwh,
            total_cost=total_cost
        )
        
        self.completed_deals.append(deal)
        
        print(f"[Marketplace] 🤝 DEAL FINALIZED: {ev_id} ↔ {station_id} @ ${accepted_price:.3f}/kWh (${total_cost:.2f})")
        
        # Notify both parties
        deal_msg = Message(
            type=MessageType.DEAL_COMPLETE,
            from_agent="coordinator",
            to_agent=ev_id,
            data=asdict(deal),
            timestamp=asyncio.get_event_loop().time(),
            message_id=f"deal_complete_{len(self.completed_deals)}"
        )
        
        await self.message_bus.publish(deal_msg)

# =====================================================================================
# MAIN CLEAN SIMULATION
# =====================================================================================

async def main():
    """Run the clean EV marketplace simulation"""
    print("🔋 CLEAN EV CHARGING MARKETPLACE")
    print("=" * 50)
    print("✨ No jank, just clean async negotiation")
    print()
    
    # Create clean message bus
    message_bus = MessageBus()
    
    # Create coordinator
    coordinator = MarketplaceCoordinator(message_bus)
    
    # Create EV agents
    ev_agents = [
        EVAgent("EV1", max_budget=0.150, energy_needed=36.0, message_bus=message_bus),
        EVAgent("EV2", max_budget=0.160, energy_needed=42.0, message_bus=message_bus),
        EVAgent("EV3", max_budget=0.140, energy_needed=30.0, message_bus=message_bus)
    ]
    
    # Create charging station agents
    cs_agents = [
        ChargingStationAgent("CSA", electricity_cost=0.080, min_margin=0.20, message_bus=message_bus),
        ChargingStationAgent("CSB", electricity_cost=0.090, min_margin=0.15, message_bus=message_bus),
        ChargingStationAgent("CSC", electricity_cost=0.075, min_margin=0.25, message_bus=message_bus)
    ]
    
    print(f"✅ Created {len(ev_agents)} EV agents and {len(cs_agents)} charging stations")
    print()
    
    print("🚀 STARTING IMMEDIATE NEGOTIATIONS...")
    print("=" * 40)
    
    # Charging stations make IMMEDIATE offers
    for cs in cs_agents:
        await cs.make_immediate_offer()
        await asyncio.sleep(0.2)  # Small delay for readability
    
    print("\n⚡ NEGOTIATIONS IN PROGRESS...")
    
    # Let negotiations play out
    await asyncio.sleep(8)  # Give time for all negotiations
    
    print(f"\n🎯 MARKETPLACE COMPLETE")
    print("=" * 30)
    print(f"✅ Deals completed: {len(coordinator.completed_deals)}")
    
    for i, deal in enumerate(coordinator.completed_deals, 1):
        print(f"  {i}. {deal.ev_id} ↔ {deal.station_id}: ${deal.total_cost:.2f}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n🛑 Marketplace interrupted")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc() 