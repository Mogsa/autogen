#!/usr/bin/env python3
"""
EV Charging Marketplace using AutoGen-Core
A pure event-driven multi-agent negotiation system where EV owners and 
charging stations negotiate through explicit offer/counter-offer messages.

This implementation uses autogen-core's foundational event system rather than
the conversational patterns of autogen-agentchat.
"""

import asyncio
import json
from dataclasses import dataclass
from typing import Dict, Optional
from autogen_core import (
    RoutedAgent, 
    SingleThreadedAgentRuntime, 
    message_handler, 
    DefaultTopicId, 
    MessageContext, 
    type_subscription
)
from autogen_core.models import UserMessage
from autogen_ext.models.ollama import OllamaChatCompletionClient

# =====================================================================================
# MESSAGE DEFINITIONS - Explicit negotiation protocol
# =====================================================================================

@dataclass
class ChargingOffer:
    """Charging station makes an offer"""
    station_id: str
    price_per_kwh: float
    available_slots: int
    offer_id: str

@dataclass 
class CounterOffer:
    """EV owner makes a counter-offer"""
    ev_id: str
    target_station: str
    counter_price: float
    offer_id: str

@dataclass
class OfferAcceptance:
    """Either party accepts an offer"""
    acceptor_id: str
    accepted_from: str
    final_price: float
    offer_id: str

@dataclass
class DealCompleted:
    """Marketplace announces completed deal"""
    ev_id: str
    station_id: str
    final_price: float
    energy_kwh: float

@dataclass
class MarketUpdate:
    """General market status updates"""
    message: str
    active_offers: int
    completed_deals: int

# =====================================================================================
# EV AGENT - Seeks lowest charging price
# =====================================================================================

@type_subscription("marketplace")
class EVAgent(RoutedAgent):
    """Electric Vehicle agent that uses LLM reasoning for negotiation decisions"""
    
    def __init__(self, ev_id: str, max_budget: float, energy_needed: float):
        super().__init__(f"EV Owner {ev_id}")
        self.ev_id = ev_id
        self.max_budget = max_budget
        self.energy_needed = energy_needed
        self.received_offers: Dict[str, ChargingOffer] = {}
        self.counter_offers_made = 0
        self.has_deal = False
        self.max_counter_offers = 2
        
        # Initialize Ollama client for LLM reasoning
        self.llm_client = OllamaChatCompletionClient(model="llama3.1:8b")
        
        print(f"[{self.ev_id}] 🚗 Initialized - Budget: ${max_budget:.3f}/kWh, Need: {energy_needed:.1f} kWh")
        print(f"[{self.ev_id}] 🔍 DEBUG: Agent registered with subscription 'marketplace'")
    
    @message_handler
    async def handle_charging_offer(self, message: ChargingOffer, ctx: MessageContext) -> None:
        """React to charging station offers using LLM reasoning"""
        print(f"[{self.ev_id}] 🔔 MESSAGE HANDLER TRIGGERED - Message type: {type(message).__name__}")
        await self._handle_charging_offer(message, ctx)
    
    @message_handler
    async def handle_deal_completed(self, message: DealCompleted, ctx: MessageContext) -> None:
        """Acknowledge completed deals"""
        print(f"[{self.ev_id}] 🔔 MESSAGE HANDLER TRIGGERED - Message type: {type(message).__name__}")
        await self._handle_deal_completed(message, ctx)
    
    async def _handle_charging_offer(self, message: ChargingOffer, ctx: MessageContext) -> None:
        """React to charging station offers using LLM reasoning"""
        print(f"[{self.ev_id}] 📩 Received offer from {message.station_id}: ${message.price_per_kwh:.3f}/kWh")
        if self.has_deal:
            print(f"[{self.ev_id}] ⏭️ Already have deal, skipping offer")
            return
            
        print(f"[{self.ev_id}] 📩 Processing offer from {message.station_id}: ${message.price_per_kwh:.3f}/kWh")
        self.received_offers[message.station_id] = message
        
        # Use LLM to make negotiation decision
        decision = await self._llm_negotiate_decision(message)
        await self._execute_decision(decision, message)
    
    async def _llm_negotiate_decision(self, offer: ChargingOffer) -> Dict:
        """Use Llama 3.1 8B to make negotiation decisions"""
        
        # Create context for LLM
        market_context = {
            "my_budget": self.max_budget,
            "energy_needed": self.energy_needed,
            "current_offer": {
                "station": offer.station_id,
                "price": offer.price_per_kwh
            },
            "previous_offers": {k: v.price_per_kwh for k, v in self.received_offers.items()},
            "counter_offers_made": self.counter_offers_made,
            "max_counter_offers": self.max_counter_offers
        }
        
        prompt = f"""You are {self.ev_id}, an electric vehicle owner negotiating for charging services.

MARKET SITUATION:
- Your maximum budget: ${self.max_budget:.3f}/kWh
- Energy needed: {self.energy_needed:.1f} kWh
- Counter-offers made so far: {self.counter_offers_made}/{self.max_counter_offers}

CURRENT OFFER:
- Station: {offer.station_id}
- Price: ${offer.price_per_kwh:.3f}/kWh

PREVIOUS OFFERS RECEIVED:
{json.dumps({k: f"${v.price_per_kwh:.3f}/kWh" for k, v in self.received_offers.items()}, indent=2)}

NEGOTIATION STRATEGY:
You want the lowest possible price but must be realistic. Consider:
1. Is this offer within your budget?
2. How does it compare to other offers?
3. Do you have counter-offers remaining?
4. Should you accept, counter-offer, or wait for better offers?

RESPOND WITH VALID JSON ONLY:
For acceptance: {{"action": "ACCEPT", "reasoning": "Brief explanation why accepting"}}
For counter-offer: {{"action": "COUNTER", "price": 0.130, "reasoning": "Brief explanation for counter price"}}
For waiting: {{"action": "WAIT", "reasoning": "Brief explanation why waiting"}}

IMPORTANT: Response must be valid JSON only, no other text."""

        try:
            # Get LLM decision
            response = await self.llm_client.create(
                messages=[UserMessage(content=prompt, source="user")]
            )
            
            llm_response = response.content
            print(f"[{self.ev_id}] 🧠 LLM thinking: {llm_response[:100]}...")
            
            # Parse LLM response - handle both string and list responses
            if isinstance(llm_response, str):
                decision = json.loads(llm_response)
            else:
                # Fallback if response is not a string
                decision = {"action": "WAIT", "reasoning": "Could not parse LLM response"}
            print(f"[{self.ev_id}] 💭 Decision: {decision['action']} - {decision['reasoning']}")
            
            return decision
            
        except Exception as e:
            print(f"[{self.ev_id}] ⚠️ LLM error: {e}, falling back to simple logic")
            # Fallback to simple decision
            if offer.price_per_kwh <= self.max_budget:
                return {"action": "ACCEPT", "reasoning": "Within budget (fallback logic)"}
            else:
                return {"action": "COUNTER", "price": self.max_budget * 0.95, "reasoning": "Counter-offer (fallback logic)"}
    
    async def _execute_decision(self, decision: Dict, offer: ChargingOffer) -> None:
        """Execute the LLM's negotiation decision"""
        action = decision.get("action", "WAIT")
        
        if action == "ACCEPT":
            print(f"[{self.ev_id}] ✅ Accepting {offer.station_id} offer: ${offer.price_per_kwh:.3f}/kWh")
            acceptance = OfferAcceptance(
                acceptor_id=self.ev_id,
                accepted_from=offer.station_id,
                final_price=offer.price_per_kwh,
                offer_id=offer.offer_id
            )
            await self.publish_message(acceptance, DefaultTopicId("marketplace"))
            self.has_deal = True
            
        elif action == "COUNTER" and self.counter_offers_made < self.max_counter_offers:
            counter_price = decision.get("price", self.max_budget * 0.9)
            print(f"[{self.ev_id}] 🔄 Counter-offering to {offer.station_id}: ${counter_price:.3f}/kWh")
            
            counter = CounterOffer(
                ev_id=self.ev_id,
                target_station=offer.station_id,
                counter_price=counter_price,
                offer_id=offer.offer_id
            )
            await self.publish_message(counter, DefaultTopicId("marketplace"))
            self.counter_offers_made += 1
            
        elif action == "WAIT":
            print(f"[{self.ev_id}] ⏳ Waiting for better offers...")
            
        else:
            # Out of options, accept best available
            if self.received_offers:
                best_offer = min(self.received_offers.values(), key=lambda x: x.price_per_kwh)
                print(f"[{self.ev_id}] 🏁 Final choice: accepting {best_offer.station_id} at ${best_offer.price_per_kwh:.3f}/kWh")
                
                acceptance = OfferAcceptance(
                    acceptor_id=self.ev_id,
                    accepted_from=best_offer.station_id,
                    final_price=best_offer.price_per_kwh,
                    offer_id=best_offer.offer_id
                )
                await self.publish_message(acceptance, DefaultTopicId("marketplace"))
                self.has_deal = True
    
    async def _handle_deal_completed(self, message: DealCompleted, ctx: MessageContext) -> None:
        """Acknowledge completed deals"""
        if message.ev_id == self.ev_id:
            total_cost = message.final_price * self.energy_needed
            print(f"[{self.ev_id}] 🎉 Deal finalized! Total cost: ${total_cost:.2f}")

# =====================================================================================
# CHARGING STATION AGENT - Maximizes profit
# =====================================================================================

@type_subscription("marketplace")
class ChargingStationAgent(RoutedAgent):
    """Charging Station agent that uses LLM reasoning for competitive pricing"""
    
    def __init__(self, station_id: str, electricity_cost: float, min_margin: float):
        super().__init__(f"Charging Station {station_id}")
        self.station_id = station_id
        self.electricity_cost = electricity_cost
        self.min_margin = min_margin
        self.min_price = electricity_cost * (1 + min_margin)
        self.available_slots = 3
        self.active_offers: Dict[str, float] = {}
        self.has_deal = False
        
        # Initialize Ollama client for LLM reasoning
        self.llm_client = OllamaChatCompletionClient(model="llama3.1:8b")
        
        print(f"[{self.station_id}] ⚡ Initialized - Cost: ${electricity_cost:.3f}, Min price: ${self.min_price:.3f}")
    
    async def make_initial_offer(self) -> None:
        """Make competitive initial offer using LLM reasoning"""
        if self.has_deal or not self.available_slots:
            return
            
        # Use LLM to determine initial pricing strategy
        initial_price = await self._llm_initial_pricing()
        offer_id = f"{self.station_id}_initial"
        
        print(f"[{self.station_id}] 📢 Initial offer: ${initial_price:.3f}/kWh")
        
        offer = ChargingOffer(
            station_id=self.station_id,
            price_per_kwh=initial_price,
            available_slots=self.available_slots,
            offer_id=offer_id
        )
        
        await self.publish_message(offer, DefaultTopicId("marketplace"))
        self.active_offers[offer_id] = initial_price
    
    async def _llm_initial_pricing(self) -> float:
        """Use LLM to determine initial pricing strategy"""
        prompt = f"""You are {self.station_id}, a charging station operator entering a competitive marketplace.

YOUR COSTS & CONSTRAINTS:
- Electricity cost: ${self.electricity_cost:.3f}/kWh
- Required minimum profit margin: {self.min_margin*100:.0f}%
- Absolute minimum price: ${self.min_price:.3f}/kWh (you CANNOT go below this)

BUSINESS STRATEGY:
You're competing against other charging stations for EV customers. You need to:
1. Price competitively to attract customers
2. Leave room for negotiation
3. Ensure profitability
4. Consider market positioning

PRICING DECISION:
What should your initial offer price be? Consider:
- Start higher than minimum to allow negotiation room
- Be competitive but profitable
- Position for market entry

RESPOND WITH VALID JSON ONLY:
{{"price": 0.125, "reasoning": "Brief explanation of pricing strategy"}}

IMPORTANT: Price must be >= ${self.min_price:.3f}. Response must be valid JSON only."""

        try:
            response = await self.llm_client.create(
                messages=[UserMessage(content=prompt, source="user")]
            )
            
            llm_response = response.content
            print(f"[{self.station_id}] 🧠 LLM pricing strategy: {llm_response[:100]}...")
            
            # Parse LLM response - handle both string and list responses
            if isinstance(llm_response, str):
                decision = json.loads(llm_response)
            else:
                # Fallback if response is not a string
                decision = {"price": self.min_price * 1.15, "reasoning": "Could not parse LLM response"}
            proposed_price = decision.get("price", self.min_price * 1.15)
            
            # Ensure price is above minimum
            final_price = max(proposed_price, self.min_price)
            print(f"[{self.station_id}] 💭 Pricing reasoning: {decision.get('reasoning', 'N/A')}")
            
            return final_price
            
        except Exception as e:
            print(f"[{self.station_id}] ⚠️ LLM pricing error: {e}, using fallback")
            return self.min_price * 1.15  # 15% above minimum as fallback
    
    @message_handler
    async def handle_counter_offer(self, message: CounterOffer, ctx: MessageContext) -> None:
        """Evaluate EV counter-offers using LLM reasoning"""
        print(f"[{self.station_id}] 🔔 MESSAGE HANDLER TRIGGERED - Message type: {type(message).__name__}")
        await self._handle_counter_offer(message, ctx)
    
    @message_handler
    async def handle_deal_completed(self, message: DealCompleted, ctx: MessageContext) -> None:
        """Acknowledge completed deals"""
        print(f"[{self.station_id}] 🔔 MESSAGE HANDLER TRIGGERED - Message type: {type(message).__name__}")
        await self._handle_deal_completed(message, ctx)
    
    async def _handle_counter_offer(self, message: CounterOffer, ctx: MessageContext) -> None:
        """Evaluate EV counter-offers using LLM reasoning"""
        if message.target_station != self.station_id or self.has_deal:
            return
            
        print(f"[{self.station_id}] 🔍 Counter-offer from {message.ev_id}: ${message.counter_price:.3f}/kWh")
        
        # Use LLM to decide how to respond to counter-offer
        decision = await self._llm_counter_offer_response(message)
        await self._execute_counter_decision(decision, message)
    
    async def _llm_counter_offer_response(self, counter_offer: CounterOffer) -> Dict:
        """Use LLM to decide response to counter-offers"""
        
        profit_at_counter = counter_offer.counter_price - self.electricity_cost
        profit_margin = (profit_at_counter / self.electricity_cost) * 100 if self.electricity_cost > 0 else 0
        
        prompt = f"""You are {self.station_id}, a charging station operator. An EV customer has made a counter-offer.

FINANCIAL SITUATION:
- Your electricity cost: ${self.electricity_cost:.3f}/kWh
- Your minimum price: ${self.min_price:.3f}/kWh (below this you lose money)
- Current active offers: {list(self.active_offers.values())}

COUNTER-OFFER RECEIVED:
- From: {counter_offer.ev_id}
- Offered price: ${counter_offer.counter_price:.3f}/kWh
- Profit at this price: ${profit_at_counter:.3f}/kWh ({profit_margin:.1f}% margin)

BUSINESS DECISION:
You must decide whether to:
1. ACCEPT - Take this counter-offer (if profitable)
2. COUNTER - Make a new competitive offer 
3. REJECT - Decline if unprofitable

Consider:
- Is the offer profitable (above ${self.min_price:.3f})?
- Market competition and customer retention
- Revenue vs. profit optimization

RESPOND WITH VALID JSON ONLY:
For acceptance: {{"action": "ACCEPT", "reasoning": "Brief explanation"}}
For new offer: {{"action": "COUNTER", "price": 0.115, "reasoning": "Brief explanation"}}
For rejection: {{"action": "REJECT", "reasoning": "Brief explanation"}}

IMPORTANT: Any price must be >= ${self.min_price:.3f}. Response must be valid JSON only."""

        try:
            response = await self.llm_client.create(
                messages=[UserMessage(content=prompt, source="user")]
            )
            
            llm_response = response.content
            print(f"[{self.station_id}] 🧠 LLM counter-offer analysis: {llm_response[:100]}...")
            
            # Parse LLM response - handle both string and list responses
            if isinstance(llm_response, str):
                decision = json.loads(llm_response)
            else:
                # Fallback if response is not a string
                decision = {"action": "REJECT", "reasoning": "Could not parse LLM response"}
            print(f"[{self.station_id}] 💭 Decision: {decision['action']} - {decision['reasoning']}")
            
            return decision
            
        except Exception as e:
            print(f"[{self.station_id}] ⚠️ LLM error: {e}, using fallback logic")
            # Fallback logic
            if counter_offer.counter_price >= self.min_price:
                return {"action": "ACCEPT", "reasoning": "Profitable counter-offer (fallback)"}
            else:
                return {"action": "REJECT", "reasoning": "Below minimum price (fallback)"}
    
    async def _execute_counter_decision(self, decision: Dict, counter_offer: CounterOffer) -> None:
        """Execute the LLM's counter-offer decision"""
        action = decision.get("action", "REJECT")
        
        if action == "ACCEPT":
            profit_margin = (counter_offer.counter_price - self.electricity_cost) / self.electricity_cost * 100
            print(f"[{self.station_id}] ✅ Accepting counter-offer (profit: {profit_margin:.1f}%)")
            
            acceptance = OfferAcceptance(
                acceptor_id=self.station_id,
                accepted_from=counter_offer.ev_id,
                final_price=counter_offer.counter_price,
                offer_id=counter_offer.offer_id
            )
            await self.publish_message(acceptance, DefaultTopicId("marketplace"))
            self.has_deal = True
            
        elif action == "COUNTER":
            new_price = decision.get("price", max(self.min_price, counter_offer.counter_price * 1.05))
            new_price = max(new_price, self.min_price)  # Ensure above minimum
            
            offer_id = f"{self.station_id}_counter_{counter_offer.ev_id}"
            print(f"[{self.station_id}] 🔄 New competitive offer: ${new_price:.3f}/kWh")
            
            new_offer = ChargingOffer(
                station_id=self.station_id,
                price_per_kwh=new_price,
                available_slots=self.available_slots,
                offer_id=offer_id
            )
            await self.publish_message(new_offer, DefaultTopicId("marketplace"))
            self.active_offers[offer_id] = new_price
            
        else:  # REJECT
            print(f"[{self.station_id}] ❌ Rejecting counter-offer: {decision.get('reasoning', 'Unprofitable')}")
    
    async def _handle_deal_completed(self, message: DealCompleted, ctx: MessageContext) -> None:
        """Acknowledge completed deals"""
        if message.station_id == self.station_id:
            profit = message.final_price - self.electricity_cost
            margin = profit / self.electricity_cost * 100
            revenue = message.final_price * message.energy_kwh
            print(f"[{self.station_id}] 💰 Deal completed! Revenue: ${revenue:.2f}, Margin: {margin:.1f}%")

# =====================================================================================
# MARKETPLACE COORDINATOR - Manages deal completion
# =====================================================================================

@type_subscription("marketplace")
class MarketplaceCoordinator(RoutedAgent):
    """Central coordinator that finalizes deals and tracks market state"""
    
    def __init__(self):
        super().__init__("Marketplace Coordinator")
        self.completed_deals = []
        self.active_negotiations = 0
        
        print("[Marketplace] 🏪 Coordinator initialized")
    
    @message_handler
    async def handle_offer_acceptance(self, message: OfferAcceptance, ctx: MessageContext) -> None:
        """Finalize deals when offers are accepted"""
        print(f"[Marketplace] 🔔 MESSAGE HANDLER TRIGGERED - Message type: {type(message).__name__}")
        await self._handle_offer_acceptance(message, ctx)
    
    async def _handle_offer_acceptance(self, message: OfferAcceptance, ctx: MessageContext) -> None:
        """Finalize deals when offers are accepted"""
        print(f"[Marketplace] 🤝 Deal accepted: {message.acceptor_id} ↔ {message.accepted_from} @ ${message.final_price:.3f}/kWh")
        
        # Determine EV and station (acceptance can come from either party)
        if message.acceptor_id.startswith("EV"):
            ev_id, station_id = message.acceptor_id, message.accepted_from
        else:
            ev_id, station_id = message.accepted_from, message.acceptor_id
        
        # Create deal completion record
        deal = DealCompleted(
            ev_id=ev_id,
            station_id=station_id,
            final_price=message.final_price,
            energy_kwh=35.0  # Simplified - could be dynamic
        )
        
        self.completed_deals.append(deal)
        await self.publish_message(deal, DefaultTopicId("marketplace"))
        
        # Check if all deals completed
        if len(self.completed_deals) >= 3:  # 3 EVs
            await self._announce_market_close()
    
    async def _announce_market_close(self) -> None:
        """Announce marketplace closure"""
        print(f"[Marketplace] 🏁 All deals completed! Market closing.")
        
        update = MarketUpdate(
            message="Marketplace session complete",
            active_offers=0,
            completed_deals=len(self.completed_deals)
        )
        await self.publish_message(update, DefaultTopicId("marketplace"))

# =====================================================================================
# MAIN SIMULATION
# =====================================================================================

async def main():
    """Execute the EV charging marketplace simulation"""
    print("🔋 EV Charging Marketplace - AutoGen Core Implementation")
    print("=" * 70)
    print("Event-driven multi-agent negotiation system")
    print()
    
    # Initialize runtime
    runtime = SingleThreadedAgentRuntime()
    
    # Create marketplace coordinator
    coordinator = MarketplaceCoordinator()
    await MarketplaceCoordinator.register(runtime, "coordinator", lambda: coordinator)
    
    # Create EV agents with different budgets
    ev_agents = [
        EVAgent("EV1", max_budget=0.150, energy_needed=36.0),
        EVAgent("EV2", max_budget=0.160, energy_needed=42.0), 
        EVAgent("EV3", max_budget=0.140, energy_needed=30.0)
    ]
    
    for ev in ev_agents:
        await EVAgent.register(runtime, ev.ev_id, lambda agent=ev: agent)
    
    # Create charging station agents with different cost structures
    cs_agents = [
        ChargingStationAgent("CSA", electricity_cost=0.080, min_margin=0.20),
        ChargingStationAgent("CSB", electricity_cost=0.090, min_margin=0.15),
        ChargingStationAgent("CSC", electricity_cost=0.075, min_margin=0.25)
    ]
    
    for cs in cs_agents:
        await ChargingStationAgent.register(runtime, cs.station_id, lambda agent=cs: agent)
    
    print(f"🤖 Registered {len(ev_agents)} EV agents and {len(cs_agents)} charging stations")
    print()
    
    print("🚀 Starting marketplace negotiations...")
    print("=" * 50)
    
    # Start the runtime FIRST
    runtime.start()
    
    print("\n🔍 DEBUG: Runtime started, now publishing initial offers...")
    
    # Charging stations make initial offers (AFTER starting runtime)
    for i, cs in enumerate(cs_agents):
        # Create initial offer and publish via runtime
        initial_price = cs.min_price * 1.15  # Simple fallback pricing
        offer_id = f"{cs.station_id}_initial"
        
        print(f"[{cs.station_id}] 📢 Initial offer: ${initial_price:.3f}/kWh")
        
        offer = ChargingOffer(
            station_id=cs.station_id,
            price_per_kwh=initial_price,
            available_slots=cs.available_slots,
            offer_id=offer_id
        )
        
        # Publish through runtime AFTER starting
        print(f"🔍 DEBUG: Publishing ChargingOffer to topic 'marketplace'")
        await runtime.publish_message(offer, DefaultTopicId("marketplace"))
        cs.active_offers[offer_id] = initial_price
        print(f"🔍 DEBUG: Message published for {cs.station_id}")
        
        # Small delay to allow processing
        await asyncio.sleep(0.5)
    
    print("\n🔍 DEBUG: Runtime started, waiting for message processing...")
    
    # Let the negotiation play out
    await asyncio.sleep(10)  # Give more time for all negotiations to complete
    
    print("🔍 DEBUG: Sleep period complete, checking for any activity...")
    
    # Stop when idle (all messages processed)
    await runtime.stop_when_idle()
    
    print()
    print("=" * 70)
    print("🎯 MARKETPLACE SIMULATION COMPLETE")
    print("=" * 70)
    print("✅ Event-driven negotiation completed")
    print("🔧 Built with AutoGen-Core foundation")

if __name__ == "__main__":
    """Entry point for the AutoGen-Core EV marketplace"""
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n🛑 Simulation interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Simulation error: {e}")
        import traceback
        traceback.print_exc()