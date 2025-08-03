#!/usr/bin/env python3
"""
EV Charging Marketplace Simulation

A complete multi-agent simulation using autogen-core where Electric Vehicle (EV) agents
negotiate with Charging Station (CS) agents through a central marketplace for optimal
charging prices. This demonstrates event-driven agent communication without LLMs.

Architecture:
- EVAgent: Seeks lowest charging prices
- CSAgent: Maximizes profit while staying competitive  
- MarketplaceAgent: Facilitates all communication and message routing

The simulation uses explicit negotiation logic rather than LLM-based decision making.
"""

import asyncio
import sys
from dataclasses import dataclass
from typing import List, Dict, Optional
from autogen_core import RoutedAgent, MessageContext, DefaultTopicId, default_subscription, message_handler, AgentId
from autogen_core import SingleThreadedAgentRuntime


# =====================================================================================
# MESSAGE DEFINITIONS
# =====================================================================================

@dataclass
class ChargingRequest:
    """Message sent by EV to request charging services"""
    ev_id: str
    battery_level: int  # Current battery percentage (0-100)
    target_battery_level: int  # Desired battery percentage (0-100) 
    max_acceptable_price: float  # Maximum price per kWh willing to pay


@dataclass
class ChargingOffer:
    """Message sent by CS to offer charging services"""
    cs_id: str
    ev_id: str
    price: float  # Price per kWh offered
    available_chargers: int  # Number of available charging ports


@dataclass
class CounterOffer:
    """Message sent by EV to counter a charging offer"""
    ev_id: str
    cs_id: str
    price: float  # Counter-offered price per kWh


@dataclass
class OfferAccepted:
    """Message sent by EV when accepting a charging offer"""
    ev_id: str
    cs_id: str
    final_price: float


@dataclass
class OfferRejected:
    """Message sent by EV/CS when rejecting an offer"""
    ev_id: str
    cs_id: str
    reason: str


@dataclass
class DealFinalized:
    """Message broadcast by marketplace when a deal is completed"""
    ev_id: str
    cs_id: str
    final_price: float
    energy_needed: float  # kWh


@dataclass
class DealFailed:
    """Message broadcast by marketplace when negotiation fails"""
    ev_id: str
    reason: str


# =====================================================================================
# EV AGENT - Seeks lowest charging price
# =====================================================================================

@default_subscription
class EVAgent(RoutedAgent):
    """
    Electric Vehicle Agent that negotiates for the best charging price.
    
    Strategy:
    1. Send charging request to marketplace
    2. Evaluate offers from charging stations
    3. Accept if price <= max_acceptable_price
    4. Counter-offer using midpoint strategy if price too high
    5. Accept best available offer or give up after one round
    """
    
    def __init__(self, ev_id: str, battery_level: int, target_battery_level: int, 
                 max_acceptable_price: float):
        super().__init__(description=f"EV-{ev_id}")
        self.ev_id = ev_id
        self.battery_level = battery_level
        self.target_battery_level = target_battery_level
        self.max_acceptable_price = max_acceptable_price
        self.energy_needed = self._calculate_energy_needed()
        self.received_offers: Dict[str, ChargingOffer] = {}
        self.negotiation_active = False
        self.deal_completed = False
        
    def _calculate_energy_needed(self) -> float:
        """Calculate kWh needed based on battery levels (simplified)"""
        # Assume average EV battery capacity of 60 kWh
        battery_capacity = 60.0
        percentage_needed = (self.target_battery_level - self.battery_level) / 100.0
        return battery_capacity * percentage_needed
    
    @message_handler
    async def handle_charging_offer(self, message: ChargingOffer, ctx: MessageContext) -> None:
        """Process charging offer from a charging station"""
        await self._handle_charging_offer(message, ctx)
    
    @message_handler
    async def handle_offer_rejected(self, message: OfferRejected, ctx: MessageContext) -> None:
        """Handle rejection of our counter-offer"""
        await self._handle_offer_rejected(message, ctx)
    
    @message_handler
    async def handle_deal_finalized(self, message: DealFinalized, ctx: MessageContext) -> None:
        """Handle successful deal completion"""
        await self._handle_deal_finalized(message, ctx)
    
    @message_handler
    async def handle_deal_failed(self, message: DealFailed, ctx: MessageContext) -> None:
        """Handle failed negotiation"""
        await self._handle_deal_failed(message, ctx)
    
    async def _handle_charging_offer(self, offer: ChargingOffer, ctx: MessageContext) -> None:
        """Process charging offer from a charging station"""
        if offer.ev_id != self.ev_id or self.deal_completed:
            return
            
        print(f"[{self.ev_id}] Received offer from {offer.cs_id}: ${offer.price:.3f}/kWh")
        self.received_offers[offer.cs_id] = offer
        
        # Decision logic: Accept if price is acceptable
        if offer.price <= self.max_acceptable_price:
            print(f"[{self.ev_id}] Accepting offer from {offer.cs_id} at ${offer.price:.3f}/kWh")
            await self.publish_message(
                OfferAccepted(self.ev_id, offer.cs_id, offer.price),
                DefaultTopicId()
            )
            self.deal_completed = True
        else:
            # Counter-offer using midpoint strategy
            counter_price = (offer.price + self.max_acceptable_price) / 2
            print(f"[{self.ev_id}] Counter-offering {offer.cs_id}: ${counter_price:.3f}/kWh (original: ${offer.price:.3f})")
            await self.publish_message(
                CounterOffer(self.ev_id, offer.cs_id, counter_price),
                DefaultTopicId()
            )
    
    async def _handle_offer_rejected(self, rejection: OfferRejected, ctx: MessageContext) -> None:
        """Handle rejection of our counter-offer"""
        if rejection.ev_id != self.ev_id:
            return
            
        print(f"[{self.ev_id}] Counter-offer rejected by {rejection.cs_id}: {rejection.reason}")
        
        # Remove this CS from consideration
        if rejection.cs_id in self.received_offers:
            del self.received_offers[rejection.cs_id]
        
        # If no more offers available, end negotiation
        if not self.received_offers:
            print(f"[{self.ev_id}] No acceptable offers available. Ending negotiation.")
            self.deal_completed = True
    
    async def _handle_deal_finalized(self, deal: DealFinalized, ctx: MessageContext) -> None:
        """Handle successful deal completion"""
        if deal.ev_id == self.ev_id:
            total_cost = deal.final_price * self.energy_needed
            print(f"[{self.ev_id}] ✅ Deal completed with {deal.cs_id}!")
            print(f"[{self.ev_id}]    Price: ${deal.final_price:.3f}/kWh")
            print(f"[{self.ev_id}]    Energy: {self.energy_needed:.1f} kWh")
            print(f"[{self.ev_id}]    Total cost: ${total_cost:.2f}")
    
    async def _handle_deal_failed(self, failure: DealFailed, ctx: MessageContext) -> None:
        """Handle failed negotiation"""
        if failure.ev_id == self.ev_id:
            print(f"[{self.ev_id}] ❌ Negotiation failed: {failure.reason}")
    


# =====================================================================================
# CHARGING STATION AGENT - Maximizes profit
# =====================================================================================

@default_subscription  
class CSAgent(RoutedAgent):
    """
    Charging Station Agent that aims to maximize profit while staying competitive.
    
    Strategy:
    1. Calculate initial offer based on cost + desired profit margin
    2. Accept counter-offers that still provide minimum profit
    3. Reject offers below cost basis
    """
    
    def __init__(self, cs_id: str, current_electricity_cost: float, 
                 available_chargers: int, min_profit_margin: float):
        super().__init__(description=f"CS-{cs_id}")
        self.cs_id = cs_id
        self.current_electricity_cost = current_electricity_cost
        self.available_chargers = available_chargers
        self.min_profit_margin = min_profit_margin
        self.active_negotiations: Dict[str, float] = {}  # ev_id -> offered_price
        
    def _calculate_initial_price(self) -> float:
        """Calculate initial offering price based on cost and desired margin"""
        return self.current_electricity_cost * (1 + self.min_profit_margin)
    
    def _is_profitable(self, price: float) -> bool:
        """Check if a price provides minimum acceptable profit"""
        return price > self.current_electricity_cost
    
    @message_handler
    async def handle_charging_request(self, message: ChargingRequest, ctx: MessageContext) -> None:
        """Respond to charging request with an offer"""
        await self._handle_charging_request(message, ctx)
    
    @message_handler
    async def handle_counter_offer(self, message: CounterOffer, ctx: MessageContext) -> None:
        """Evaluate and respond to counter-offer"""
        await self._handle_counter_offer(message, ctx)
    
    @message_handler
    async def handle_deal_finalized(self, message: DealFinalized, ctx: MessageContext) -> None:
        """Handle successful deal completion"""
        await self._handle_deal_finalized(message, ctx)
    
    async def _handle_charging_request(self, request: ChargingRequest, ctx: MessageContext) -> None:
        """Respond to charging request with an offer"""
        if self.available_chargers <= 0:
            print(f"[{self.cs_id}] No available chargers for {request.ev_id}")
            return
            
        initial_price = self._calculate_initial_price()
        self.active_negotiations[request.ev_id] = initial_price
        
        print(f"[{self.cs_id}] Offering {request.ev_id}: ${initial_price:.3f}/kWh")
        print(f"[{self.cs_id}]   Cost basis: ${self.current_electricity_cost:.3f}/kWh")
        print(f"[{self.cs_id}]   Target margin: {self.min_profit_margin*100:.1f}%")
        
        offer = ChargingOffer(
            self.cs_id,
            request.ev_id, 
            initial_price,
            self.available_chargers
        )
        
        await self.publish_message(offer, DefaultTopicId())
    
    async def _handle_counter_offer(self, counter: CounterOffer, ctx: MessageContext) -> None:
        """Evaluate and respond to counter-offer"""
        if counter.cs_id != self.cs_id or counter.ev_id not in self.active_negotiations:
            return
            
        original_offer = self.active_negotiations[counter.ev_id]
        profit_margin = (counter.price - self.current_electricity_cost) / self.current_electricity_cost
        
        print(f"[{self.cs_id}] Counter-offer from {counter.ev_id}: ${counter.price:.3f}/kWh")
        print(f"[{self.cs_id}]   Original offer: ${original_offer:.3f}/kWh")
        print(f"[{self.cs_id}]   Profit margin: {profit_margin*100:.1f}%")
        
        if self._is_profitable(counter.price):
            # Accept the counter-offer
            print(f"[{self.cs_id}] Accepting counter-offer from {counter.ev_id}")
            await self.publish_message(
                OfferAccepted(counter.ev_id, self.cs_id, counter.price),
                DefaultTopicId()
            )
        else:
            # Reject - not profitable enough
            reason = f"Price ${counter.price:.3f} below profitable threshold"
            print(f"[{self.cs_id}] Rejecting counter-offer from {counter.ev_id}: {reason}")
            await self.publish_message(
                OfferRejected(counter.ev_id, self.cs_id, reason),
                DefaultTopicId()
            )
            
        # Remove from active negotiations
        del self.active_negotiations[counter.ev_id]
    
    async def _handle_deal_finalized(self, deal: DealFinalized, ctx: MessageContext) -> None:
        """Handle successful deal completion"""
        if deal.cs_id == self.cs_id:
            profit = deal.final_price - self.current_electricity_cost
            margin = (profit / self.current_electricity_cost) * 100
            revenue = deal.final_price * deal.energy_needed
            
            print(f"[{self.cs_id}] ✅ Deal completed with {deal.ev_id}!")
            print(f"[{self.cs_id}]    Revenue: ${revenue:.2f}")
            print(f"[{self.cs_id}]    Profit margin: {margin:.1f}%")
            
            # Update available chargers
            self.available_chargers -= 1


# =====================================================================================
# MARKETPLACE AGENT - Central coordination hub
# =====================================================================================

@default_subscription
class MarketplaceAgent(RoutedAgent):
    """
    Central marketplace that facilitates communication between EVs and Charging Stations.
    
    Responsibilities:
    1. Route charging requests to all available charging stations
    2. Forward offers and counter-offers between agents
    3. Finalize deals and broadcast completion status
    4. Maintain transaction records
    """
    
    def __init__(self):
        super().__init__(description="marketplace")
        self.registered_evs: List[str] = []
        self.registered_css: List[str] = []
        self.completed_deals: List[Dict] = []
        self.failed_negotiations: List[Dict] = []
        
        # Hardcode the agent registrations for this simulation
        self.registered_evs = ["1", "2", "3"]
        self.registered_css = ["A", "B", "C"]
        print(f"[Marketplace] Auto-registered {len(self.registered_evs)} EVs and {len(self.registered_css)} charging stations")
        
    def register_ev(self, ev_id: str) -> None:
        """Register an EV agent with the marketplace"""
        if ev_id not in self.registered_evs:
            self.registered_evs.append(ev_id)
            print(f"[Marketplace] Registered EV: {ev_id}")
    
    def register_cs(self, cs_id: str) -> None:
        """Register a charging station with the marketplace"""
        if cs_id not in self.registered_css:
            self.registered_css.append(cs_id)
            print(f"[Marketplace] Registered CS: {cs_id}")
    
    @message_handler
    async def handle_charging_request(self, message: ChargingRequest, ctx: MessageContext) -> None:
        """Forward charging request to all registered charging stations"""
        await self._handle_charging_request(message, ctx)
    
    @message_handler
    async def handle_charging_offer(self, message: ChargingOffer, ctx: MessageContext) -> None:
        """Forward charging offer to the target EV"""
        await self._handle_charging_offer(message, ctx)
    
    @message_handler
    async def handle_counter_offer(self, message: CounterOffer, ctx: MessageContext) -> None:
        """Forward counter-offer to the target charging station"""
        await self._handle_counter_offer(message, ctx)
    
    @message_handler
    async def handle_offer_accepted(self, message: OfferAccepted, ctx: MessageContext) -> None:
        """Finalize the deal and broadcast completion"""
        await self._handle_offer_accepted(message, ctx)
    
    @message_handler
    async def handle_offer_rejected(self, message: OfferRejected, ctx: MessageContext) -> None:
        """Forward rejection and potentially end negotiation"""
        await self._handle_offer_rejected(message, ctx)
    
    async def _handle_charging_request(self, request: ChargingRequest, ctx: MessageContext) -> None:
        """Forward charging request to all registered charging stations"""
        print(f"[Marketplace] Broadcasting charging request from {request.ev_id} to {len(self.registered_css)} stations")
        
        for cs_id in self.registered_css:
            await self.publish_message(request, DefaultTopicId())
    
    async def _handle_charging_offer(self, offer: ChargingOffer, ctx: MessageContext) -> None:
        """Forward charging offer to the target EV"""
        print(f"[Marketplace] Forwarding offer of ${offer.price:.3f}/kWh from {offer.cs_id} to {offer.ev_id}")
        await self.publish_message(offer, DefaultTopicId())
    
    async def _handle_counter_offer(self, counter: CounterOffer, ctx: MessageContext) -> None:
        """Forward counter-offer to the target charging station"""
        print(f"[Marketplace] Forwarding counter-offer of ${counter.price:.3f}/kWh from {counter.ev_id} to {counter.cs_id}")
        await self.publish_message(counter, DefaultTopicId())
    
    async def _handle_offer_accepted(self, acceptance: OfferAccepted, ctx: MessageContext) -> None:
        """Finalize the deal and broadcast completion"""
        print(f"[Marketplace] Deal accepted: {acceptance.ev_id} ↔ {acceptance.cs_id} at ${acceptance.final_price:.3f}/kWh")
        
        # Calculate energy for the deal (simplified - using 30 kWh average)
        energy_needed = 30.0  # This should ideally come from the original request
        
        deal = DealFinalized(
            acceptance.ev_id,
            acceptance.cs_id, 
            acceptance.final_price,
            energy_needed
        )
        
        # Broadcast to all agents
        for ev_id in self.registered_evs:
            await self.publish_message(deal, DefaultTopicId())
        for cs_id in self.registered_css:
            await self.publish_message(deal, DefaultTopicId())
        
        # Record the deal
        self.completed_deals.append({
            'ev_id': acceptance.ev_id,
            'cs_id': acceptance.cs_id,
            'price': acceptance.final_price,
            'energy': energy_needed,
            'total_cost': acceptance.final_price * energy_needed
        })
    
    async def _handle_offer_rejected(self, rejection: OfferRejected, ctx: MessageContext) -> None:
        """Forward rejection and potentially end negotiation"""
        print(f"[Marketplace] Offer rejected: {rejection.ev_id} ↔ {rejection.cs_id}")
        
        # Forward rejection to EV
        await self.publish_message(rejection, DefaultTopicId())
    
    def print_summary(self) -> None:
        """Print final marketplace summary"""
        print("\n" + "="*80)
        print("MARKETPLACE SUMMARY")
        print("="*80)
        
        print(f"\nRegistered Agents:")
        print(f"  EVs: {len(self.registered_evs)} ({', '.join(self.registered_evs)})")
        print(f"  Charging Stations: {len(self.registered_css)} ({', '.join(self.registered_css)})")
        
        print(f"\nCompleted Deals: {len(self.completed_deals)}")
        for deal in self.completed_deals:
            print(f"  {deal['ev_id']} ← {deal['cs_id']}: ${deal['price']:.3f}/kWh "
                  f"× {deal['energy']:.1f} kWh = ${deal['total_cost']:.2f}")
        
        if self.completed_deals:
            avg_price = sum(deal['price'] for deal in self.completed_deals) / len(self.completed_deals)
            total_revenue = sum(deal['total_cost'] for deal in self.completed_deals)
            print(f"\nMarket Statistics:")
            print(f"  Average price: ${avg_price:.3f}/kWh") 
            print(f"  Total revenue: ${total_revenue:.2f}")
        
        print("="*80)


# =====================================================================================
# MAIN SIMULATION
# =====================================================================================

async def main():
    """
    Main simulation setup and execution.
    
    Creates diverse EV and CS agents, runs negotiations, and displays results.
    """
    print("🔋 EV Charging Marketplace Simulation")
    print("="*50)
    
    # Initialize the runtime
    runtime = SingleThreadedAgentRuntime()
    
    # Agent configurations
    ev_configs = [
        {"ev_id": "1", "battery_level": 20, "target_battery_level": 80, "max_acceptable_price": 0.15},
        {"ev_id": "2", "battery_level": 35, "target_battery_level": 90, "max_acceptable_price": 0.20},
        {"ev_id": "3", "battery_level": 10, "target_battery_level": 85, "max_acceptable_price": 0.12},
    ]
    
    cs_configs = [
        {"cs_id": "A", "current_electricity_cost": 0.08, "available_chargers": 3, "min_profit_margin": 0.25},
        {"cs_id": "B", "current_electricity_cost": 0.10, "available_chargers": 5, "min_profit_margin": 0.15},
        {"cs_id": "C", "current_electricity_cost": 0.09, "available_chargers": 2, "min_profit_margin": 0.30},
    ]
    
    # Register agents with runtime
    marketplace = None
    await MarketplaceAgent.register(runtime, "marketplace", lambda: MarketplaceAgent())
    
    for ev_config in ev_configs:
        await EVAgent.register(runtime, f"EV-{ev_config['ev_id']}", 
                              lambda config=ev_config: EVAgent(**config))
    
    for cs_config in cs_configs:
        await CSAgent.register(runtime, f"CS-{cs_config['cs_id']}", 
                              lambda config=cs_config: CSAgent(**config))
    
    print(f"\nInitialized {len(ev_configs)} EV agents and {len(cs_configs)} charging station agents")
    print("\nStarting negotiations...\n")
    
    # Start the runtime
    runtime.start()
    
    try:
        # Initiate charging requests from EVs with small delays
        for i, ev_config in enumerate(ev_configs):
            if i > 0:
                await asyncio.sleep(1)  # Small delay between requests
            
            print(f"\n--- Starting negotiation for EV-{ev_config['ev_id']} ---")
            
            # Calculate energy needed for this EV
            battery_capacity = 60.0
            percentage_needed = (ev_config['target_battery_level'] - ev_config['battery_level']) / 100.0
            energy_needed = battery_capacity * percentage_needed
            
            # Create a charging request and send it to the marketplace
            request = ChargingRequest(
                ev_config['ev_id'], 
                ev_config['battery_level'], 
                ev_config['target_battery_level'], 
                ev_config['max_acceptable_price']
            )
            
            print(f"[{ev_config['ev_id']}] Starting charging request...")
            print(f"[{ev_config['ev_id']}]   Battery: {ev_config['battery_level']}% → {ev_config['target_battery_level']}%")
            print(f"[{ev_config['ev_id']}]   Max price: ${ev_config['max_acceptable_price']:.3f}/kWh")
            print(f"[{ev_config['ev_id']}]   Energy needed: {energy_needed:.1f} kWh")
            
            await runtime.send_message(request, AgentId("marketplace", "default"))
                
            # Give time for negotiation to complete
            await asyncio.sleep(2)
        
        # Allow final message processing
        await asyncio.sleep(3)
        
    finally:
        # Stop runtime and print summary
        await runtime.stop_when_idle()
        # Since we can't access the marketplace instance directly, just print basic summary
        print("\n" + "="*80)
        print("SIMULATION COMPLETED")
        print("="*80)


if __name__ == "__main__":
    """Entry point for the EV charging marketplace simulation"""
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nSimulation interrupted by user")
        sys.exit(0)
    except Exception as e:
        import traceback
        print(f"\n\nSimulation error: {e}")
        print("Full traceback:")
        traceback.print_exc()
        sys.exit(1)