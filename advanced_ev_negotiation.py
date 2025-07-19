#!/usr/bin/env python3
"""
Advanced Multi-Agent EV Charging Negotiation Simulation

Core Architecture:
- Round-based negotiation system with central Orchestrator
- Direct agent-to-agent communication via inboxes
- Parallel negotiations between multiple agents
- LLM-powered decision making for strategic pricing
- 20% target margins for both EV and CS agents

Key Features:
- EVs seek deals 20% below max budget
- Charging stations seek deals 20% above break-even cost
- Multiple simultaneous negotiations per agent
- Automatic deal finalization and agent deactivation
- Maximum 10 rounds to prevent infinite loops
"""

import asyncio
import json
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
from enum import Enum
import uuid
import time
from autogen_ext.models.ollama import OllamaChatCompletionClient
from autogen_core.models import UserMessage


# =====================================================================================
# CORE DATA STRUCTURES
# =====================================================================================

@dataclass
class Offer:
    """Core offer structure for all agent communications"""
    sender_id: str
    price: float
    status: str  # INITIAL, COUNTER, ACCEPT
    round_number: int
    offer_id: str = None
    
    def __post_init__(self):
        if self.offer_id is None:
            self.offer_id = str(uuid.uuid4())[:8]


class OfferStatus(Enum):
    INITIAL = "INITIAL"
    COUNTER = "COUNTER" 
    ACCEPT = "ACCEPT"


# =====================================================================================
# BASE AGENT CLASS
# =====================================================================================

class Agent:
    """Base agent class with inbox management and LLM integration"""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.inbox: List[Offer] = []
        self.is_active = True
        self.llm_client = OllamaChatCompletionClient(model="llama3.1:8b")
        
        print(f"[{self.agent_id}] 🤖 Agent initialized and ready for negotiation")
    
    def receive_offer(self, offer: Offer):
        """Add offer to inbox for processing in next round"""
        if self.is_active:
            self.inbox.append(offer)
            print(f"[{self.agent_id}] 📬 Received {offer.status} offer from {offer.sender_id}: ${offer.price:.3f}")
    
    async def process_inbox(self, round_number: int) -> List[Tuple[str, Offer]]:
        """Process all offers in inbox and return outgoing messages"""
        if not self.is_active:
            return []
        
        print(f"[{self.agent_id}] 📋 Processing {len(self.inbox)} offers in round {round_number}")
        
        outgoing_messages = []
        for offer in self.inbox:
            response = await self._process_individual_offer(offer, round_number)
            if response:
                recipient_id, response_offer = response
                outgoing_messages.append((recipient_id, response_offer))
        
        return outgoing_messages
    
    async def _process_individual_offer(self, offer: Offer, round_number: int) -> Optional[Tuple[str, Offer]]:
        """Process a single offer - to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement offer processing logic")
    
    def deactivate(self, reason: str = "Deal completed"):
        """Remove agent from active negotiation"""
        self.is_active = False
        print(f"[{self.agent_id}] 🛑 Agent deactivated: {reason}")


# =====================================================================================
# EV AGENT - SEEKS LOW PRICES
# =====================================================================================

class EV_Agent(Agent):
    """Electric Vehicle agent seeking charging deals below target price"""
    
    def __init__(self, agent_id: str, max_budget: float, energy_needed: float = 35.0):
        super().__init__(agent_id)
        self.max_budget = max_budget
        self.energy_needed = energy_needed
        # More aggressive target - EVs want 35% below max budget (very greedy)
        self.target_price = max_budget * 0.65
        # Even "good deals" threshold is high - 25% below max budget  
        self.excellent_deal_threshold = max_budget * 0.75
        
        print(f"[{self.agent_id}] 🚗 EV Agent ready - Max budget: ${max_budget:.3f}, Target: ${self.target_price:.3f}, Excellent deal: ${self.excellent_deal_threshold:.3f}")
    
    async def _process_individual_offer(self, offer: Offer, round_number: int) -> Optional[Tuple[str, Offer]]:
        """Process individual offer using EV logic - now mostly LLM-driven"""
        
        # Only auto-accept truly excellent deals (very rare)
        if offer.price <= self.target_price:
            print(f"[{self.agent_id}] ✅ Auto-accepting {offer.sender_id} offer: ${offer.price:.3f} <= ${self.target_price:.3f} (RARE excellent deal!)")
            
            accept_offer = Offer(
                sender_id=self.agent_id,
                price=offer.price,
                status=OfferStatus.ACCEPT.value,
                round_number=round_number
            )
            return (offer.sender_id, accept_offer)
        
        # Use LLM for ALL other decisions (most common path)
        try:
            decision = await self._llm_negotiate_decision(offer, round_number)
            if decision:
                action = decision.get("action")
                
                if action == "ACCEPT":
                    print(f"[{self.agent_id}] ✅ LLM decided to ACCEPT {offer.sender_id}: ${offer.price:.3f}")
                    accept_offer = Offer(
                        sender_id=self.agent_id,
                        price=offer.price,
                        status=OfferStatus.ACCEPT.value,
                        round_number=round_number
                    )
                    return (offer.sender_id, accept_offer)
                
                elif action == "COUNTER":
                    counter_price = decision.get("price")
                    if counter_price and counter_price < offer.price:
                        print(f"[{self.agent_id}] 🔄 LLM counter-offering {offer.sender_id}: ${counter_price:.3f}")
                        
                        counter_offer = Offer(
                            sender_id=self.agent_id,
                            price=counter_price,
                            status=OfferStatus.COUNTER.value,
                            round_number=round_number
                        )
                        return (offer.sender_id, counter_offer)
                
                # "REJECT" or "WAIT" - no response
                print(f"[{self.agent_id}] ⏭️ LLM decided to {action} offer from {offer.sender_id}")
                return None
            
        except Exception as e:
            print(f"[{self.agent_id}] ⚠️ LLM error for {offer.sender_id}: {e}")
        
        # Fallback: reject if LLM fails
        print(f"[{self.agent_id}] ⏭️ No response to {offer.sender_id} (LLM failed or price too high: ${offer.price:.3f})")
        return None
    
    async def _llm_negotiate_decision(self, offer: Offer, round_number: int) -> Optional[Dict]:
        """Use LLM for comprehensive negotiation decisions with strategic thinking"""
        
        # Calculate market context
        urgency_factor = "HIGH" if round_number >= 7 else "MEDIUM" if round_number >= 4 else "LOW"
        price_gap = offer.price - self.target_price
        acceptable_threshold = self.max_budget * 0.90  # Will accept up to 90% of budget if desperate
        
        prompt = f"""You are {self.agent_id}, a strategic EV owner negotiating for charging services in round {round_number}/10.

FINANCIAL SITUATION:
- Your absolute maximum budget: ${self.max_budget:.3f}/kWh  
- Your ideal target price: ${self.target_price:.3f}/kWh (35% below max)
- Current offer from {offer.sender_id}: ${offer.price:.3f}/kWh
- Price gap from your target: ${price_gap:.3f}/kWh

NEGOTIATION CONTEXT:
- Round: {round_number}/10 (Urgency: {urgency_factor})
- If you don't get a deal, you get nothing
- Other charging stations are also competing
- You're trying to be greedy but realistic

STRATEGIC OPTIONS:
1. ACCEPT - Take this offer now (if it's good enough or time is running out)
2. COUNTER - Make a lower counter-offer (risk them rejecting)  
3. REJECT - Wait for better offers from other stations
4. WAIT - Hold off this round, see what else comes

STRATEGIC THINKING:
- Early rounds (1-3): Be aggressive, push for low prices
- Middle rounds (4-6): Balance greed with realism  
- Late rounds (7-10): Consider accepting reasonable offers

If COUNTER, your price should be between ${self.target_price:.3f} and ${offer.price:.3f}.
If ACCEPT, explain why you're taking this deal now.

RESPOND WITH VALID JSON:
{{"action": "ACCEPT", "reasoning": "Why accepting this offer"}}
{{"action": "COUNTER", "price": 0.125, "reasoning": "Why counter-offering at this price"}}  
{{"action": "REJECT", "reasoning": "Why rejecting and waiting"}}
{{"action": "WAIT", "reasoning": "Why waiting this round"}}

JSON RESPONSE:"""
        
        try:
            response = await self.llm_client.create(
                messages=[UserMessage(content=prompt, source="user")]
            )
            
            # Parse JSON response
            response_text = response.content.strip()
            decision = json.loads(response_text)
            
            # Validate decision
            action = decision.get("action", "").upper()
            if action in ["ACCEPT", "COUNTER", "REJECT", "WAIT"]:
                print(f"[{self.agent_id}] 🧠 LLM reasoning: {decision.get('reasoning', 'No reason given')}")
                return decision
            else:
                print(f"[{self.agent_id}] 🚫 Invalid LLM action: {action}")
                return None
                
        except (json.JSONDecodeError, ValueError) as e:
            print(f"[{self.agent_id}] 🚫 Could not parse LLM JSON response: {response_text}")
            return None


# =====================================================================================
# CHARGING STATION AGENT - SEEKS HIGH MARGINS
# =====================================================================================

class CS_Agent(Agent):
    """Charging Station agent seeking profitable deals above target price"""
    
    def __init__(self, agent_id: str, break_even_cost: float, available_capacity: int = 3):
        super().__init__(agent_id)
        self.break_even_cost = break_even_cost
        self.available_capacity = available_capacity
        # More aggressive target - CS wants 50% above break-even (very greedy)
        self.target_price = break_even_cost * 1.50
        # Start even higher - 75% above break-even for initial offers
        self.initial_offer_price = break_even_cost * 1.75
        # Minimum acceptable is still break-even + 25%
        self.minimum_acceptable = break_even_cost * 1.25
        
        print(f"[{self.agent_id}] ⚡ CS Agent ready - Break-even: ${break_even_cost:.3f}, Target: ${self.target_price:.3f}, Initial: ${self.initial_offer_price:.3f}")
    
    async def _process_individual_offer(self, offer: Offer, round_number: int) -> Optional[Tuple[str, Offer]]:
        """Process individual counter-offer using CS logic - now mostly LLM-driven"""
        
        # Only process COUNTER offers (not initial offers)
        if offer.status != OfferStatus.COUNTER.value:
            return None
        
        # Only auto-accept truly excellent counter-offers (very rare)
        if offer.price >= self.target_price:
            print(f"[{self.agent_id}] ✅ Auto-accepting {offer.sender_id} counter: ${offer.price:.3f} >= ${self.target_price:.3f} (RARE excellent deal!)")
            
            accept_offer = Offer(
                sender_id=self.agent_id,
                price=offer.price,
                status=OfferStatus.ACCEPT.value,
                round_number=round_number
            )
            return (offer.sender_id, accept_offer)
        
        # Use LLM for ALL other decisions (most common path)
        try:
            decision = await self._llm_strategic_response(offer, round_number)
            if decision:
                action = decision.get("action")
                
                if action == "ACCEPT":
                    print(f"[{self.agent_id}] ✅ LLM decided to ACCEPT {offer.sender_id}: ${offer.price:.3f}")
                    accept_offer = Offer(
                        sender_id=self.agent_id,
                        price=offer.price,
                        status=OfferStatus.ACCEPT.value,
                        round_number=round_number
                    )
                    return (offer.sender_id, accept_offer)
                
                elif action == "COUNTER":
                    counter_price = decision.get("price")
                    if counter_price and counter_price >= self.break_even_cost:
                        print(f"[{self.agent_id}] 🔄 LLM counter-responding to {offer.sender_id}: ${counter_price:.3f}")
                        
                        counter_offer = Offer(
                            sender_id=self.agent_id,
                            price=counter_price,
                            status=OfferStatus.COUNTER.value,
                            round_number=round_number
                        )
                        return (offer.sender_id, counter_offer)
                
                # "REJECT" or "WAIT" - no response
                print(f"[{self.agent_id}] ⏭️ LLM decided to {action} counter-offer from {offer.sender_id}")
                return None
                
        except Exception as e:
            print(f"[{self.agent_id}] ⚠️ LLM error for {offer.sender_id}: {e}")
        
        # Fallback: reject if LLM fails
        print(f"[{self.agent_id}] ⏭️ No response to {offer.sender_id} (LLM failed or counter too low: ${offer.price:.3f})")
        return None
    
    async def _llm_strategic_response(self, offer: Offer, round_number: int) -> Optional[Dict]:
        """Use LLM for comprehensive strategic responses to counter-offers"""
        
        # Calculate market context
        urgency_factor = "HIGH" if round_number >= 8 else "MEDIUM" if round_number >= 5 else "LOW"
        profit_margin = ((offer.price - self.break_even_cost) / self.break_even_cost) * 100
        competitive_pressure = "HIGH" if round_number <= 3 else "MEDIUM"
        
        prompt = f"""You are {self.agent_id}, a strategic charging station operator in round {round_number}/10.

FINANCIAL SITUATION:
- Your break-even cost: ${self.break_even_cost:.3f}/kWh
- Your target profit price: ${self.target_price:.3f}/kWh (50% above break-even)
- Minimum acceptable: ${self.minimum_acceptable:.3f}/kWh (25% above break-even)
- Counter-offer from {offer.sender_id}: ${offer.price:.3f}/kWh
- Profit margin at this price: {profit_margin:.1f}%

BUSINESS CONTEXT:
- Round: {round_number}/10 (Urgency: {urgency_factor})
- Competitive pressure: {competitive_pressure} (other stations are competing)
- If no deal, you get $0 revenue from this customer
- You want to maximize profit but also secure deals

STRATEGIC OPTIONS:
1. ACCEPT - Take this counter-offer (if profitable enough or time pressure)
2. COUNTER - Make a new offer (risk them going to competitors)
3. REJECT - Decline and wait (risk losing customer entirely)

STRATEGIC THINKING:
- Early rounds (1-4): Be greedy, push for high margins
- Middle rounds (5-7): Balance profit with deal closure
- Late rounds (8-10): Consider accepting lower but profitable offers

If COUNTER, your price should be between ${offer.price:.3f} and ${self.target_price:.3f}.
If ACCEPT, explain why this margin is acceptable now.

RESPOND WITH VALID JSON:
{{"action": "ACCEPT", "reasoning": "Why accepting this counter-offer"}}
{{"action": "COUNTER", "price": 0.145, "reasoning": "Why counter-offering at this price"}}
{{"action": "REJECT", "reasoning": "Why rejecting this counter-offer"}}

JSON RESPONSE:"""
        
        try:
            response = await self.llm_client.create(
                messages=[UserMessage(content=prompt, source="user")]
            )
            
            # Parse JSON response
            response_text = response.content.strip()
            decision = json.loads(response_text)
            
            # Validate decision
            action = decision.get("action", "").upper()
            if action in ["ACCEPT", "COUNTER", "REJECT"]:
                print(f"[{self.agent_id}] 🧠 LLM reasoning: {decision.get('reasoning', 'No reason given')}")
                return decision
            else:
                print(f"[{self.agent_id}] 🚫 Invalid LLM action: {action}")
                return None
                
        except (json.JSONDecodeError, ValueError) as e:
            print(f"[{self.agent_id}] 🚫 Could not parse LLM JSON response: {response_text}")
            return None
    
    async def make_initial_offers(self, ev_agents: List[EV_Agent], round_number: int) -> List[Tuple[str, Offer]]:
        """Generate initial offers to all active EV agents"""
        initial_offers = []
        
        for ev_agent in ev_agents:
            if ev_agent.is_active:
                initial_offer = Offer(
                    sender_id=self.agent_id,
                    price=self.initial_offer_price,
                    status=OfferStatus.INITIAL.value,
                    round_number=round_number
                )
                initial_offers.append((ev_agent.agent_id, initial_offer))
                print(f"[{self.agent_id}] 📢 Initial offer to {ev_agent.agent_id}: ${self.initial_offer_price:.3f}")
        
        return initial_offers


# =====================================================================================
# CENTRAL ORCHESTRATOR - MANAGES SIMULATION ROUNDS
# =====================================================================================

class Orchestrator:
    """Central coordinator managing round-based negotiation simulation"""
    
    def __init__(self, ev_agents: List[EV_Agent], cs_agents: List[CS_Agent], max_rounds: int = 10):
        self.ev_agents = ev_agents
        self.cs_agents = cs_agents
        self.all_agents = ev_agents + cs_agents
        self.max_rounds = max_rounds
        self.current_round = 0
        self.completed_deals: List[Dict] = []
        
        print(f"[Orchestrator] 🎮 Initialized with {len(ev_agents)} EVs, {len(cs_agents)} charging stations")
        print(f"[Orchestrator] 🕐 Maximum rounds: {max_rounds}")
    
    async def run_simulation(self):
        """Execute the complete negotiation simulation"""
        print("\n" + "="*80)
        print("🔋 ADVANCED EV CHARGING NEGOTIATION SIMULATION")
        print("="*80)
        print(f"🎯 EV Goal: Secure deals 35% below max budget (very greedy)")
        print(f"💰 CS Goal: Secure deals 50% above break-even cost (very greedy)")
        print(f"🔥 Both sides must negotiate - auto-accepts are rare!")
        print("="*80)
        
        # Round 1: Charging stations make initial offers
        await self._execute_initial_round()
        
        # Main negotiation rounds
        for round_num in range(2, self.max_rounds + 1):
            if not await self._execute_negotiation_round(round_num):
                break
        
        await self._print_final_results()
    
    async def _execute_initial_round(self):
        """Round 1: Charging stations broadcast initial offers"""
        self.current_round = 1
        print(f"\n🚀 ROUND {self.current_round}: Initial Offers")
        print("-" * 50)
        
        all_messages = []
        
        # Collect initial offers from all charging stations
        for cs_agent in self.cs_agents:
            if cs_agent.is_active:
                initial_offers = await cs_agent.make_initial_offers(self.ev_agents, self.current_round)
                all_messages.extend(initial_offers)
        
        # Deliver messages to recipients
        await self._deliver_messages(all_messages)
        
        print(f"[Orchestrator] 📬 Delivered {len(all_messages)} initial offers")
    
    async def _execute_negotiation_round(self, round_number: int) -> bool:
        """Execute a single negotiation round, return False if simulation should end"""
        self.current_round = round_number
        print(f"\n🔄 ROUND {round_number}: Negotiations")
        print("-" * 50)
        
        # Check termination conditions
        active_evs = [ev for ev in self.ev_agents if ev.is_active]
        if not active_evs:
            print("[Orchestrator] 🏁 All EVs have completed deals - simulation ending")
            return False
        
        # Process all agent inboxes
        all_messages = []
        for agent in self.all_agents:
            if agent.is_active:
                messages = await agent.process_inbox(round_number)
                all_messages.extend(messages)
        
        # Clear all inboxes for next round
        self._clear_all_inboxes()
        
        # Deliver new messages
        await self._deliver_messages(all_messages)
        
        # Check for deal completions
        await self._process_deal_completions(all_messages)
        
        print(f"[Orchestrator] 📬 Round {round_number} complete: {len(all_messages)} messages exchanged")
        
        return True
    
    def _clear_all_inboxes(self):
        """Clear all agent inboxes to prepare for next round"""
        for agent in self.all_agents:
            agent.inbox.clear()
    
    async def _deliver_messages(self, messages: List[Tuple[str, Offer]]):
        """Deliver messages to recipient agents"""
        for recipient_id, offer in messages:
            recipient = self._find_agent_by_id(recipient_id)
            if recipient and recipient.is_active:
                recipient.receive_offer(offer)
    
    async def _process_deal_completions(self, messages: List[Tuple[str, Offer]]):
        """Process ACCEPT messages and deactivate agents"""
        for recipient_id, offer in messages:
            if offer.status == OfferStatus.ACCEPT.value:
                # Find both agents involved in the deal
                acceptor = self._find_agent_by_id(offer.sender_id)
                acceptee = self._find_agent_by_id(recipient_id)
                
                if acceptor and acceptee and both_agents_active(acceptor, acceptee):
                    # Record the deal
                    deal = {
                        'round': self.current_round,
                        'ev_agent': acceptor.agent_id if isinstance(acceptor, EV_Agent) else acceptee.agent_id,
                        'cs_agent': acceptor.agent_id if isinstance(acceptor, CS_Agent) else acceptee.agent_id,
                        'final_price': offer.price,
                        'total_cost': offer.price * 35.0  # Standard energy amount
                    }
                    self.completed_deals.append(deal)
                    
                    # Deactivate both agents
                    acceptor.deactivate("Deal completed")
                    acceptee.deactivate("Deal completed")
                    
                    print(f"[Orchestrator] 🤝 DEAL COMPLETED: {deal['ev_agent']} ↔ {deal['cs_agent']} @ ${offer.price:.3f}")
    
    def _find_agent_by_id(self, agent_id: str) -> Optional[Agent]:
        """Find agent by ID"""
        for agent in self.all_agents:
            if agent.agent_id == agent_id:
                return agent
        return None
    
    async def _print_final_results(self):
        """Print comprehensive simulation results"""
        print("\n" + "="*80)
        print("🎯 SIMULATION RESULTS")
        print("="*80)
        
        print(f"\n📊 COMPLETED DEALS: {len(self.completed_deals)}")
        for i, deal in enumerate(self.completed_deals, 1):
            print(f"  {i}. {deal['ev_agent']} ↔ {deal['cs_agent']}: ${deal['final_price']:.3f}/kWh (Total: ${deal['total_cost']:.2f})")
        
        active_evs = [ev for ev in self.ev_agents if ev.is_active]
        active_css = [cs for cs in self.cs_agents if cs.is_active]
        
        print(f"\n📈 MARKET EFFICIENCY:")
        print(f"  • EV Success Rate: {((len(self.ev_agents) - len(active_evs)) / len(self.ev_agents)) * 100:.1f}%")
        print(f"  • CS Success Rate: {((len(self.cs_agents) - len(active_css)) / len(self.cs_agents)) * 100:.1f}%")
        print(f"  • Total Rounds: {self.current_round}")
        
        if self.completed_deals:
            avg_price = sum(deal['final_price'] for deal in self.completed_deals) / len(self.completed_deals)
            total_revenue = sum(deal['total_cost'] for deal in self.completed_deals)
            print(f"  • Average Final Price: ${avg_price:.3f}/kWh")
            print(f"  • Total Market Revenue: ${total_revenue:.2f}")
        
        print(f"\n🤖 AGENTS STILL ACTIVE:")
        print(f"  • EVs: {len(active_evs)} ({[ev.agent_id for ev in active_evs]})")
        print(f"  • Charging Stations: {len(active_css)} ({[cs.agent_id for cs in active_css]})")
        
        print("="*80)


# =====================================================================================
# HELPER FUNCTIONS
# =====================================================================================

def both_agents_active(agent1: Agent, agent2: Agent) -> bool:
    """Check if both agents are still active"""
    return agent1.is_active and agent2.is_active


# =====================================================================================
# MAIN SIMULATION SETUP
# =====================================================================================

async def main():
    """Setup and run the advanced EV charging negotiation simulation"""
    
    print("🔌 Initializing Advanced EV Charging Negotiation Simulation...")
    print("🦙 Using Llama 3.1 8B via Ollama for strategic decision-making")
    
    # Create diverse EV agents with different budgets
    ev_agents = [
        EV_Agent("EV1", max_budget=0.150, energy_needed=36.0),
        EV_Agent("EV2", max_budget=0.180, energy_needed=42.0),
        EV_Agent("EV3", max_budget=0.140, energy_needed=30.0),
        EV_Agent("EV4", max_budget=0.200, energy_needed=45.0),
        EV_Agent("EV5", max_budget=0.130, energy_needed=25.0)
    ]
    
    # Create charging station agents with different cost structures
    cs_agents = [
        CS_Agent("CS_A", break_even_cost=0.080, available_capacity=3),
        CS_Agent("CS_B", break_even_cost=0.095, available_capacity=5),
        CS_Agent("CS_C", break_even_cost=0.075, available_capacity=2),
        CS_Agent("CS_D", break_even_cost=0.100, available_capacity=4)
    ]
    
    # Create and run orchestrator
    orchestrator = Orchestrator(ev_agents, cs_agents, max_rounds=10)
    await orchestrator.run_simulation()


if __name__ == "__main__":
    """Entry point for advanced EV charging negotiation simulation"""
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n🛑 Simulation interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Simulation error: {e}")
        import traceback
        traceback.print_exc()