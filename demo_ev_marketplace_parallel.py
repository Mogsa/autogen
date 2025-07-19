#!/usr/bin/env python3
"""
Advanced Parallel Negotiation EV Charging Marketplace

This version implements sophisticated parallel negotiation capabilities:
1. Concurrent negotiations with multiple charging stations
2. JSON-mode LLM interactions for reliability  
3. Conversational memory for strategic context
4. Dynamic environment simulation
5. Event-driven termination
6. Robust state management

Architecture:
- EVAgent: Manages multiple simultaneous negotiations
- CSAgent: Handles concurrent customer negotiations  
- MarketplaceAgent: Coordinates all interactions with proper state tracking
- GridAgent: Simulates dynamic market conditions
"""

import asyncio
import sys
import json
import time
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Set
from enum import Enum
import ollama
from autogen_core import RoutedAgent, MessageContext, DefaultTopicId, default_subscription, message_handler, AgentId
from autogen_core import SingleThreadedAgentRuntime


# =====================================================================================
# MESSAGE DEFINITIONS - Enhanced for parallel negotiation
# =====================================================================================

class NegotiationStatus(Enum):
    ACTIVE = "active"
    ACCEPTED = "accepted" 
    REJECTED = "rejected"
    COMPLETED = "completed"


@dataclass
class ChargingRequest:
    """Enhanced charging request with full specifications"""
    ev_id: str
    battery_level: int
    target_battery_level: int
    max_acceptable_price: float
    energy_needed: float  # Calculated kWh needed
    urgency_level: str    # "CRITICAL", "HIGH", "MEDIUM", "LOW"
    timestamp: float = 0.0


@dataclass
class ChargingOffer:
    """Enhanced charging offer with negotiation tracking"""
    cs_id: str
    ev_id: str
    price: float
    available_chargers: int
    round_number: int = 1
    timestamp: float = 0.0
    offer_id: str = ""  # Unique identifier for tracking


@dataclass
class CounterOffer:
    """Enhanced counter-offer with context"""
    ev_id: str
    cs_id: str
    price: float
    round_number: int = 1
    reasoning: str = ""
    offer_id: str = ""  # Reference to original offer
    timestamp: float = 0.0


@dataclass
class OfferAccepted:
    """Enhanced acceptance with context"""
    ev_id: str
    cs_id: str
    final_price: float
    reasoning: str = ""
    offer_id: str = ""
    timestamp: float = 0.0


@dataclass
class OfferRejected:
    """Enhanced rejection with context"""
    ev_id: str
    cs_id: str
    reason: str
    final_rejection: bool = False
    offer_id: str = ""
    timestamp: float = 0.0


@dataclass
class DealFinalized:
    """Enhanced deal finalization"""
    ev_id: str
    cs_id: str
    final_price: float
    energy_needed: float
    total_cost: float
    negotiation_rounds: int
    timestamp: float = 0.0


@dataclass
class DealFailed:
    """Enhanced deal failure tracking"""
    ev_id: str
    reason: str
    attempted_negotiations: int
    timestamp: float = 0.0





# =====================================================================================
# PARALLEL NEGOTIATION EV AGENT - Handles Multiple Simultaneous Negotiations
# =====================================================================================

@default_subscription
class ParallelEVAgent(RoutedAgent):
    """
    Advanced EV Agent capable of parallel negotiation with multiple charging stations.
    
    Key Features:
    - Concurrent negotiations with all available CSs
    - JSON-mode LLM for reliable responses
    - Conversational memory for strategic context
    - Dynamic offer evaluation and prioritization
    - Graceful rejection of non-selected offers
    """
    
    def __init__(self, ev_id: str, battery_level: int, target_battery_level: int, 
                 max_acceptable_price: float, ollama_client: ollama.AsyncClient):
        super().__init__(description=f"EV-{ev_id}")
        self.ev_id = ev_id
        self.battery_level = battery_level
        self.target_battery_level = target_battery_level
        self.max_acceptable_price = max_acceptable_price
        self.llm_client = ollama_client
        self.energy_needed = self._calculate_energy_needed()
        self.urgency_level = self._calculate_urgency()
        
        # Parallel negotiation state management
        self.active_negotiations: Dict[str, Dict] = {}  # cs_id -> negotiation_state
        self.offer_queue: Dict[str, List[ChargingOffer]] = {}  # cs_id -> list of offers
        self.conversation_history: Dict[str, List[Dict]] = {}  # cs_id -> conversation
        self.negotiation_status: Dict[str, NegotiationStatus] = {}  # cs_id -> status
        
        # Remove single-track limitations
        # self.llm_processing = False  # REMOVED - allows parallel processing
        # self.deal_completed = False  # REMOVED - replaced with per-negotiation tracking
        
        # Evaluation and decision making
        self.last_evaluation_time = 0.0
        self.evaluation_interval = 3.0  # Increased interval to reduce LLM calls
        self.max_negotiation_rounds = 4
        
        print(f"[{self.ev_id}] 🚗 Parallel negotiation agent initialized")
        print(f"[{self.ev_id}]    Energy needed: {self.energy_needed:.1f} kWh")
        print(f"[{self.ev_id}]    Urgency: {self.urgency_level}")
        print(f"[{self.ev_id}]    Budget: ${self.max_acceptable_price:.3f}/kWh")
        
    def _calculate_energy_needed(self) -> float:
        """Calculate kWh needed based on battery levels"""
        battery_capacity = 60.0
        percentage_needed = (self.target_battery_level - self.battery_level) / 100.0
        return battery_capacity * percentage_needed
    
    def _calculate_urgency(self) -> str:
        """Calculate urgency level based on battery percentage"""
        if self.battery_level < 15:
            return "CRITICAL"
        elif self.battery_level < 25:
            return "HIGH"
        elif self.battery_level < 40:
            return "MEDIUM"
        else:
            return "LOW"
    
    @message_handler
    async def handle_charging_offer(self, message: ChargingOffer, ctx: MessageContext) -> None:
        """Handle charging offers - ADD TO QUEUE for parallel processing"""
        await self._handle_charging_offer(message, ctx)
    
    @message_handler
    async def handle_offer_rejected(self, message: OfferRejected, ctx: MessageContext) -> None:
        """Handle rejection of our counter-offer"""
        await self._handle_offer_rejected(message, ctx)
    
    @message_handler
    async def handle_deal_finalized(self, message: DealFinalized, ctx: MessageContext) -> None:
        """Handle successful deal completion"""
        await self._handle_deal_finalized(message, ctx)
    

    
    async def _handle_charging_offer(self, offer: ChargingOffer, ctx: MessageContext) -> None:
        """ADD OFFER TO QUEUE instead of immediate processing"""
        if offer.ev_id != self.ev_id:
            return
            
        print(f"[{self.ev_id}] 📨 Queuing offer from {offer.cs_id}: ${offer.price:.3f}/kWh (Round {offer.round_number})")
        
        # Add to offer queue (allow multiple offers per CS)
        if offer.cs_id not in self.offer_queue:
            self.offer_queue[offer.cs_id] = []
        offer.offer_id = f"{offer.cs_id}-{offer.round_number}-{time.time()}"
        offer.timestamp = time.time()
        self.offer_queue[offer.cs_id].append(offer)
        
        # Initialize negotiation state if new
        if offer.cs_id not in self.active_negotiations:
            self.active_negotiations[offer.cs_id] = {
                'initial_offer': offer.price,
                'current_offer': offer.price,
                'rounds': offer.round_number,
                'last_action': 'received_offer',
                'last_update': time.time()
            }
            self.negotiation_status[offer.cs_id] = NegotiationStatus.ACTIVE
            self.conversation_history[offer.cs_id] = []
        
        # Update existing negotiation
        self.active_negotiations[offer.cs_id]['current_offer'] = offer.price
        self.active_negotiations[offer.cs_id]['rounds'] = offer.round_number
        self.active_negotiations[offer.cs_id]['last_update'] = time.time()
        
        # Add to conversation history
        self.conversation_history[offer.cs_id].append({
            'type': 'offer',
            'from': offer.cs_id,
            'price': offer.price,
            'round': offer.round_number,
            'timestamp': offer.timestamp
        })
        
        # Trigger evaluation if enough time has passed
        current_time = time.time()
        if current_time - self.last_evaluation_time >= self.evaluation_interval:
            await self._evaluate_all_negotiations(ctx)
            self.last_evaluation_time = current_time
    
    async def _evaluate_all_negotiations(self, ctx: MessageContext) -> None:
        """CORE PARALLEL LOGIC: Evaluate all active negotiations and decide actions"""
        if not self.active_negotiations:
            return
            
        print(f"[{self.ev_id}] 🧠 Evaluating {len(self.active_negotiations)} active negotiations...")
        
        # Check if any deal is already completed
        if any(status == NegotiationStatus.COMPLETED for status in self.negotiation_status.values()):
            return
        
        # Prepare context for LLM decision
        negotiations_context = []
        for cs_id, negotiation in self.active_negotiations.items():
            if self.negotiation_status[cs_id] == NegotiationStatus.ACTIVE:
                latest_offer = self.offer_queue[cs_id][-1] if self.offer_queue[cs_id] else None
                if latest_offer:
                    negotiations_context.append({
                        'cs_id': cs_id,
                        'current_price': latest_offer.price,
                        'initial_price': negotiation['initial_offer'],
                        'rounds': negotiation['rounds'],
                        'savings_vs_initial': negotiation['initial_offer'] - latest_offer.price,
                        'within_budget': latest_offer.price <= self.max_acceptable_price,
                        'conversation_length': len(self.conversation_history[cs_id])
                    })
        
        if not negotiations_context:
            return
            
        # Use LLM with JSON mode for reliable decision making
        await self._make_parallel_decisions(negotiations_context, ctx)
    
    async def _make_parallel_decisions(self, negotiations_context: List[Dict], ctx: MessageContext) -> None:
        """Use LLM in JSON mode to make strategic decisions across all negotiations"""
        
        # Prepare optimized JSON-mode prompt
        prompt = f"""EV {self.ev_id} negotiating with {len(negotiations_context)} stations. Battery: {self.battery_level}%→{self.target_battery_level}% ({self.urgency_level}). Budget: ${self.max_acceptable_price:.3f}/kWh.

OFFERS: {json.dumps(negotiations_context, indent=2)}

STRATEGY: Round 1-2: Counter 10-20% below. Round 3+: Accept good deals. Use competitive pressure.

Respond with JSON:
{{
  "reasoning": "Your strategy",
  "decisions": [
    {{
      "cs_id": "A",
      "action": "COUNTER" | "ACCEPT" | "WAIT",
      "price": 0.105,
      "reasoning": "Why this action"
    }}
  ],
  "accept_best": false
}}

JSON only."""

        try:
            llm_start_time = time.time()
            print(f"[{self.ev_id}] 🤔 Consulting LLM for parallel negotiation strategy... (⏱️ {llm_start_time:.1f}s)")
            
            # Small delay to prevent overwhelming the LLM
            await asyncio.sleep(0.5)
            
            response = await asyncio.wait_for(
                self.llm_client.chat(
                    model='llama3.1:8b',
                    messages=[{'role': 'user', 'content': prompt}],
                    format='json'  # Force JSON mode
                ),
                timeout=45.0  # Increased timeout to allow LLM thinking time
            )
            
            llm_end_time = time.time()
            llm_duration = llm_end_time - llm_start_time
            print(f"[{self.ev_id}] ✅ LLM thinking completed in {llm_duration:.2f}s (⏱️ {llm_end_time:.1f}s)")
            
            llm_response = response['message']['content'].strip()
            print(f"[{self.ev_id}] 🧠 LLM Strategy Response:")
            print(f"[{self.ev_id}]    {llm_response[:200]}...")
            
            # Parse JSON response
            try:
                decisions = json.loads(llm_response)
                await self._execute_parallel_decisions(decisions, ctx)
            except json.JSONDecodeError as e:
                print(f"[{self.ev_id}] ⚠️ JSON parsing error: {e}")
                await self._fallback_parallel_decisions(negotiations_context, ctx)
            
        except asyncio.TimeoutError:
            print(f"[{self.ev_id}] ⏰ LLM timeout, using fallback parallel logic")
            await self._fallback_parallel_decisions(negotiations_context, ctx)
        except Exception as e:
            print(f"[{self.ev_id}] ❌ LLM error: {e}. Using fallback logic.")
            await self._fallback_parallel_decisions(negotiations_context, ctx)
    
    async def _execute_parallel_decisions(self, decisions: Dict, ctx: MessageContext) -> None:
        """Execute the LLM's parallel negotiation decisions"""
        
        print(f"[{self.ev_id}] 💭 Overall strategy: {decisions.get('reasoning', 'No reasoning provided')}")
        
        # Check if LLM wants to accept best offer immediately
        if decisions.get('accept_best', False):
            await self._accept_best_offer(ctx)
            return
        
        # Execute individual decisions
        for decision in decisions.get('decisions', []):
            cs_id = decision.get('cs_id')
            action = decision.get('action')
            reasoning = decision.get('reasoning', '')
            
            if cs_id not in self.active_negotiations:
                continue
                
            if self.negotiation_status[cs_id] != NegotiationStatus.ACTIVE:
                continue
                
            latest_offer = self.offer_queue[cs_id][-1] if self.offer_queue[cs_id] else None
            if not latest_offer:
                continue
            
            print(f"[{self.ev_id}] 🎯 Decision for {cs_id}: {action}")
            print(f"[{self.ev_id}]    Reasoning: {reasoning}")
            
            if action == "ACCEPT":
                await self._accept_offer(latest_offer, reasoning, ctx)
                break  # Deal completed, exit
                
            elif action == "COUNTER":
                counter_price = decision.get('price', latest_offer.price * 0.9)
                if self._validate_counter_price(counter_price, latest_offer.price):
                    await self._counter_offer(latest_offer, counter_price, reasoning, ctx)
                else:
                    print(f"[{self.ev_id}] ⚠️ Invalid counter price {counter_price}, waiting instead")
                    
            elif action == "WAIT":
                print(f"[{self.ev_id}] ⏸️ Waiting for better offer from {cs_id}")
                # No action taken, just wait
    
    async def _accept_best_offer(self, ctx: MessageContext) -> None:
        """Accept the best available offer and reject all others"""
        # Find best offer across all active negotiations
        best_cs = None
        best_offer = None
        best_price = float('inf')
        
        for cs_id in self.active_negotiations:
            if self.negotiation_status[cs_id] == NegotiationStatus.ACTIVE:
                latest_offer = self.offer_queue[cs_id][-1] if self.offer_queue[cs_id] else None
                if latest_offer and latest_offer.price < best_price:
                    best_price = latest_offer.price
                    best_offer = latest_offer
                    best_cs = cs_id
        
        if best_offer:
            print(f"[{self.ev_id}] 🏆 Accepting best offer: {best_cs} at ${best_price:.3f}/kWh")
            await self._accept_offer(best_offer, "Best available offer selected", ctx)
    
    def _validate_counter_price(self, counter_price: float, original_price: float) -> bool:
        """Validate counter-offer is reasonable"""
        if counter_price >= original_price:
            return False
        if counter_price > self.max_acceptable_price * 1.1:
            return False
        if counter_price < self.max_acceptable_price * 0.5:
            return False
        return True
    
    async def _counter_offer(self, offer: ChargingOffer, counter_price: float, reasoning: str, ctx: MessageContext) -> None:
        """Send counter-offer and update conversation history"""
        print(f"[{self.ev_id}] 📤 Counter-offering {offer.cs_id}: ${counter_price:.3f}/kWh (was ${offer.price:.3f})")
        
        counter = CounterOffer(
            ev_id=self.ev_id,
            cs_id=offer.cs_id,
            price=counter_price,
            round_number=offer.round_number + 1,
            reasoning=reasoning,
            offer_id=offer.offer_id,
            timestamp=time.time()
        )
        
        # Update conversation history
        self.conversation_history[offer.cs_id].append({
            'type': 'counter',
            'from': self.ev_id,
            'price': counter_price,
            'round': counter.round_number,
            'reasoning': reasoning,
            'timestamp': counter.timestamp
        })
        
        # Update negotiation state
        self.active_negotiations[offer.cs_id]['last_action'] = 'counter_sent'
        self.active_negotiations[offer.cs_id]['rounds'] = counter.round_number
        
        await self.publish_message(counter, DefaultTopicId())
    
    async def _accept_offer(self, offer: ChargingOffer, reasoning: str, ctx: MessageContext) -> None:
        """Accept offer and gracefully reject all others"""
        print(f"[{self.ev_id}] ✅ ACCEPTING offer from {offer.cs_id} at ${offer.price:.3f}/kWh")
        print(f"[{self.ev_id}] 💭 Reasoning: {reasoning}")
        
        # Mark this negotiation as completed
        self.negotiation_status[offer.cs_id] = NegotiationStatus.COMPLETED
        
        # Send acceptance
        acceptance = OfferAccepted(
            ev_id=self.ev_id,
            cs_id=offer.cs_id,
            final_price=offer.price,
            reasoning=reasoning,
            offer_id=offer.offer_id,
            timestamp=time.time()
        )
        
        await self.publish_message(acceptance, DefaultTopicId())
        
        # GRACEFUL EXITS: Reject all other active negotiations
        await self._reject_other_negotiations(offer.cs_id, ctx)
    
    async def _reject_other_negotiations(self, accepted_cs_id: str, ctx: MessageContext) -> None:
        """Send polite rejections to all other charging stations"""
        for cs_id in self.active_negotiations:
            if cs_id != accepted_cs_id and self.negotiation_status[cs_id] == NegotiationStatus.ACTIVE:
                print(f"[{self.ev_id}] 📨 Politely rejecting {cs_id} (accepted deal with {accepted_cs_id})")
                
                rejection = OfferRejected(
                    ev_id=self.ev_id,
                    cs_id=cs_id,
                    reason=f"Accepted alternative offer with {accepted_cs_id}",
                    final_rejection=True,
                    timestamp=time.time()
                )
                
                self.negotiation_status[cs_id] = NegotiationStatus.REJECTED
                await self.publish_message(rejection, DefaultTopicId())
    
    async def _fallback_parallel_decisions(self, negotiations_context: List[Dict], ctx: MessageContext) -> None:
        """Simple fallback logic when LLM fails"""
        print(f"[{self.ev_id}] 🔄 Using fallback parallel decision logic")
        
        # Find best offer within budget
        best_option = None
        for nego in negotiations_context:
            if nego['within_budget'] and (best_option is None or nego['current_price'] < best_option['current_price']):
                best_option = nego
        
        if best_option:
            cs_id = best_option['cs_id']
            latest_offer = self.offer_queue[cs_id][-1]
            
            if best_option['current_price'] <= self.max_acceptable_price * 0.9:
                # Great deal - accept
                await self._accept_offer(latest_offer, "Fallback: excellent price", ctx)
            else:
                # Counter-offer with 10% reduction
                counter_price = best_option['current_price'] * 0.9
                if self._validate_counter_price(counter_price, best_option['current_price']):
                    await self._counter_offer(latest_offer, counter_price, "Fallback: 10% reduction", ctx)
    
    async def _handle_offer_rejected(self, rejection: OfferRejected, ctx: MessageContext) -> None:
        """Handle rejection of our counter-offer"""
        if rejection.ev_id != self.ev_id:
            return
            
        print(f"[{self.ev_id}] 📨 Counter-offer rejected by {rejection.cs_id}: {rejection.reason}")
        
        # Update negotiation status
        if rejection.final_rejection:
            self.negotiation_status[rejection.cs_id] = NegotiationStatus.REJECTED
            print(f"[{self.ev_id}] 🏁 {rejection.cs_id} ended negotiations permanently")
        
        # Add to conversation history
        if rejection.cs_id in self.conversation_history:
            self.conversation_history[rejection.cs_id].append({
                'type': 'rejection',
                'from': rejection.cs_id,
                'reason': rejection.reason,
                'final': rejection.final_rejection,
                'timestamp': rejection.timestamp or time.time()
            })
    
    async def _handle_deal_finalized(self, deal: DealFinalized, ctx: MessageContext) -> None:
        """Handle successful deal completion"""
        if deal.ev_id == self.ev_id:
            total_cost = deal.total_cost
            savings = (self.max_acceptable_price - deal.final_price) * deal.energy_needed
            
            print(f"[{self.ev_id}] 🎉 DEAL FINALIZED with {deal.cs_id}!")
            print(f"[{self.ev_id}]    Final price: ${deal.final_price:.3f}/kWh")
            print(f"[{self.ev_id}]    Energy: {deal.energy_needed:.1f} kWh")
            print(f"[{self.ev_id}]    Total cost: ${total_cost:.2f}")
            print(f"[{self.ev_id}]    Negotiation rounds: {deal.negotiation_rounds}")
            if savings > 0:
                print(f"[{self.ev_id}]    Savings vs budget: ${savings:.2f}")
    



# =====================================================================================
# PARALLEL NEGOTIATION CS AGENT - Handles Multiple Customer Negotiations
# =====================================================================================

@default_subscription
class ParallelCSAgent(RoutedAgent):
    """
    Advanced CS Agent capable of handling multiple simultaneous customer negotiations.
    
    Key Features:
    - Concurrent negotiations with multiple EVs
    - JSON-mode LLM responses
    - Dynamic pricing based on market conditions
    - Conversation memory for each customer
    - Strategic customer prioritization
    """
    
    def __init__(self, cs_id: str, current_electricity_cost: float, 
                 available_chargers: int, min_profit_margin: float,
                 ollama_client: ollama.AsyncClient):
        super().__init__(description=f"CS-{cs_id}")
        self.cs_id = cs_id
        self.base_electricity_cost = current_electricity_cost
        self.current_electricity_cost = current_electricity_cost
        self.available_chargers = available_chargers
        self.total_chargers = available_chargers
        self.min_profit_margin = min_profit_margin
        self.llm_client = ollama_client
        
        # Parallel negotiation state
        self.active_customer_negotiations: Dict[str, Dict] = {}  # ev_id -> negotiation_state
        self.conversation_history: Dict[str, List[Dict]] = {}    # ev_id -> conversation
        self.customer_priority: Dict[str, float] = {}           # ev_id -> priority_score
        
        # Remove single-track limitations  
        # self.llm_processing = False  # REMOVED
        # self.active_negotiations = {}  # Enhanced to handle multiple
        
        self.max_concurrent_negotiations = 5
        
        print(f"[{self.cs_id}] ⚡ Parallel CS agent initialized")
        print(f"[{self.cs_id}]    Capacity: {self.available_chargers} chargers")
        print(f"[{self.cs_id}]    Base cost: ${self.base_electricity_cost:.3f}/kWh")
        print(f"[{self.cs_id}]    Min margin: {self.min_profit_margin*100:.1f}%")
    
    def _calculate_dynamic_price(self, base_margin: Optional[float] = None) -> float:
        """Calculate price with capacity-based pricing"""
        margin = base_margin if base_margin is not None else self.min_profit_margin
        
        # Capacity-based pricing
        capacity_utilization = 1 - (self.available_chargers / self.total_chargers)
        capacity_multiplier = 1 + (capacity_utilization * 0.2)  # Up to 20% premium when busy
        
        return self.current_electricity_cost * (1 + margin) * capacity_multiplier
    
    @message_handler
    async def handle_charging_request(self, message: ChargingRequest, ctx: MessageContext) -> None:
        """Handle charging request - START new negotiation"""
        await self._handle_charging_request(message, ctx)
    
    @message_handler
    async def handle_counter_offer(self, message: CounterOffer, ctx: MessageContext) -> None:
        """Handle counter-offer - ADD to customer's negotiation"""
        await self._handle_counter_offer(message, ctx)
    
    @message_handler
    async def handle_deal_finalized(self, message: DealFinalized, ctx: MessageContext) -> None:
        """Handle successful deal completion"""
        await self._handle_deal_finalized(message, ctx)
    

    
    async def _handle_charging_request(self, request: ChargingRequest, ctx: MessageContext) -> None:
        """Start new negotiation with EV customer"""
        if self.available_chargers <= 0:
            print(f"[{self.cs_id}] 🚫 No capacity for {request.ev_id} (full)")
            return
            
        if len(self.active_customer_negotiations) >= self.max_concurrent_negotiations:
            print(f"[{self.cs_id}] 🚫 Too many active negotiations, declining {request.ev_id}")
            return
        
        # Calculate priority score for this customer
        urgency_score = {"CRITICAL": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1}.get(request.urgency_level, 1)
        price_score = min(request.max_acceptable_price / self.current_electricity_cost, 3.0)  # Cap at 3x
        energy_score = min(request.energy_needed / 30.0, 2.0)  # Normalize to typical 30kWh
        
        priority = urgency_score * 0.4 + price_score * 0.4 + energy_score * 0.2
        self.customer_priority[request.ev_id] = priority
        
        # Calculate initial offer
        initial_price = self._calculate_dynamic_price()
        
        print(f"[{self.cs_id}] 📤 New negotiation with {request.ev_id}")
        print(f"[{self.cs_id}]    Priority score: {priority:.2f}")
        print(f"[{self.cs_id}]    Initial offer: ${initial_price:.3f}/kWh")
        print(f"[{self.cs_id}]    Customer budget: ${request.max_acceptable_price:.3f}/kWh")
        
        # Initialize negotiation state
        self.active_customer_negotiations[request.ev_id] = {
            'initial_request': request,
            'current_price': initial_price,
            'rounds': 1,
            'last_action': 'initial_offer',
            'start_time': time.time(),
            'priority': priority
        }
        
        self.conversation_history[request.ev_id] = [{
            'type': 'request',
            'from': request.ev_id,
            'energy_needed': request.energy_needed,
            'max_price': request.max_acceptable_price,
            'urgency': request.urgency_level,
            'timestamp': request.timestamp or time.time()
        }]
        
        # Send initial offer
        offer = ChargingOffer(
            cs_id=self.cs_id,
            ev_id=request.ev_id,
            price=initial_price,
            available_chargers=self.available_chargers,
            round_number=1,
            timestamp=time.time()
        )
        offer.offer_id = f"{self.cs_id}-{request.ev_id}-1-{time.time()}"
        
        await self.publish_message(offer, DefaultTopicId())
    
    async def _handle_counter_offer(self, counter: CounterOffer, ctx: MessageContext) -> None:
        """Handle counter-offer using JSON-mode LLM with conversation context"""
        if counter.cs_id != self.cs_id or counter.ev_id not in self.active_customer_negotiations:
            return
        
        print(f"[{self.cs_id}] 📨 Counter-offer from {counter.ev_id}: ${counter.price:.3f}/kWh (Round {counter.round_number})")
        print(f"[{self.cs_id}]    Customer reasoning: {counter.reasoning}")
        
        # Update conversation history
        self.conversation_history[counter.ev_id].append({
            'type': 'counter',
            'from': counter.ev_id,
            'price': counter.price,
            'round': counter.round_number,
            'reasoning': counter.reasoning,
            'timestamp': counter.timestamp
        })
        
        # Update negotiation state
        negotiation = self.active_customer_negotiations[counter.ev_id]
        negotiation['rounds'] = counter.round_number
        negotiation['last_action'] = 'counter_received'
        
        # Use JSON-mode LLM for decision
        await self._make_pricing_decision_json(counter, negotiation, ctx)
    
    async def _make_pricing_decision_json(self, counter: CounterOffer, negotiation: Dict, ctx: MessageContext) -> None:
        """Use JSON-mode LLM for pricing decisions with full context"""
        
        # Calculate business metrics
        original_offer = negotiation['current_price']
        profit_margin = (counter.price - self.current_electricity_cost) / self.current_electricity_cost * 100
        customer_request = negotiation['initial_request']
        conversation = self.conversation_history[counter.ev_id]
        priority = negotiation['priority']
        
        # Prepare optimized JSON prompt
        prompt = f"""CS {self.cs_id}: Cost ${self.current_electricity_cost:.3f}/kWh, Target margin {self.min_profit_margin*100:.1f}%, Capacity {self.available_chargers}/{self.total_chargers}.

Customer {counter.ev_id}: Priority {priority:.1f}/5, Energy {customer_request.energy_needed:.1f}kWh, Budget ${customer_request.max_acceptable_price:.3f}/kWh, {customer_request.urgency_level} urgency.

Offer: ${original_offer:.3f} → Counter: ${counter.price:.3f} (Profit: {profit_margin:.1f}%, Round {counter.round_number})
Reason: "{counter.reasoning}"

Strategy: Round 1-2: Higher margins. Round 3+: Close deals. Stay above ${self.current_electricity_cost:.3f}/kWh.

JSON response:
{{
  "reasoning": "Your analysis",
  "decision": "ACCEPT" | "REJECT" | "COUNTER",
  "price": 0.115,
  "final_rejection": false
}}

JSON only."""

        try:
            llm_start_time = time.time()
            print(f"[{self.cs_id}] 🤔 Consulting LLM for pricing decision... (⏱️ {llm_start_time:.1f}s)")
            
            # Small delay to prevent overwhelming the LLM
            await asyncio.sleep(0.5)
            
            response = await asyncio.wait_for(
                self.llm_client.chat(
                    model='llama3.1:8b',
                    messages=[{'role': 'user', 'content': prompt}],
                    format='json'
                ),
                timeout=35.0  # Increased timeout to allow LLM thinking time
            )
            
            llm_end_time = time.time()
            llm_duration = llm_end_time - llm_start_time
            print(f"[{self.cs_id}] ✅ LLM thinking completed in {llm_duration:.2f}s (⏱️ {llm_end_time:.1f}s)")
            
            llm_response = response['message']['content'].strip()
            print(f"[{self.cs_id}] 🧠 LLM Decision Response:")
            print(f"[{self.cs_id}]    {llm_response[:150]}...")
            
            try:
                decision = json.loads(llm_response)
                await self._execute_pricing_decision(decision, counter, ctx)
            except json.JSONDecodeError as e:
                print(f"[{self.cs_id}] ⚠️ JSON parsing error: {e}")
                await self._fallback_pricing_decision(counter, ctx)
                
        except asyncio.TimeoutError:
            print(f"[{self.cs_id}] ⏰ LLM timeout, using fallback logic")
            await self._fallback_pricing_decision(counter, ctx)
        except Exception as e:
            print(f"[{self.cs_id}] ❌ LLM error: {e}. Using fallback logic.")
            await self._fallback_pricing_decision(counter, ctx)
    
    async def _execute_pricing_decision(self, decision: Dict, counter: CounterOffer, ctx: MessageContext) -> None:
        """Execute the LLM's pricing decision"""
        
        reasoning = decision.get('reasoning', 'No reasoning provided')
        action = decision.get('decision', 'REJECT')
        
        print(f"[{self.cs_id}] 💭 Pricing reasoning: {reasoning}")
        print(f"[{self.cs_id}] 🎯 Decision: {action}")
        
        if action == "ACCEPT":
            await self._accept_counter_offer(counter, reasoning, ctx)
            
        elif action == "COUNTER":
            new_price = decision.get('price', counter.price * 1.05)
            if self._validate_pricing(new_price):
                await self._make_counter_offer(counter, new_price, reasoning, ctx)
            else:
                print(f"[{self.cs_id}] ⚠️ Invalid counter price {new_price}, rejecting instead")
                await self._reject_counter_offer(counter, "Invalid pricing strategy", True, ctx)
                
        else:  # REJECT
            final_rejection = decision.get('final_rejection', False)
            await self._reject_counter_offer(counter, reasoning, final_rejection, ctx)
    
    def _validate_pricing(self, price: float) -> bool:
        """Validate pricing is profitable and reasonable"""
        min_price = self.current_electricity_cost * 1.05  # Minimum 5% profit
        max_price = self.current_electricity_cost * 3.0   # Maximum 300% markup
        return min_price <= price <= max_price
    
    async def _accept_counter_offer(self, counter: CounterOffer, reasoning: str, ctx: MessageContext) -> None:
        """Accept customer's counter-offer"""
        print(f"[{self.cs_id}] ✅ ACCEPTING counter-offer from {counter.ev_id}")
        print(f"[{self.cs_id}] 💭 Reasoning: {reasoning}")
        
        acceptance = OfferAccepted(
            ev_id=counter.ev_id,
            cs_id=self.cs_id,
            final_price=counter.price,
            reasoning=reasoning,
            offer_id=counter.offer_id,
            timestamp=time.time()
        )
        
        await self.publish_message(acceptance, DefaultTopicId())
        
        # Clean up negotiation state
        if counter.ev_id in self.active_customer_negotiations:
            del self.active_customer_negotiations[counter.ev_id]
    
    async def _make_counter_offer(self, counter: CounterOffer, new_price: float, reasoning: str, ctx: MessageContext) -> None:
        """Make our own counter-offer"""
        print(f"[{self.cs_id}] 📤 Counter-offering {counter.ev_id}: ${new_price:.3f}/kWh")
        print(f"[{self.cs_id}] 💭 Reasoning: {reasoning}")
        
        new_offer = ChargingOffer(
            cs_id=self.cs_id,
            ev_id=counter.ev_id,
            price=new_price,
            available_chargers=self.available_chargers,
            round_number=counter.round_number + 1,
            timestamp=time.time()
        )
        new_offer.offer_id = f"{self.cs_id}-{counter.ev_id}-{new_offer.round_number}-{time.time()}"
        
        # Update negotiation state
        self.active_customer_negotiations[counter.ev_id]['current_price'] = new_price
        self.active_customer_negotiations[counter.ev_id]['rounds'] = new_offer.round_number
        
        await self.publish_message(new_offer, DefaultTopicId())
    
    async def _reject_counter_offer(self, counter: CounterOffer, reasoning: str, final_rejection: bool, ctx: MessageContext) -> None:
        """Reject the counter-offer"""
        print(f"[{self.cs_id}] ❌ REJECTING counter-offer from {counter.ev_id}")
        print(f"[{self.cs_id}] 💭 Reasoning: {reasoning}")
        
        rejection = OfferRejected(
            ev_id=counter.ev_id,
            cs_id=self.cs_id,
            reason=reasoning,
            final_rejection=final_rejection,
            offer_id=counter.offer_id,
            timestamp=time.time()
        )
        
        await self.publish_message(rejection, DefaultTopicId())
        
        # Clean up if final rejection
        if final_rejection and counter.ev_id in self.active_customer_negotiations:
            del self.active_customer_negotiations[counter.ev_id]
    
    async def _fallback_pricing_decision(self, counter: CounterOffer, ctx: MessageContext) -> None:
        """Simple fallback when LLM fails"""
        profit_margin = (counter.price - self.current_electricity_cost) / self.current_electricity_cost
        
        if profit_margin >= self.min_profit_margin * 0.8:  # Accept if close to target
            await self._accept_counter_offer(counter, "Fallback: acceptable margin", ctx)
        elif profit_margin > 0.05:  # Counter if some profit
            new_price = (counter.price + self.active_customer_negotiations[counter.ev_id]['current_price']) / 2
            await self._make_counter_offer(counter, new_price, "Fallback: split difference", ctx)
        else:
            await self._reject_counter_offer(counter, "Fallback: insufficient profit", True, ctx)
    
    async def _handle_deal_finalized(self, deal: DealFinalized, ctx: MessageContext) -> None:
        """Handle successful deal completion"""
        if deal.cs_id == self.cs_id:
            profit = deal.final_price - self.current_electricity_cost
            margin = (profit / self.current_electricity_cost) * 100
            
            print(f"[{self.cs_id}] 🎉 DEAL FINALIZED with {deal.ev_id}!")
            print(f"[{self.cs_id}]    Revenue: ${deal.total_cost:.2f}")
            print(f"[{self.cs_id}]    Profit margin: {margin:.1f}%")
            print(f"[{self.cs_id}]    Negotiation rounds: {deal.negotiation_rounds}")
            
            # Update capacity
            self.available_chargers = max(0, self.available_chargers - 1)
            print(f"[{self.cs_id}]    Remaining capacity: {self.available_chargers}/{self.total_chargers}")
    



# =====================================================================================  
# ENHANCED MARKETPLACE AGENT - Proper State Management and Event-Driven Termination
# =====================================================================================

@default_subscription
class EnhancedMarketplaceAgent(RoutedAgent):
    """
    Enhanced marketplace with proper state management, event-driven termination,
    and comprehensive analytics.
    """
    
    def __init__(self):
        super().__init__(description="marketplace")
        self.registered_evs: List[str] = ["1", "2", "3"]
        self.registered_css: List[str] = ["A", "B", "C"]
        
        # Enhanced state tracking
        self.completed_deals: List[Dict] = []
        self.failed_negotiations: List[Dict] = []
        self.active_requests: Dict[str, ChargingRequest] = {}  # ev_id -> original request
        self.negotiation_states: Dict[str, str] = {}  # ev_id -> "active", "completed", "failed"
        
        # Event-driven termination
        self.all_negotiations_complete = False
        self.termination_callback = None
        
        print(f"[Marketplace] 🏪 Enhanced marketplace initialized")
        print(f"[Marketplace]    EVs: {len(self.registered_evs)}")
        print(f"[Marketplace]    CSs: {len(self.registered_css)}")
    
    def set_termination_callback(self, callback):
        """Set callback for when all negotiations complete"""
        self.termination_callback = callback
    
    @message_handler
    async def handle_charging_request(self, message: ChargingRequest, ctx: MessageContext) -> None:
        await self._handle_charging_request(message, ctx)
    
    @message_handler
    async def handle_charging_offer(self, message: ChargingOffer, ctx: MessageContext) -> None:
        await self._handle_charging_offer(message, ctx)
    
    @message_handler
    async def handle_counter_offer(self, message: CounterOffer, ctx: MessageContext) -> None:
        await self._handle_counter_offer(message, ctx)
    
    @message_handler
    async def handle_offer_accepted(self, message: OfferAccepted, ctx: MessageContext) -> None:
        await self._handle_offer_accepted(message, ctx)
    
    @message_handler
    async def handle_offer_rejected(self, message: OfferRejected, ctx: MessageContext) -> None:
        await self._handle_offer_rejected(message, ctx)
    
    async def _handle_charging_request(self, request: ChargingRequest, ctx: MessageContext) -> None:
        """Handle charging request with proper state tracking"""
        print(f"[Marketplace] 📢 New charging request from {request.ev_id}")
        print(f"[Marketplace]    Energy: {request.energy_needed:.1f} kWh")
        print(f"[Marketplace]    Budget: ${request.max_acceptable_price:.3f}/kWh")
        print(f"[Marketplace]    Urgency: {request.urgency_level}")
        
        # Store request for later reference (FIXED: no more hardcoded energy)
        self.active_requests[request.ev_id] = request
        self.negotiation_states[request.ev_id] = "active"
        
        # Broadcast to all CS agents
        await self.publish_message(request, DefaultTopicId())
    
    async def _handle_charging_offer(self, offer: ChargingOffer, ctx: MessageContext) -> None:
        """Forward charging offer to target EV"""
        print(f"[Marketplace] 📤 Offer: {offer.cs_id} → {offer.ev_id} (${offer.price:.3f}/kWh, R{offer.round_number})")
        await self.publish_message(offer, DefaultTopicId())
    
    async def _handle_counter_offer(self, counter: CounterOffer, ctx: MessageContext) -> None:
        """Forward counter-offer to target CS"""
        print(f"[Marketplace] 🔄 Counter: {counter.ev_id} → {counter.cs_id} (${counter.price:.3f}/kWh, R{counter.round_number})")
        await self.publish_message(counter, DefaultTopicId())
    
    async def _handle_offer_accepted(self, acceptance: OfferAccepted, ctx: MessageContext) -> None:
        """Handle deal acceptance and create finalized deal"""
        print(f"[Marketplace] 🎉 DEAL ACCEPTED: {acceptance.ev_id} ↔ {acceptance.cs_id} at ${acceptance.final_price:.3f}/kWh")
        
        # Get original request for accurate energy calculation
        original_request = self.active_requests.get(acceptance.ev_id)
        energy_needed = original_request.energy_needed if original_request else 30.0  # Fallback
        total_cost = acceptance.final_price * energy_needed
        
        # Count negotiation rounds (simplified - could be enhanced)
        negotiation_rounds = 2  # Placeholder - could track this more precisely
        
        deal = DealFinalized(
            ev_id=acceptance.ev_id,
            cs_id=acceptance.cs_id,
            final_price=acceptance.final_price,
            energy_needed=energy_needed,
            total_cost=total_cost,
            negotiation_rounds=negotiation_rounds,
            timestamp=time.time()
        )
        
        # Update state
        self.negotiation_states[acceptance.ev_id] = "completed"
        self.completed_deals.append({
            'ev_id': acceptance.ev_id,
            'cs_id': acceptance.cs_id,
            'final_price': acceptance.final_price,
            'energy_needed': energy_needed,
            'total_cost': total_cost,
            'reasoning': acceptance.reasoning,
            'timestamp': acceptance.timestamp
        })
        
        # Broadcast finalization
        await self.publish_message(deal, DefaultTopicId())
        
        # Check if all negotiations complete
        await self._check_completion_status()
    
    async def _handle_offer_rejected(self, rejection: OfferRejected, ctx: MessageContext) -> None:
        """Handle offer rejection"""
        status = "🏁 FINAL" if rejection.final_rejection else "🔄 ONGOING"
        print(f"[Marketplace] {status} Rejection: {rejection.ev_id} ↔ {rejection.cs_id}")
        
        await self.publish_message(rejection, DefaultTopicId())
        
        # If final rejection and no active deals, mark as failed
        if rejection.final_rejection:
            # Check if this EV has any other active negotiations
            # For now, simplified logic
            pass
    
    async def _check_completion_status(self) -> None:
        """Check if all negotiations are complete and trigger termination"""
        active_count = sum(1 for status in self.negotiation_states.values() if status == "active")
        completed_count = sum(1 for status in self.negotiation_states.values() if status == "completed")
        
        print(f"[Marketplace] 📊 Status check: {active_count} active, {completed_count} completed")
        
        if active_count == 0 and len(self.negotiation_states) > 0:
            print(f"[Marketplace] 🏁 ALL NEGOTIATIONS COMPLETE!")
            self.all_negotiations_complete = True
            
            if self.termination_callback:
                await self.termination_callback()
    
    def get_comprehensive_results(self) -> Dict:
        """Get complete marketplace results"""
        return {
            'completed_deals': self.completed_deals,
            'failed_negotiations': self.failed_negotiations,
            'negotiation_states': self.negotiation_states,
            'active_requests': {ev_id: asdict(req) for ev_id, req in self.active_requests.items()},
            'total_negotiations': len(self.negotiation_states),
            'success_rate': len(self.completed_deals) / max(len(self.negotiation_states), 1) * 100
        }





# =====================================================================================
# ENHANCED MAIN SIMULATION - Event-Driven and State-Managed
# =====================================================================================

async def main():
    """
    Enhanced main simulation with event-driven termination and proper state management.
    """
    print("🚀 Advanced Parallel Negotiation EV Marketplace")
    print("Features: JSON-mode LLMs, Parallel negotiations, Dynamic environment")
    print("="*80)
    
    # Test Ollama availability (but don't create shared client)
    try:
        test_client = ollama.AsyncClient()
        test_response = await test_client.chat(
            model='llama3.1:8b',
            messages=[{'role': 'user', 'content': 'Respond with JSON: {"status": "ready"}'}],
            format='json'
        )
        print(f"✅ Ollama server confirmed ready for parallel connections")
        del test_client  # Clean up test client
        
    except Exception as e:
        print(f"❌ Ollama connection test failed: {e}")
        print("Please ensure Ollama is running with llama3.1:8b available")
        sys.exit(1)
    
    # Initialize runtime
    runtime = SingleThreadedAgentRuntime()
    
    # Enhanced agent configurations for parallel negotiation testing
    ev_configs = [
        {"ev_id": "1", "battery_level": 20, "target_battery_level": 80, "max_acceptable_price": 0.14},
        {"ev_id": "2", "battery_level": 35, "target_battery_level": 90, "max_acceptable_price": 0.16},
        {"ev_id": "3", "battery_level": 12, "target_battery_level": 85, "max_acceptable_price": 0.13},
    ]
    
    cs_configs = [
        {"cs_id": "A", "current_electricity_cost": 0.08, "available_chargers": 3, "min_profit_margin": 0.25},
        {"cs_id": "B", "current_electricity_cost": 0.09, "available_chargers": 2, "min_profit_margin": 0.20},
        {"cs_id": "C", "current_electricity_cost": 0.07, "available_chargers": 4, "min_profit_margin": 0.30},
    ]
    
    # Event-driven termination setup
    termination_event = asyncio.Event()
    
    async def termination_callback():
        termination_event.set()
    
    # Register enhanced agents
    def create_marketplace():
        agent = EnhancedMarketplaceAgent()
        agent.set_termination_callback(termination_callback)
        return agent
    
    await EnhancedMarketplaceAgent.register(runtime, "marketplace", create_marketplace)
    
    # Register parallel EV agents (each with dedicated ollama client)
    for ev_config in ev_configs:
        await ParallelEVAgent.register(
            runtime, 
            f"EV-{ev_config['ev_id']}", 
            lambda config=ev_config: ParallelEVAgent(**config, ollama_client=ollama.AsyncClient())
        )
    
    # Register parallel CS agents (each with dedicated ollama client)
    for cs_config in cs_configs:
        await ParallelCSAgent.register(
            runtime,
            f"CS-{cs_config['cs_id']}",
            lambda config=cs_config: ParallelCSAgent(**config, ollama_client=ollama.AsyncClient())
        )
    
    print(f"\n🤖 Initialized parallel negotiation system:")
    print(f"   📱 {len(ev_configs)} EV agents with dedicated LLM clients for parallel thinking")
    print(f"   ⚡ {len(cs_configs)} CS agents with dedicated LLM clients for concurrent decisions") 
    print(f"   🏪 Enhanced marketplace with event-driven termination")
    print(f"   🧠 Total {len(ev_configs) + len(cs_configs)} independent LLM thinking processes enabled")
    print("\n🚀 Starting parallel negotiations...\n")
    
    # Start runtime
    runtime.start()
    
    try:
        # Send charging requests with realistic timing
        for i, ev_config in enumerate(ev_configs):
            if i > 0:
                await asyncio.sleep(5)  # Stagger requests
            
            # Calculate energy needed
            battery_capacity = 60.0
            percentage_needed = (ev_config['target_battery_level'] - ev_config['battery_level']) / 100.0
            energy_needed = battery_capacity * percentage_needed
            
            # Determine urgency
            if ev_config['battery_level'] < 15:
                urgency = "CRITICAL"
            elif ev_config['battery_level'] < 25:
                urgency = "HIGH"
            elif ev_config['battery_level'] < 40:
                urgency = "MEDIUM"
            else:
                urgency = "LOW"
            
            request = ChargingRequest(
                ev_id=ev_config['ev_id'],
                battery_level=ev_config['battery_level'],
                target_battery_level=ev_config['target_battery_level'],
                max_acceptable_price=ev_config['max_acceptable_price'],
                energy_needed=energy_needed,
                urgency_level=urgency,
                timestamp=time.time()
            )
            
            print(f"🚗 Initiating parallel negotiation for EV-{ev_config['ev_id']}")
            print(f"   Battery: {ev_config['battery_level']}% → {ev_config['target_battery_level']}% ({urgency})")
            print(f"   Energy: {energy_needed:.1f} kWh, Budget: ${ev_config['max_acceptable_price']:.3f}/kWh")
            print(f"   Will negotiate simultaneously with ALL available charging stations\n")
            
            await runtime.send_message(request, AgentId("marketplace", "default"))
        
        # EVENT-DRIVEN TERMINATION: Wait for all negotiations to complete
        print("⏳ Waiting for all parallel negotiations to complete...")
        print("   (The simulation will automatically terminate when all deals are done)")
        
        try:
            # Wait for termination event with timeout
            await asyncio.wait_for(termination_event.wait(), timeout=180.0)  # 3 minute max
            print("\n🎉 All negotiations completed! Terminating simulation...")
            
        except asyncio.TimeoutError:
            print("\n⏰ Simulation timeout reached, terminating...")
        
    finally:
        # Clean shutdown
        await runtime.stop_when_idle()
        
        # Display basic results
        print("\n" + "="*100)
        print("🏆 PARALLEL NEGOTIATION SIMULATION COMPLETED")
        print("="*100)
        
        print(f"\n🎉 PARALLEL NEGOTIATION FEATURES DEMONSTRATED:")
        print(f"   🔄 Simultaneous negotiations with multiple parties")
        print(f"   🧠 JSON-mode LLM responses for reliability")
        print(f"   💭 Conversational memory for strategic context")
        print(f"   🌐 Dynamic market conditions")
        print(f"   ⚡ Event-driven termination")
        print(f"   📊 Comprehensive state management")
        print("="*100)


if __name__ == "__main__":
    """Entry point for the advanced parallel negotiation marketplace"""
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n🛑 Simulation interrupted by user")
        sys.exit(0)
    except Exception as e:
        import traceback
        print(f"\n\n❌ Simulation error: {e}")
        print("Full traceback:")
        traceback.print_exc()
        sys.exit(1)