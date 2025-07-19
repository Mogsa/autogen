#!/usr/bin/env python3
"""
Multi-Agent EV Charging Marketplace with LLM-Powered Thinking Loop

This simulation creates 6 independent LLM agents (3 EVs + 3 CSs) that negotiate
through a shared JSON state file. Each agent operates in a continuous "thinking loop":
- Read market_state.json  
- Make LLM-powered decision
- Send update to MarketplaceAgent
- MarketplaceAgent updates state file
- File change triggers other agents to think

Key features:
- Current/previous value tracking for negotiation history
- Asynchronous agent thinking loops  
- File-based state management
- Real-time market coordination
"""

import asyncio
import json
import time
import os
import tempfile
from datetime import datetime
from typing import Dict, List, Optional, Any
import ollama
from dataclasses import dataclass, asdict


# =====================================================================================
# DATA STRUCTURES
# =====================================================================================

@dataclass
class OfferValue:
    """Represents a current or previous offer value"""
    price: float
    round: int

@dataclass
class Offer:
    """CS offer with current and previous values"""
    current: OfferValue
    previous: Optional[OfferValue] = None

@dataclass
class CounterOffer:
    """EV counter-offer with current and previous values"""
    target_cs_id: str
    current: OfferValue
    previous: Optional[OfferValue] = None

@dataclass
class Deal:
    """Completed charging deal"""
    ev_id: str
    cs_id: str
    price: float
    status: str = "completed"
    timestamp: str = ""

@dataclass
class MarketState:
    """Complete market state"""
    last_updated: str
    status: str  # NEGOTIATING, COMPLETE
    offers: Dict[str, Dict]  # CS-A: {current: {...}, previous: {...}}
    counter_offers: Dict[str, Dict]  # EV-1: {target_cs_id: "CS-A", current: {...}, previous: {...}}
    deals: List[Dict]

@dataclass
class AgentDecision:
    """Decision from an agent"""
    agent_id: str
    action: str  # ACCEPT, COUNTER, WAIT, HOLD
    target_id: Optional[str] = None
    price: Optional[float] = None
    timestamp: str = ""


# =====================================================================================
# MARKETPLACE AGENT - State Management & File I/O
# =====================================================================================

class MarketplaceAgent:
    """
    Central coordinator that manages market_state.json file.
    Receives decisions from agents and atomically updates the state.
    """
    
    def __init__(self, agents_queue: asyncio.Queue, state_file: str = "market_state.json"):
        self.state_file = state_file
        self.agents_queue = agents_queue  # Use the shared queue from main()
        self.round_counter = {}  # Track rounds per agent
        
        # Initialize market state
        self._initialize_market_state()
    
    def _initialize_market_state(self):
        """Create initial market state file"""
        initial_state = MarketState(
            last_updated=datetime.now().isoformat(),
            status="NEGOTIATING",
            offers={},
            counter_offers={},
            deals=[]
        )
        self._write_state_atomic(asdict(initial_state))
        print(f"[Marketplace] 🏪 Initialized market state: {self.state_file}")
    
    def _write_state_atomic(self, state_dict: Dict):
        """Atomically write state to file to prevent corruption during concurrent access"""
        # Write to temporary file first, then atomic rename
        temp_file = f"{self.state_file}.tmp"
        try:
            with open(temp_file, 'w') as f:
                json.dump(state_dict, f, indent=2)
            
            # Atomic rename (works on all platforms)
            if os.name == 'nt':  # Windows
                if os.path.exists(self.state_file):
                    os.remove(self.state_file)
            os.rename(temp_file, self.state_file)
            
        except Exception as e:
            if os.path.exists(temp_file):
                os.remove(temp_file)
            raise e
    
    def read_market_state(self) -> Dict:
        """Read current market state from file"""
        try:
            with open(self.state_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"[Marketplace] ⚠️ State file not found, reinitializing...")
            self._initialize_market_state()
            return self.read_market_state()
    
    async def process_agent_decision(self, decision: AgentDecision):
        """Process a decision from an agent and update market state"""
        print(f"[Marketplace] 🔍 DEBUG: Received decision from {decision.agent_id}: {decision.action}")
        state = self.read_market_state()
        
        print(f"[Marketplace] 📝 Processing {decision.action} from {decision.agent_id}")
        
        # Only increment round counter for COUNTER actions (actual price offers)
        current_round = self.round_counter.get(decision.agent_id, 0)
        if decision.action == "COUNTER":
            if decision.agent_id not in self.round_counter:
                self.round_counter[decision.agent_id] = 1
            else:
                self.round_counter[decision.agent_id] += 1
            current_round = self.round_counter[decision.agent_id]
            print(f"[Marketplace] 📊 {decision.agent_id} pricing round: {current_round}")
        else:
            print(f"[Marketplace] ⏸️ {decision.action} action - no round increment")
        
        # Update state based on decision type
        if decision.agent_id.startswith("CS-"):
            await self._handle_cs_decision(state, decision, current_round)
        elif decision.agent_id.startswith("EV-"):
            await self._handle_ev_decision(state, decision, current_round)
        
        # Check for termination condition
        if len(state["deals"]) >= 3:  # All 3 EVs have deals
            state["status"] = "COMPLETE"
            print(f"[Marketplace] 🎉 All negotiations complete!")
        
        # Update timestamp and write
        state["last_updated"] = datetime.now().isoformat()
        print(f"[Marketplace] 🔄 About to write updated state to JSON...")
        self._write_state_atomic(state)
        print(f"[Marketplace] ✅ JSON file updated successfully!")
        
        print(f"[Marketplace] ✅ State updated - {len(state['deals'])}/3 deals completed")
        
        # Print current market state for visibility
        print(f"[Marketplace] 📊 CURRENT MARKET STATE:")
        print(f"   Offers: {len(state.get('offers', {}))}")
        for cs_id, offer in state.get('offers', {}).items():
            current_price = offer['current']['price']
            round_num = offer['current']['round']
            print(f"     {cs_id}: ${current_price:.3f}/kWh (Round {round_num})")
        
        print(f"   Counter-offers: {len(state.get('counter_offers', {}))}")
        for ev_id, counter in state.get('counter_offers', {}).items():
            target = counter['target_cs_id']
            current_price = counter['current']['price']
            round_num = counter['current']['round']
            print(f"     {ev_id} → {target}: ${current_price:.3f}/kWh (Round {round_num})")
        
        print(f"   Deals: {len(state.get('deals', []))}")
        for deal in state.get('deals', []):
            print(f"     {deal['ev_id']} ↔ {deal['cs_id']}: ${deal['price']:.3f}/kWh")
        print()
    
    async def _handle_cs_decision(self, state: Dict, decision: AgentDecision, round_num: int):
        """Handle decision from a CS agent"""
        if decision.action == "COUNTER":
            # CS is making a new offer or counter-offer
            if decision.price is None:
                print(f"[Marketplace] ⚠️ CS {decision.agent_id} sent COUNTER without price")
                return
            current_offer = OfferValue(price=decision.price, round=round_num)
            
            # Preserve previous offer if it exists
            previous_offer = None
            if decision.agent_id in state["offers"]:
                previous_offer = state["offers"][decision.agent_id]["current"]
            
            state["offers"][decision.agent_id] = {
                "current": asdict(current_offer),
                "previous": previous_offer
            }
            
        elif decision.action == "ACCEPT":
            # CS accepting an EV's counter-offer
            if decision.target_id is None:
                print(f"[Marketplace] ⚠️ CS {decision.agent_id} sent ACCEPT without target_id")
                return
            if decision.target_id in state["counter_offers"]:
                counter_offer = state["counter_offers"][decision.target_id]
                deal = Deal(
                    ev_id=decision.target_id,
                    cs_id=decision.agent_id,
                    price=counter_offer["current"]["price"],
                    timestamp=datetime.now().isoformat()
                )
                state["deals"].append(asdict(deal))
                
                # Remove completed negotiations
                if decision.target_id in state["counter_offers"]:
                    del state["counter_offers"][decision.target_id]
                if decision.agent_id in state["offers"]:
                    del state["offers"][decision.agent_id]
    
    async def _handle_ev_decision(self, state: Dict, decision: AgentDecision, round_num: int):
        """Handle decision from an EV agent"""
        if decision.action == "COUNTER":
            # EV making a counter-offer
            if decision.price is None:
                print(f"[Marketplace] ⚠️ EV {decision.agent_id} sent COUNTER without price")
                return
            current_counter = OfferValue(price=decision.price, round=round_num)
            
            # Preserve previous counter-offer if it exists  
            previous_counter = None
            if decision.agent_id in state["counter_offers"]:
                previous_counter = state["counter_offers"][decision.agent_id]["current"]
            
            state["counter_offers"][decision.agent_id] = {
                "target_cs_id": decision.target_id,
                "current": asdict(current_counter),
                "previous": previous_counter
            }
            
        elif decision.action == "ACCEPT":
            # EV accepting a CS offer
            if decision.target_id is None:
                print(f"[Marketplace] ⚠️ EV {decision.agent_id} sent ACCEPT without target_id")
                return
            if decision.target_id in state["offers"]:
                offer = state["offers"][decision.target_id]
                deal = Deal(
                    ev_id=decision.agent_id,
                    cs_id=decision.target_id,
                    price=offer["current"]["price"],
                    timestamp=datetime.now().isoformat()
                )
                state["deals"].append(asdict(deal))
                
                # Remove completed negotiations
                if decision.agent_id in state["counter_offers"]:
                    del state["counter_offers"][decision.agent_id]
                if decision.target_id in state["offers"]:
                    del state["offers"][decision.target_id]
    
    async def run(self):
        """Main marketplace coordination loop"""
        print(f"[Marketplace] 🚀 Starting coordination loop...")
        
        while True:
            try:
                # Wait for agent decisions
                print(f"[Marketplace] ⏳ Waiting for agent decisions...")
                decision = await self.agents_queue.get()
                print(f"[Marketplace] 📨 Received decision from {decision.agent_id}: {decision.action}")
                await self.process_agent_decision(decision)
                
                # Check termination
                state = self.read_market_state()
                if state["status"] == "COMPLETE":
                    print(f"[Marketplace] 🏁 Marketplace shutting down - all deals completed")
                    break
                    
            except Exception as e:
                print(f"[Marketplace] ❌ Error processing decision: {e}")
                await asyncio.sleep(1)


# =====================================================================================
# EV AGENT - LLM-Powered Electric Vehicle  
# =====================================================================================

class EVAgent:
    """
    Electric Vehicle agent that uses LLM to make charging negotiation decisions.
    Operates in continuous thinking loop: read state → think → decide → update.
    """
    
    def __init__(self, ev_id: str, max_acceptable_price: float, energy_needed: float, 
                 marketplace_queue: asyncio.Queue):
        self.ev_id = ev_id
        self.max_acceptable_price = max_acceptable_price
        self.energy_needed = energy_needed
        self.marketplace_queue = marketplace_queue
        self.llm_client = ollama.AsyncClient()
        self.has_deal = False
        self.thinking_delay = 1.0  # Base thinking time
        self.negotiation_rounds = 0  # Track how many rounds we've negotiated
        
        print(f"[{self.ev_id}] 🚗 EV initialized - Budget: ${max_acceptable_price:.3f}/kWh, Energy: {energy_needed:.1f} kWh")
        print(f"[{self.ev_id}] 📋 Strategy: Negotiate for 3 rounds, then accept best available offer")
    
    async def read_market_state(self) -> Dict:
        """Read current market state from file"""
        try:
            with open("market_state.json", 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {"offers": {}, "counter_offers": {}, "deals": []}
    
    async def think_and_decide(self, market_state: Dict) -> Optional[AgentDecision]:
        """Use LLM to analyze market and make decision"""
        # Count actual COUNTER offers made so far (from market state or internal tracking)
        actual_counter_rounds = self.negotiation_rounds  # We'll increment this only when making COUNTER
        print(f"[{self.ev_id}] 🤔 Thinking about market state... (Made {actual_counter_rounds} counter-offers so far)")
        
        # After 3 actual counter-offers, accept the best available offer
        if actual_counter_rounds >= 3:
            return await self._accept_best_offer(market_state)
        
        # Get available stations for validation
        available_stations = list(market_state.get("offers", {}).keys())
        stations_str = ", ".join(available_stations) if available_stations else "None currently offering"
        
        # Create LLM prompt with market analysis
        prompt = f"""You are {self.ev_id} (Electric Vehicle). Your MAXIMUM acceptable price is ${self.max_acceptable_price:.3f}/kWh.
Energy needed: {self.energy_needed:.1f} kWh
Counter-offers made so far: {actual_counter_rounds}/3 (WAIT/HOLD actions don't count as rounds)

AVAILABLE CHARGING STATIONS: {stations_str}

Current market state:
{json.dumps(market_state, indent=2)}

CRITICAL RULES:
1. COUNTER-OFFERS must be LOWER than the station's current offer (you want to pay LESS)
2. ONLY reference existing stations: {available_stations}
3. Your counter-offer must be reasonable (not extremely low)
4. Strategy: Start aggressive, become more reasonable as rounds progress

NEGOTIATION STRATEGY:
- Counter-offers 1-2: Counter-offer 5-15% BELOW their current price (be aggressive)
- Counter-offer 3: This is your last chance before forced acceptance
- After 3 counter-offers: Must accept the best available offer

ANALYSIS REQUIRED:
- Which stations have the lowest current offers?
- Are any prices trending down (previous > current)?
- Which offers are within your ${self.max_acceptable_price:.3f}/kWh budget?

RESPOND WITH VALID JSON ONLY (no explanations, no other text):

For counter-offer:
{{"action": "COUNTER", "target_cs_id": "CS-A", "price": 0.115}}

For acceptance:
{{"action": "ACCEPT", "target_cs_id": "CS-B"}}

For waiting:
{{"action": "WAIT"}}

IMPORTANT: Response must be valid JSON starting with {{ and ending with }}"""

        try:
            response = await asyncio.wait_for(
                self.llm_client.chat(
                    model='llama3.1:8b',
                    messages=[{'role': 'user', 'content': prompt}],
                    format='json'
                ),
                timeout=15.0
            )
            
            llm_response = response['message']['content'].strip()
            print(f"[{self.ev_id}] 🔍 Raw LLM response: {llm_response[:100]}...")
            
            decision_data = json.loads(llm_response)
            action = decision_data.get('action', 'UNKNOWN')
            print(f"[{self.ev_id}] 💭 LLM decided: {action}")
            
            # Validate action is one of the expected values
            if action not in ["COUNTER", "ACCEPT", "WAIT"]:
                print(f"[{self.ev_id}] ⚠️ Invalid LLM action '{action}', treating as WAIT")
                return None
            
            # Convert to AgentDecision with validation
            if decision_data.get("action") == "COUNTER":
                target_cs = decision_data.get("target_cs_id")
                proposed_price = decision_data.get("price")
                
                # Re-read fresh market state for validation
                fresh_market_state = await self.read_market_state()
                
                # Validate target CS exists
                if target_cs not in fresh_market_state.get("offers", {}):
                    print(f"[{self.ev_id}] ⚠️ LLM referenced non-existent station {target_cs}, waiting instead")
                    print(f"[{self.ev_id}] 🔍 Available stations: {list(fresh_market_state.get('offers', {}).keys())}")
                    return None
                
                # Validate counter-offer is lower than current CS offer
                current_cs_price = fresh_market_state["offers"][target_cs]["current"]["price"]
                if proposed_price >= current_cs_price:
                    print(f"[{self.ev_id}] ⚠️ LLM counter-offer ${proposed_price:.3f} not lower than CS offer ${current_cs_price:.3f}, waiting instead")
                    return None
                
                # Increment our counter since we're making an actual COUNTER offer
                self.negotiation_rounds += 1
                print(f"[{self.ev_id}] 📤 Making counter-offer #{self.negotiation_rounds}")
                
                return AgentDecision(
                    agent_id=self.ev_id,
                    action="COUNTER",
                    target_id=target_cs,
                    price=proposed_price,
                    timestamp=datetime.now().isoformat()
                )
                
            elif decision_data.get("action") == "ACCEPT":
                target_cs = decision_data.get("target_cs_id")
                
                # Re-read fresh market state for validation
                fresh_market_state = await self.read_market_state()
                
                # Validate target CS exists
                if target_cs not in fresh_market_state.get("offers", {}):
                    print(f"[{self.ev_id}] ⚠️ LLM tried to accept non-existent station {target_cs}, waiting instead")
                    print(f"[{self.ev_id}] 🔍 Available stations: {list(fresh_market_state.get('offers', {}).keys())}")
                    return None
                
                return AgentDecision(
                    agent_id=self.ev_id,
                    action="ACCEPT",
                    target_id=target_cs,
                    timestamp=datetime.now().isoformat()
                )
            else:
                # WAIT - no decision to send
                return None
                
        except asyncio.TimeoutError:
            print(f"[{self.ev_id}] ⏰ LLM thinking timeout, waiting...")
            return None
        except Exception as e:
            print(f"[{self.ev_id}] ❌ LLM error: {e}, waiting...")
            return None
    
    async def _accept_best_offer(self, market_state: Dict) -> Optional[AgentDecision]:
        """After 3 rounds, accept the best available offer"""
        offers = market_state.get("offers", {})
        
        if not offers:
            print(f"[{self.ev_id}] 🚫 No offers available to accept after 3 rounds")
            return None
        
        # Find the best offer (lowest price within max budget)
        best_cs_id = None
        best_price = float('inf')
        
        for cs_id, offer in offers.items():
            price = offer['current']['price']
            if price <= self.max_acceptable_price and price < best_price:
                best_price = price
                best_cs_id = cs_id
        
        # If no offer within budget, accept the cheapest one anyway
        if best_cs_id is None:
            print(f"[{self.ev_id}] ⚠️ No offers within budget, accepting cheapest available")
            for cs_id, offer in offers.items():
                price = offer['current']['price']
                if price < best_price:
                    best_price = price
                    best_cs_id = cs_id
        
        if best_cs_id:
            print(f"[{self.ev_id}] ✅ Round 3 complete - accepting best offer from {best_cs_id}: ${best_price:.3f}/kWh")
            return AgentDecision(
                agent_id=self.ev_id,
                action="ACCEPT",
                target_id=best_cs_id,
                timestamp=datetime.now().isoformat()
            )
        
        return None
    
    async def run(self):
        """Main EV thinking loop"""
        print(f"[{self.ev_id}] 🚀 Starting thinking loop...")
        
        while not self.has_deal:
            try:
                # Read current market state
                market_state = await self.read_market_state()
                
                # Check if we already have a deal
                for deal in market_state.get("deals", []):
                    if deal.get("ev_id") == self.ev_id:
                        self.has_deal = True
                        print(f"[{self.ev_id}] ✅ Deal completed! Price: ${deal['price']:.3f}/kWh with {deal['cs_id']}")
                        return
                
                # Think and make decision
                decision = await self.think_and_decide(market_state)
                
                # Send decision to marketplace if we made one
                if decision:
                    print(f"[{decision.agent_id}] 📤 Sending {decision.action} decision to marketplace")
                    await self.marketplace_queue.put(decision)
                
                # Thinking cooldown
                await asyncio.sleep(self.thinking_delay)
                
            except Exception as e:
                print(f"[{self.ev_id}] ❌ Error in thinking loop: {e}")
                await asyncio.sleep(2)


# =====================================================================================
# CS AGENT - LLM-Powered Charging Station
# =====================================================================================

class CSAgent:
    """
    Charging Station agent that uses LLM to make pricing and negotiation decisions.
    Operates in continuous thinking loop: read state → think → decide → update.
    """
    
    def __init__(self, cs_id: str, electricity_cost: float, min_profit_margin: float,
                 marketplace_queue: asyncio.Queue):
        self.cs_id = cs_id
        self.electricity_cost = electricity_cost
        self.min_profit_margin = min_profit_margin
        self.marketplace_queue = marketplace_queue
        self.llm_client = ollama.AsyncClient()
        self.has_deal = False
        self.thinking_delay = 1.0  # Base thinking time
        
        # Calculate initial offer price
        self.initial_price = electricity_cost * (1 + min_profit_margin)
        
        print(f"[{self.cs_id}] ⚡ CS initialized - Cost: ${electricity_cost:.3f}, Margin: {min_profit_margin*100:.1f}%, Initial: ${self.initial_price:.3f}")
    
    async def read_market_state(self) -> Dict:
        """Read current market state from file"""
        try:
            with open("market_state.json", 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {"offers": {}, "counter_offers": {}, "deals": []}
    
    async def think_and_decide(self, market_state: Dict) -> Optional[AgentDecision]:
        """Use LLM to analyze market and make pricing decision"""
        print(f"[{self.cs_id}] 🤔 Thinking about market state...")
        
        # Get available EVs for validation
        available_evs = list(market_state.get("counter_offers", {}).keys())
        evs_str = ", ".join(available_evs) if available_evs else "None currently negotiating"
        
        # Create LLM prompt with market analysis
        prompt = f"""You are {self.cs_id} (Charging Station). Your electricity cost is ${self.electricity_cost:.3f}/kWh.
Minimum profit margin: {self.min_profit_margin*100:.1f}%
Minimum acceptable price: ${self.electricity_cost * (1 + self.min_profit_margin):.3f}/kWh

ACTIVE EV NEGOTIATIONS: {evs_str}

Current market state:
{json.dumps(market_state, indent=2)}

CRITICAL RULES:
1. NEVER sell below your cost: ${self.electricity_cost:.3f}/kWh
2. Target profit margin: {self.min_profit_margin*100:.1f}%+
3. ONLY reference existing EVs: {available_evs}
4. When making COUNTER offers, set a profitable price

BUSINESS STRATEGY:
- ACCEPT EV counter-offers that give you good profit margins
- COUNTER with prices above ${self.electricity_cost * (1 + self.min_profit_margin):.3f}/kWh
- Be competitive but maintain profitability

ANALYSIS REQUIRED:
- Which EV counter-offers are above your minimum price?
- Are any EVs offering increasing prices (previous < current)?
- What's the most profitable deal you can accept?

RESPOND WITH VALID JSON ONLY (no explanations, no other text):

For accepting EV counter-offer:
{{"action": "ACCEPT", "target_ev_id": "EV-1"}}

For making new offer:
{{"action": "COUNTER", "price": 0.145}}

For holding position:
{{"action": "HOLD"}}

IMPORTANT: Response must be valid JSON starting with {{ and ending with }}"""

        try:
            response = await asyncio.wait_for(
                self.llm_client.chat(
                    model='llama3.1:8b',
                    messages=[{'role': 'user', 'content': prompt}],
                    format='json'
                ),
                timeout=15.0
            )
            
            llm_response = response['message']['content'].strip()
            print(f"[{self.cs_id}] 🔍 Raw LLM response: {llm_response[:100]}...")
            
            decision_data = json.loads(llm_response)
            action = decision_data.get('action', 'UNKNOWN')
            print(f"[{self.cs_id}] 💭 LLM decided: {action}")
            
            # Validate action is one of the expected values
            if action not in ["ACCEPT", "COUNTER", "HOLD"]:
                print(f"[{self.cs_id}] ⚠️ Invalid LLM action '{action}', treating as HOLD")
                return None
            
            # Convert to AgentDecision with validation
            if decision_data.get("action") == "ACCEPT":
                target_ev = decision_data.get("target_ev_id")
                
                # Re-read fresh market state for validation
                fresh_market_state = await self.read_market_state()
                
                # Validate target EV exists
                if target_ev not in fresh_market_state.get("counter_offers", {}):
                    print(f"[{self.cs_id}] ⚠️ LLM tried to accept non-existent EV {target_ev}, holding instead")
                    print(f"[{self.cs_id}] 🔍 Available EVs: {list(fresh_market_state.get('counter_offers', {}).keys())}")
                    return None
                
                # Validate the counter-offer is profitable
                ev_offer_price = fresh_market_state["counter_offers"][target_ev]["current"]["price"]
                min_acceptable = self.electricity_cost * (1 + self.min_profit_margin)
                if ev_offer_price < min_acceptable:
                    print(f"[{self.cs_id}] ⚠️ EV {target_ev} offer ${ev_offer_price:.3f} below minimum ${min_acceptable:.3f}, holding instead")
                    return None
                
                return AgentDecision(
                    agent_id=self.cs_id,
                    action="ACCEPT",
                    target_id=target_ev,
                    timestamp=datetime.now().isoformat()
                )
                
            elif decision_data.get("action") == "COUNTER":
                proposed_price = decision_data.get("price", self.initial_price)
                
                # Validate price is above minimum
                min_acceptable = self.electricity_cost * (1 + self.min_profit_margin)
                if proposed_price < min_acceptable:
                    print(f"[{self.cs_id}] ⚠️ LLM proposed price ${proposed_price:.3f} below minimum ${min_acceptable:.3f}, holding instead")
                    return None
                
                return AgentDecision(
                    agent_id=self.cs_id,
                    action="COUNTER",
                    price=proposed_price,
                    timestamp=datetime.now().isoformat()
                )
            else:
                # HOLD - no decision to send  
                return None
                
        except asyncio.TimeoutError:
            print(f"[{self.cs_id}] ⏰ LLM thinking timeout, holding position...")
            return None
        except Exception as e:
            print(f"[{self.cs_id}] ❌ LLM error: {e}, holding position...")
            return None
    
    async def run(self):
        """Main CS thinking loop"""
        print(f"[{self.cs_id}] 🚀 Starting thinking loop...")
        
        # Wait a moment for marketplace to be ready
        await asyncio.sleep(0.5)
        
        # Start with initial offer
        print(f"[{self.cs_id}] 💰 Making initial offer: ${self.initial_price:.3f}/kWh")
        initial_decision = AgentDecision(
            agent_id=self.cs_id,
            action="COUNTER",  # ← This should create the initial offer
            price=self.initial_price,
            timestamp=datetime.now().isoformat()
        )
        await self.marketplace_queue.put(initial_decision)  # ← Should update JSON
        print(f"[{self.cs_id}] 📤 Initial offer sent to marketplace queue")
        print(f"[{self.cs_id}] 🔍 DEBUG: Decision added to queue, queue size: {self.marketplace_queue.qsize()}")
        
        while not self.has_deal:
            try:
                # Read current market state
                market_state = await self.read_market_state()
                
                # Check if we already have a deal
                for deal in market_state.get("deals", []):
                    if deal.get("cs_id") == self.cs_id:
                        self.has_deal = True
                        print(f"[{self.cs_id}] ✅ Deal completed! Price: ${deal['price']:.3f}/kWh with {deal['ev_id']}")
                        return
                
                # Think and make decision
                decision = await self.think_and_decide(market_state)
                
                # Send decision to marketplace if we made one
                if decision:
                    print(f"[{decision.agent_id}] 📤 Sending {decision.action} decision to marketplace")
                    await self.marketplace_queue.put(decision)
                
                # Thinking cooldown
                await asyncio.sleep(self.thinking_delay)
                
            except Exception as e:
                print(f"[{self.cs_id}] ❌ Error in thinking loop: {e}")
                await asyncio.sleep(2)


# =====================================================================================
# MAIN SIMULATION
# =====================================================================================

async def main():
    """
    Main simulation coordinator that launches all agents and manages the thinking loop.
    """
    print("🚀 Multi-Agent EV Charging Marketplace with LLM Thinking Loop")
    print("=" * 80)
    print("🧠 Each agent thinks independently using Llama 3.1 8B")
    print("📄 All coordination happens through market_state.json")
    print("⚡ Real-time negotiation with current/previous value tracking")
    print()
    
    # Test Ollama connection
    try:
        test_client = ollama.AsyncClient()
        await test_client.chat(
            model='llama3.1:8b',
            messages=[{'role': 'user', 'content': 'Say "Ready"'}]
        )
        print("✅ Ollama server confirmed ready")
        del test_client
    except Exception as e:
        print(f"❌ Ollama connection failed: {e}")
        print("Please ensure Ollama is running with 'ollama serve' and llama3.1:8b is available")
        return
    
    # Initialize marketplace
    marketplace_queue = asyncio.Queue()
    marketplace_agent = MarketplaceAgent(agents_queue=marketplace_queue)
    
    # Initialize agents
    ev_agents = [
        EVAgent("EV-1", max_acceptable_price=0.150, energy_needed=36.0, marketplace_queue=marketplace_queue),
        EVAgent("EV-2", max_acceptable_price=0.160, energy_needed=33.0, marketplace_queue=marketplace_queue),
        EVAgent("EV-3", max_acceptable_price=0.140, energy_needed=45.0, marketplace_queue=marketplace_queue),
    ]
    
    cs_agents = [
        CSAgent("CS-A", electricity_cost=0.080, min_profit_margin=0.25, marketplace_queue=marketplace_queue),
        CSAgent("CS-B", electricity_cost=0.090, min_profit_margin=0.20, marketplace_queue=marketplace_queue),
        CSAgent("CS-C", electricity_cost=0.070, min_profit_margin=0.30, marketplace_queue=marketplace_queue),
    ]
    
    print(f"\n🤖 Launching {len(ev_agents)} EV agents and {len(cs_agents)} CS agents...")
    print("🔄 Agents will negotiate through market_state.json until all deals are completed")
    print()
    
    # Launch all agents concurrently
    tasks = [
        asyncio.create_task(marketplace_agent.run()),
        *[asyncio.create_task(ev.run()) for ev in ev_agents],
        *[asyncio.create_task(cs.run()) for cs in cs_agents]
    ]
    
    try:
        # Wait for all tasks to complete (marketplace will signal completion)
        await asyncio.gather(*tasks, return_exceptions=True)
        
    except KeyboardInterrupt:
        print("\n🛑 Simulation interrupted by user")
        
    finally:
        # Display final market state
        final_state = marketplace_agent.read_market_state()
        print("\n" + "=" * 80)
        print("🎯 FINAL MARKET STATE")
        print("=" * 80)
        print(json.dumps(final_state, indent=2))
        
        # Summary
        deals = final_state.get("deals", [])
        print(f"\n✅ Completed deals: {len(deals)}/3")
        for deal in deals:
            print(f"   {deal['ev_id']} ↔ {deal['cs_id']}: ${deal['price']:.3f}/kWh")
        
        print(f"\n📄 Full negotiation history saved in: market_state.json")
        print("🧠 All decisions were made by independent LLM agents!")


if __name__ == "__main__":
    """Entry point for the multi-agent EV marketplace simulation"""
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n🛑 Simulation interrupted by user")
    except Exception as e:
        import traceback
        print(f"\n\n❌ Simulation error: {e}")
        print("Full traceback:")
        traceback.print_exc()