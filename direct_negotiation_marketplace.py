#!/usr/bin/env python3
"""
Direct Negotiation EV Charging Marketplace

A decentralized multi-agent system where agents communicate directly
using individual asyncio queues for processing offers. This ensures
sequential, thoughtful negotiation without a central coordinator.
"""

import asyncio
import json
import random
import time # Added for timing metrics
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any, TYPE_CHECKING
from enum import Enum
from autogen_ext.models.ollama import OllamaChatCompletionClient
from autogen_core.models import UserMessage

# =====================================================================================
# DATA STRUCTURES FOR DIRECT COMMUNICATION
# =====================================================================================

class MessageType(Enum):
    INITIAL_OFFER = "initial_offer"
    COUNTER_OFFER = "counter_offer"
    ACCEPTANCE = "acceptance"
    REJECTION = "rejection"
    DEAL_CONFIRMED = "deal_confirmed"

@dataclass
class Envelope:
    """A wrapper for messages to direct them between agents."""
    type: MessageType
    from_agent_id: str
    to_agent_id: str
    payload: Dict[str, Any]

# Use TYPE_CHECKING to resolve circular dependencies for type hinting
if TYPE_CHECKING:
    from __main__ import EVAgent, ChargingStationAgent

# =====================================================================================
# CHARGING STATION AGENT - WITH DIRECT COMMUNICATION
# =====================================================================================

class ChargingStationAgent:
    """A charging station that negotiates directly with multiple EVs."""

    def __init__(self, station_id: str, electricity_cost: float, min_margin: float, num_slots: int):
        self.station_id = station_id
        self.electricity_cost = electricity_cost
        self.min_price = electricity_cost * (1 + min_margin)
        self.available_slots = num_slots
        self.negotiation_queue = asyncio.Queue()
        self.ev_agents: List['EVAgent'] = []
        self.deals_made: List[Dict] = []
        self.llm = OllamaChatCompletionClient(model="llama3.1:8b")
        print(f"[{self.station_id}] ⚡ Ready with {self.available_slots} slots. Min price: ${self.min_price:.3f}/kWh")

    def link_ev_agents(self, ev_agents: List['EVAgent']):
        """Establish direct communication links to EV agents."""
        self.ev_agents = ev_agents

    async def receive_message(self, envelope: Envelope):
        """Receives a message and puts it in the personal queue."""
        await self.negotiation_queue.put(envelope)

    async def send_initial_offers(self):
        """Calculates an initial price and sends it to all EVs."""
        initial_price = await self._determine_initial_price()
        print(f"[{self.station_id}] 📢 Broadcasting initial offer: ${initial_price:.3f}/kWh")

        for ev in self.ev_agents:
            offer_payload = {
                "price_per_kwh": initial_price,
                "offer_id": f"{self.station_id}-{ev.ev_id}-initial"
            }
            envelope = Envelope(
                type=MessageType.INITIAL_OFFER,
                from_agent_id=self.station_id,
                to_agent_id=ev.ev_id,
                payload=offer_payload
            )
            await ev.receive_message(envelope)

    async def process_negotiations(self):
        """The main processing loop for the charging station."""
        
        # Keep processing as long as there are slots and potential for messages
        processing_start_time = asyncio.get_event_loop().time()
        max_wait_time = 10.0 # Wait a total of 10 seconds for messages

        while self.available_slots > 0:
            if asyncio.get_event_loop().time() - processing_start_time > max_wait_time:
                print(f"[{self.station_id}] Timed out waiting for EV responses.")
                break

            try:
                # Use a smaller timeout for each get() to allow checking the overall timer
                envelope: Envelope = await asyncio.wait_for(self.negotiation_queue.get(), timeout=2.0)
                
                if envelope.type == MessageType.COUNTER_OFFER:
                    await self._handle_counter_offer(envelope)
                elif envelope.type == MessageType.ACCEPTANCE:
                    await self._handle_acceptance(envelope)

                self.negotiation_queue.task_done()
            except asyncio.TimeoutError:
                # This is now expected, just means no message arrived in the last 2 seconds.
                # The outer loop will re-check the main timer.
                continue
        
        print(f"[{self.station_id}]  मार्केट से बाहर (slots left: {self.available_slots})")


    async def _handle_counter_offer(self, envelope: Envelope):
        """Uses LLM to evaluate a counter-offer from an EV."""
        counter_price = envelope.payload["counter_price"]
        ev_id = envelope.from_agent_id
        print(f"[{self.station_id}] 🔍 Received counter from {ev_id}: ${counter_price:.3f}/kWh")

        # LLM decides to accept or reject
        decision = await self._evaluate_counter_offer(counter_price, ev_id)

        target_ev = next((ev for ev in self.ev_agents if ev.ev_id == ev_id), None)
        if not target_ev: return

        if decision.get("action") == "ACCEPT" and self.available_slots > 0:
            print(f"[{self.station_id}] ✅ Accepting counter from {ev_id}")
            self.available_slots -= 1
            deal_payload = {
                "station_id": self.station_id,
                "ev_id": ev_id,
                "final_price": counter_price,
                "reason": "Accepted counter-offer"
            }
            self.deals_made.append(deal_payload)
            
            # Send confirmation back to EV
            confirmation = Envelope(
                type=MessageType.DEAL_CONFIRMED,
                from_agent_id=self.station_id,
                to_agent_id=ev_id,
                payload=deal_payload
            )
            await target_ev.receive_message(confirmation)
        else:
            print(f"[{self.station_id}] ❌ Rejecting counter from {ev_id}: {decision.get('reasoning')}")

    async def _handle_acceptance(self, envelope: Envelope):
        """Handles a direct acceptance from an EV."""
        if self.available_slots > 0:
            price = envelope.payload["price_per_kwh"]
            ev_id = envelope.from_agent_id
            print(f"[{self.station_id}] ✅ Offer accepted by {ev_id} at ${price:.3f}/kWh")
            self.available_slots -= 1
            deal_payload = {
                "station_id": self.station_id,
                "ev_id": ev_id,
                "final_price": price,
                "reason": "Initial offer accepted"
            }
            self.deals_made.append(deal_payload)

            target_ev = next((ev for ev in self.ev_agents if ev.ev_id == ev_id), None)
            if target_ev:
                confirmation = Envelope(
                    type=MessageType.DEAL_CONFIRMED,
                    from_agent_id=self.station_id,
                    to_agent_id=ev_id,
                    payload=deal_payload
                )
                await target_ev.receive_message(confirmation)
        else:
            print(f"[{self.station_id}] ⏭️ Acceptance from {envelope.from_agent_id} ignored, no slots left.")


    async def _determine_initial_price(self) -> float:
        """LLM-powered competitive pricing strategy."""
        # This function can be expanded with more market context if needed
        prompt = f"""You are {self.station_id}, a charging station setting an initial competitive price.
- Your absolute minimum price is ${self.min_price:.3f}/kWh.
- You have {self.available_slots} slots available.
- Strategy: Price competitively to attract buyers but leave some room for negotiation.

Respond with JSON only: {{"price": 0.125, "reasoning": "Brief explanation"}}"""
        try:
            response = await self.llm.create(messages=[UserMessage(content=prompt, source="user")])
            llm_output = response.content
            if isinstance(llm_output, str):
                decision = json.loads(llm_output)
                return max(self.min_price, float(decision.get("price", self.min_price * 1.2)))
        except Exception:
            pass
        return self.min_price * 1.2

    async def _evaluate_counter_offer(self, counter_price: float, ev_id: str) -> Dict:
        """LLM-powered evaluation of a counter-offer."""
        is_profitable = counter_price >= self.min_price
        prompt = f"""You are {self.station_id}. EV {ev_id} has counter-offered ${counter_price:.3f}/kWh.
- Your minimum price is ${self.min_price:.3f}/kWh.
- The offer is profitable: {is_profitable}.
- You have {self.available_slots} slots.

Decide whether to ACCEPT or REJECT.
Respond with JSON only: {{"action": "ACCEPT", "reasoning": "Brief explanation"}} or {{"action": "REJECT", "reasoning": "Brief explanation"}}"""
        try:
            response = await self.llm.create(messages=[UserMessage(content=prompt, source="user")])
            llm_output = response.content
            if isinstance(llm_output, str):
                return json.loads(llm_output)
        except Exception:
            pass
        return {"action": "REJECT", "reasoning": "Fallback due to parse error"}

# =====================================================================================
# EV AGENT - WITH DIRECT COMMUNICATION
# =====================================================================================

class EVAgent:
    """An EV owner that negotiates directly with multiple charging stations."""

    def __init__(self, ev_id: str, max_budget: float, energy_needed: float):
        self.ev_id = ev_id
        self.max_budget = max_budget
        self.energy_needed = energy_needed
        self.negotiation_queue = asyncio.Queue()
        self.charging_stations: List['ChargingStationAgent'] = []
        self.final_deal: Optional[Dict] = None
        self.deal_pending = False # New flag to prevent multiple acceptances
        self.llm = OllamaChatCompletionClient(model="llama3.1:8b")
        print(f"[{self.ev_id}] 🚗 Ready. Budget: ${self.max_budget:.3f}/kWh, Need: {self.energy_needed} kWh")

    def link_cs_agents(self, cs_agents: List['ChargingStationAgent']):
        """Establish direct communication links to CS agents."""
        self.charging_stations = cs_agents

    async def receive_message(self, envelope: Envelope):
        """Receives a message and puts it in the personal queue."""
        if not self.final_deal and not self.deal_pending:
            await self.negotiation_queue.put(envelope)

    async def process_negotiations(self):
        """The main processing loop for the EV agent."""
        # Wait for all initial offers to arrive
        await asyncio.sleep(1) 

        while not self.final_deal and not self.deal_pending and not self.negotiation_queue.empty():
            try:
                envelope: Envelope = await self.negotiation_queue.get()
                
                if envelope.type == MessageType.INITIAL_OFFER:
                    await self._handle_initial_offer(envelope)
                elif envelope.type == MessageType.DEAL_CONFIRMED:
                    self._handle_deal_confirmed(envelope)

                self.negotiation_queue.task_done()
            except asyncio.CancelledError:
                break
        
        status = f"DEAL with {self.final_deal['station_id']}" if self.final_deal else "NO DEAL"
        print(f"[{self.ev_id}] 🤝 मार्केट से बाहर ({status})")

    def _handle_deal_confirmed(self, envelope: Envelope):
        """Handles the final deal confirmation from a station."""
        if not self.final_deal: # Accept the first confirmed deal
            print(f"[{self.ev_id}] 🎉 Deal confirmed with {envelope.from_agent_id}!")
            self.final_deal = envelope.payload
            
            # Empty the queue to stop any further processing of other offers
            while not self.negotiation_queue.empty():
                try:
                    self.negotiation_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
            
            print(f"[{self.ev_id}] 🛑 Halting all other negotiations.")

    async def _handle_initial_offer(self, envelope: Envelope):
        """Uses LLM to evaluate an initial offer."""
        price = envelope.payload["price_per_kwh"]
        station_id = envelope.from_agent_id
        print(f"[{self.ev_id}] 📩 Received offer from {station_id}: ${price:.3f}/kWh")

        # LLM decides to accept or counter
        decision = await self._evaluate_initial_offer(price, station_id)
        action = decision.get("action")
        
        target_station = next((cs for cs in self.charging_stations if cs.station_id == station_id), None)
        if not target_station: return

        if action == "ACCEPT":
            if not self.deal_pending:
                self.deal_pending = True
                print(f"[{self.ev_id}] ✅ Accepting offer from {station_id}, pending confirmation...")
                acceptance = Envelope(
                    type=MessageType.ACCEPTANCE,
                    from_agent_id=self.ev_id,
                    to_agent_id=station_id,
                    payload=envelope.payload
                )
                await target_station.receive_message(acceptance)

        elif action == "COUNTER":
            if not self.deal_pending:
                counter_price = decision.get("price", self.max_budget)
                print(f"[{self.ev_id}] 🔄 Countering {station_id} with ${counter_price:.3f}/kWh")
                counter_payload = {"counter_price": counter_price}
                counter = Envelope(
                    type=MessageType.COUNTER_OFFER,
                    from_agent_id=self.ev_id,
                    to_agent_id=station_id,
                    payload=counter_payload
                )
                await target_station.receive_message(counter)
            
    async def _evaluate_initial_offer(self, price: float, station_id: str) -> Dict:
        """LLM-powered evaluation of an initial offer."""
        is_in_budget = price <= self.max_budget
        prompt = f"""You are {self.ev_id}. You received an offer of ${price:.3f}/kWh from {station_id}.
- Your max budget is ${self.max_budget:.3f}/kWh.
- The offer is in budget: {is_in_budget}.

Decide whether to ACCEPT, or make a COUNTER offer.
If countering, suggest a new price.
Respond with JSON only: {{"action": "ACCEPT", "reasoning": "..."}} or {{"action": "COUNTER", "price": 0.123, "reasoning": "..."}}"""
        try:
            response = await self.llm.create(messages=[UserMessage(content=prompt, source="user")])
            llm_output = response.content
            if isinstance(llm_output, str):
                return json.loads(llm_output)
        except Exception:
            pass
        return {"action": "REJECT", "reasoning": "Fallback due to parse error"}

# =====================================================================================
# MAIN SIMULATION
# =====================================================================================

async def main():
    """Sets up and runs the direct negotiation marketplace."""
    print("🔋 Direct Negotiation EV Charging Marketplace - 5x5 Agent Test")
    print("=" * 60)
    
    start_time = time.time()

    # 1. Create 5x5 Agents
    ev_agents = [
        EVAgent(ev_id=f"EV{i+1}", max_budget=0.150 + (i * 0.005), energy_needed=random.randint(30, 50))
        for i in range(5)
    ]
    
    cs_agents = [
        ChargingStationAgent(
            station_id=f"CS{chr(65+i)}", 
            electricity_cost=0.08 + (i * 0.005), 
            min_margin=0.15 + (i*0.02), 
            num_slots=random.randint(1, 3)
        )
        for i in range(5)
    ]

    # 2. Link Agents for Direct Communication
    for ev in ev_agents:
        ev.link_cs_agents(cs_agents)
    for cs in cs_agents:
        cs.link_ev_agents(ev_agents)
    
    print(f"\n🔗 Created and linked {len(ev_agents)} EV agents and {len(cs_agents)} CS agents.")
    
    # 3. Start the Market: CS agents broadcast initial offers
    print("\n🚀 Market Opening: Charging Stations sending offers...")
    initial_offer_tasks = [cs.send_initial_offers() for cs in cs_agents]
    await asyncio.gather(*initial_offer_tasks)

    # 4. Run Agent Processing Loops Concurrently
    print("\n⚡ Negotiations in Progress: Agents processing their queues...")
    processing_tasks = [ev.process_negotiations() for ev in ev_agents] + \
                       [cs.process_negotiations() for cs in cs_agents]
    
    await asyncio.gather(*processing_tasks)

    # 5. Print Final Results
    end_time = time.time()
    negotiation_duration = end_time - start_time

    print("\n" + "=" * 60)
    print("🎯 MARKETPLACE CLOSED - FINAL DEALS")
    print("=" * 60)
    
    print(f" Negotiation Duration: {negotiation_duration:.2f} seconds")
    print("-" * 60)

    total_deals = 0
    deals_by_cs = {}
    for cs in cs_agents:
        if cs.deals_made:
            total_deals += len(cs.deals_made)
            deals_by_cs[cs.station_id] = cs.deals_made
    
    if total_deals == 0:
        print("No deals were made.")
    else:
        print(f"Total Deals Made: {total_deals}")
        for station_id, deals in deals_by_cs.items():
            print(f"  Deals for {station_id}:")
            for deal in deals:
                print(f"    - EV: {deal['ev_id']}, Price: ${deal['final_price']:.3f}/kWh")

if __name__ == "__main__":
    asyncio.run(main()) 