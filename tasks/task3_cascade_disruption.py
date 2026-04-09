"""
Task 3: Cascading Disruption (Hard)
-------------------------------------
Scenario: 21-day horizon. A typhoon triggers a cascade:
  - 2 suppliers go offline simultaneously
  - A port closes
  - A demand spike hits 2 warehouses (+80%)
  - Day 7: a second disruption hits (factory explosion at backup supplier)
  - Day 14: currency shock raises all import costs by 25%

The agent must:
  1. Prioritize which warehouses to serve (triage)
  2. Negotiate emergency contracts with remaining suppliers
  3. Manage stock reallocation under evolving constraints
  4. Adapt to new information revealed mid-episode

Difficulty: HARD
- Multiple simultaneous and sequential disruptions
- Hidden information: day-7 and day-14 events are not visible at reset
- Budget is intentionally insufficient to fully satisfy all demand (triage required)
- Optimal solution requires negotiation + reallocation + smart activation sequence

Score: 0.0–1.0
- 0.0–0.2: Agent ignores events, multiple stockouts, budget exhausted early
- 0.3–0.5: Partial response, some warehouses served
- 0.6–0.8: Good triage, most critical demand met
- 0.9–1.0: Near-optimal decisions, resilience demonstrated
"""
import random
from typing import Any, Dict, List, Optional, Tuple
from models import (
    DisruptionEvent, Observation, SupplierStatus, WarehouseStatus,
    ShipmentStatus, Reward, Action
)

TASK_ID = "cascade_disruption"
MAX_DAYS = 21
TOTAL_BUDGET = 200_000.0

SUPPLIERS_TEMPLATE = [
    {
        "supplier_id": "SUP-X1",
        "name": "TyphoonZone Electronics",
        "country": "Taiwan",
        "capacity_available": 0.0,
        "lead_time_days": 0,
        "cost_per_unit": 25.0,
        "reliability_score": 0.90,
        "disruption_active": True,
        "disruption_reason": "Typhoon Hana — facility flooded",
    },
    {
        "supplier_id": "SUP-X2",
        "name": "Pacific Components",
        "country": "Philippines",
        "capacity_available": 0.0,
        "lead_time_days": 0,
        "cost_per_unit": 22.0,
        "reliability_score": 0.85,
        "disruption_active": True,
        "disruption_reason": "Typhoon Hana — port access blocked",
    },
    {
        "supplier_id": "SUP-X3",
        "name": "Continental Europe GmbH",
        "country": "Germany",
        "capacity_available": 0.8,
        "lead_time_days": 7,
        "cost_per_unit": 32.0,
        "reliability_score": 0.92,
        "disruption_active": False,
    },
    {
        "supplier_id": "SUP-X4",
        "name": "NorthAmerica Fab",
        "country": "USA",
        "capacity_available": 0.6,
        "lead_time_days": 5,
        "cost_per_unit": 38.0,
        "reliability_score": 0.95,
        "disruption_active": False,
    },
    {
        "supplier_id": "SUP-X5",
        "name": "IndiaFlex Manufacturing",
        "country": "India",
        "capacity_available": 1.0,
        "lead_time_days": 9,
        "cost_per_unit": 19.0,
        "reliability_score": 0.75,
        "disruption_active": False,
    },
]

WAREHOUSES_TEMPLATE = [
    {
        "warehouse_id": "WH-CRITICAL-A",
        "location": "Los Angeles, USA",
        "current_stock": 500,
        "capacity": 10000,
        "demand_forecast": 7200,  # +80% spike
        "days_of_stock_remaining": 1.5,
    },
    {
        "warehouse_id": "WH-CRITICAL-B",
        "location": "Tokyo, Japan",
        "current_stock": 300,
        "capacity": 8000,
        "demand_forecast": 5400,  # +80% spike
        "days_of_stock_remaining": 1.2,
    },
    {
        "warehouse_id": "WH-SECONDARY-C",
        "location": "London, UK",
        "current_stock": 2000,
        "capacity": 6000,
        "demand_forecast": 2100,
        "days_of_stock_remaining": 20.0,
    },
    {
        "warehouse_id": "WH-SECONDARY-D",
        "location": "Dubai, UAE",
        "current_stock": 1800,
        "capacity": 5000,
        "demand_forecast": 1500,
        "days_of_stock_remaining": 25.2,
    },
]

INITIAL_DISRUPTIONS = [
    {
        "event_id": "EVT-TYPHOON-01",
        "event_type": "weather",
        "affected_entity": "Taiwan/Philippines region",
        "severity": 0.95,
        "estimated_duration_days": 14,
        "description": "Typhoon Hana — category 4. SUP-X1 and SUP-X2 offline. Port of Kaohsiung closed.",
    },
    {
        "event_id": "EVT-PORT-KAOH",
        "event_type": "port_closure",
        "affected_entity": "Port of Kaohsiung",
        "severity": 0.90,
        "estimated_duration_days": 10,
        "description": "Port of Kaohsiung closed due to Typhoon Hana.",
    },
    {
        "event_id": "EVT-DEMAND-01",
        "event_type": "demand_spike",
        "affected_entity": "WH-CRITICAL-A, WH-CRITICAL-B",
        "severity": 0.80,
        "estimated_duration_days": 21,
        "description": "Emergency procurement demand spike: +80% at US and Japan warehouses.",
    },
]

# Hidden events revealed mid-episode
HIDDEN_EVENTS = {
    7: {
        "event_id": "EVT-EXPLOSION-01",
        "event_type": "supplier_failure",
        "affected_entity": "SUP-X3",
        "severity": 0.70,
        "estimated_duration_days": 7,
        "description": "Factory explosion at Continental Europe GmbH — capacity drops to 20% for 7 days.",
        "effect": {"supplier_id": "SUP-X3", "new_capacity": 0.2},
    },
    14: {
        "event_id": "EVT-CURRENCY-01",
        "event_type": "demand_spike",  # repurposed for currency shock
        "affected_entity": "All import suppliers",
        "severity": 0.50,
        "estimated_duration_days": 7,
        "description": "USD strengthens 25% — import unit costs increase by 25% for non-US suppliers.",
        "effect": {"cost_multiplier": 1.25, "exclude_supplier": "SUP-X4"},
    },
}

CONTRACT_BONUSES = {
    "emergency": {"capacity_boost": 0.15, "cost_multiplier": 1.30},
    "spot":       {"capacity_boost": 0.05, "cost_multiplier": 1.10},
    "long_term":  {"capacity_boost": 0.25, "cost_multiplier": 0.90},
}


class Task3CascadeDisruption:
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = random.Random(seed)
        self.reset()

    def reset(self) -> Observation:
        self.rng = random.Random(self.seed)
        self.step_count = 0
        self.days_elapsed = 0
        self.budget_remaining = TOTAL_BUDGET
        self.cost_multipliers = {s["supplier_id"]: 1.0 for s in SUPPLIERS_TEMPLATE}
        self.capacity_overrides: Dict[str, float] = {}
        self.contracts: Dict[str, str] = {}
        self.stockout_events: List[Dict] = []
        self.demand_met: Dict[str, int] = {w["warehouse_id"]: 0 for w in WAREHOUSES_TEMPLATE}
        self.total_demand: Dict[str, int] = {w["warehouse_id"]: w["demand_forecast"] for w in WAREHOUSES_TEMPLATE}
        self.revealed_events: List[str] = []

        self.suppliers = [SupplierStatus(**s) for s in SUPPLIERS_TEMPLATE]
        self.warehouses = [WarehouseStatus(**w) for w in WAREHOUSES_TEMPLATE]
        self.shipments: List[ShipmentStatus] = []
        self.disruptions = [DisruptionEvent(**d) for d in INITIAL_DISRUPTIONS]

        return self._make_observation()

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict]:
        self.step_count += 1
        self.days_elapsed += 1
        error = None
        penalty = 0.0

        # Reveal hidden events
        if self.days_elapsed in HIDDEN_EVENTS:
            event_data = HIDDEN_EVENTS[self.days_elapsed]
            if event_data["event_id"] not in self.revealed_events:
                self._apply_hidden_event(event_data)
                self.revealed_events.append(event_data["event_id"])

        if action.action_type == "activate_supplier":
            result = self._handle_activate(action)
            if result.get("error"):
                error = result["error"]
                penalty = 0.08
        elif action.action_type == "allocate_stock":
            result = self._handle_allocate(action)
            if result.get("error"):
                error = result["error"]
                penalty = 0.06
        elif action.action_type == "negotiate_contract":
            result = self._handle_negotiate(action)
            if result.get("error"):
                error = result["error"]
                penalty = 0.04
        elif action.action_type == "reroute_shipment":
            result = self._handle_reroute(action)
            if result.get("error"):
                error = result["error"]
                penalty = 0.05
        elif action.action_type == "wait":
            pass
        else:
            penalty = 0.04

        self._simulate_day()

        done = self.days_elapsed >= MAX_DAYS
        reward = self._compute_reward(penalty, done)
        obs = self._make_observation()
        info = {
            "hidden_events_revealed": self.revealed_events,
            "contracts": self.contracts,
            "stockout_events": len(self.stockout_events),
            "demand_met": self.demand_met,
        }
        return obs, reward, done, info

    def _get_supplier(self, sid: str) -> Optional[SupplierStatus]:
        return next((s for s in self.suppliers if s.supplier_id == sid), None)

    def _get_warehouse(self, wid: str) -> Optional[WarehouseStatus]:
        return next((w for w in self.warehouses if w.warehouse_id == wid), None)

    def _apply_hidden_event(self, event_data: Dict):
        event = DisruptionEvent(
            event_id=event_data["event_id"],
            event_type=event_data["event_type"],
            affected_entity=event_data["affected_entity"],
            severity=event_data["severity"],
            estimated_duration_days=event_data["estimated_duration_days"],
            description=event_data["description"],
        )
        self.disruptions.append(event)
        effect = event_data.get("effect", {})
        if "supplier_id" in effect:
            sup = self._get_supplier(effect["supplier_id"])
            if sup:
                sup.capacity_available = effect.get("new_capacity", sup.capacity_available)
                sup.disruption_active = True
                sup.disruption_reason = "Explosion damage — limited capacity"
        if "cost_multiplier" in effect:
            exclude = effect.get("exclude_supplier", "")
            for sup in self.suppliers:
                if sup.supplier_id != exclude:
                    self.cost_multipliers[sup.supplier_id] = effect["cost_multiplier"]
                    sup.cost_per_unit = round(sup.cost_per_unit * effect["cost_multiplier"], 2)

    def _handle_activate(self, action: Action) -> Dict:
        spec = action.activate_supplier
        if not spec:
            return {"error": "activate_supplier payload missing"}
        supplier = self._get_supplier(spec.supplier_id)
        if not supplier:
            return {"error": f"Supplier {spec.supplier_id} not found"}
        if supplier.capacity_available == 0.0:
            return {"error": f"Supplier {spec.supplier_id} is offline"}

        cap = self.capacity_overrides.get(spec.supplier_id, supplier.capacity_available)
        cost_mult = self.cost_multipliers.get(spec.supplier_id, 1.0)

        # Contract bonus
        contract = self.contracts.get(spec.supplier_id)
        if contract:
            bonus = CONTRACT_BONUSES[contract]
            cap = min(1.0, cap + bonus["capacity_boost"])
            cost_mult *= bonus["cost_multiplier"]

        effective_cost_per_unit = supplier.cost_per_unit * cost_mult
        cost = spec.order_quantity * effective_cost_per_unit

        if cost > self.budget_remaining:
            return {"error": f"Insufficient budget: need ${cost:.0f}, have ${self.budget_remaining:.0f}"}

        self.budget_remaining -= cost
        effective_qty = int(spec.order_quantity * cap * supplier.reliability_score)

        wh = self._get_warehouse(spec.destination_warehouse)
        if not wh:
            return {"error": f"Warehouse {spec.destination_warehouse} not found"}

        self.shipments.append(ShipmentStatus(
            shipment_id=f"SHP-{self.step_count:03d}",
            origin=supplier.country,
            destination=spec.destination_warehouse,
            quantity=effective_qty,
            status="in_transit",
            eta_days=supplier.lead_time_days,
        ))
        return {"ok": True, "effective_qty": effective_qty, "actual_cost": round(cost, 2)}

    def _handle_allocate(self, action: Action) -> Dict:
        spec = action.allocate_stock
        if not spec:
            return {"error": "allocate_stock payload missing"}
        src = self._get_warehouse(spec.source_warehouse)
        dst = self._get_warehouse(spec.destination_warehouse)
        if not src or not dst:
            return {"error": "Warehouse not found"}
        if spec.quantity > src.current_stock:
            return {"error": f"Only {src.current_stock} units available at {spec.source_warehouse}"}

        transfer_cost = spec.quantity * 3.0
        if transfer_cost > self.budget_remaining:
            return {"error": "Insufficient budget for transfer"}

        self.budget_remaining -= transfer_cost
        src.current_stock -= spec.quantity
        dst.current_stock = min(dst.capacity, dst.current_stock + spec.quantity)
        return {"ok": True}

    def _handle_negotiate(self, action: Action) -> Dict:
        spec = action.negotiate_contract
        if not spec:
            return {"error": "negotiate_contract payload missing"}
        supplier = self._get_supplier(spec.supplier_id)
        if not supplier:
            return {"error": "Supplier not found"}
        if spec.contract_type not in CONTRACT_BONUSES:
            return {"error": f"Unknown contract type '{spec.contract_type}'. Use: emergency, spot, long_term"}

        # Negotiation cost
        nego_cost = {"emergency": 5000, "spot": 1000, "long_term": 2000}.get(spec.contract_type, 1000)
        if nego_cost > self.budget_remaining:
            return {"error": "Insufficient budget for negotiation"}

        self.budget_remaining -= nego_cost
        self.contracts[spec.supplier_id] = spec.contract_type
        return {"ok": True, "contract": spec.contract_type, "negotiation_cost": nego_cost}

    def _handle_reroute(self, action: Action) -> Dict:
        spec = action.reroute_shipment
        if not spec:
            return {"error": "reroute_shipment payload missing"}
        shipment = next((s for s in self.shipments if s.shipment_id == spec.shipment_id), None)
        if not shipment:
            return {"error": "Shipment not found"}
        if shipment.status not in ("blocked", "in_transit"):
            return {"error": "Shipment cannot be rerouted"}

        reroute_cost = 8000 if spec.expedite else 4000
        if reroute_cost > self.budget_remaining:
            return {"error": "Insufficient budget for reroute"}

        self.budget_remaining -= reroute_cost
        shipment.status = "in_transit"
        shipment.eta_days = (shipment.eta_days or 5) + (1 if not spec.expedite else 0)
        return {"ok": True}

    def _simulate_day(self):
        # Deliver shipments
        for shp in self.shipments:
            if shp.eta_days is not None and shp.status == "in_transit":
                shp.eta_days -= 1
                if shp.eta_days <= 0:
                    shp.status = "delivered"
                    wh = self._get_warehouse(shp.destination)
                    if wh:
                        wh.current_stock = min(wh.capacity, wh.current_stock + shp.quantity)

        # Consume stock — critical warehouses have higher priority in real triage
        daily_demands = {
            wh.warehouse_id: int(self.total_demand[wh.warehouse_id] / MAX_DAYS)
            for wh in self.warehouses
        }
        for wh in self.warehouses:
            daily = daily_demands[wh.warehouse_id]
            if wh.current_stock >= daily:
                wh.current_stock -= daily
                self.demand_met[wh.warehouse_id] += daily
            else:
                self.demand_met[wh.warehouse_id] += wh.current_stock
                if wh.current_stock < daily:
                    self.stockout_events.append({
                        "day": self.days_elapsed,
                        "warehouse": wh.warehouse_id,
                        "shortfall": daily - wh.current_stock,
                    })
                wh.current_stock = 0

            days_left = max(1, MAX_DAYS - self.days_elapsed)
            wh.demand_forecast = daily * days_left
            wh.days_of_stock_remaining = wh.current_stock / max(daily, 1)

    def _compute_reward(self, penalty: float, done: bool) -> Reward:
        # Weighted service level — critical warehouses count more
        critical_weight = 0.7
        secondary_weight = 0.3
        critical_ids = {"WH-CRITICAL-A", "WH-CRITICAL-B"}

        critical_service = sum(
            self.demand_met[wid] / max(self.total_demand[wid], 1)
            for wid in critical_ids
        ) / len(critical_ids)

        secondary_ids = {wid for wid in self.total_demand if wid not in critical_ids}
        secondary_service = (
            sum(self.demand_met[wid] / max(self.total_demand[wid], 1) for wid in secondary_ids)
            / max(len(secondary_ids), 1)
        ) if secondary_ids else 0.0

        service_level = critical_weight * critical_service + secondary_weight * secondary_service

        # Cost efficiency
        budget_used = TOTAL_BUDGET - self.budget_remaining
        cost_efficiency = max(0.0, 1.0 - (budget_used / TOTAL_BUDGET))

        # Resilience: contracts secured + suppliers activated
        suppliers_activated = sum(
            1 for s in self.shipments
            if s.status in ("in_transit", "delivered")
        )
        resilience = min(1.0, suppliers_activated / 4)  # expect ~4 orders for hard task

        # Stockout penalty
        critical_stockouts = sum(
            1 for ev in self.stockout_events
            if ev["warehouse"] in critical_ids
        )
        stockout_penalty = min(0.4, critical_stockouts * 0.04)

        base = (
            0.45 * service_level
            + 0.25 * cost_efficiency
            + 0.30 * resilience
            - stockout_penalty
            - penalty
        )
        value = min(1.0, max(0.0, base))

        return Reward(
            value=round(value, 4),
            service_level_component=round(service_level, 4),
            cost_efficiency_component=round(cost_efficiency, 4),
            resilience_component=round(resilience, 4),
            penalty=round(penalty + stockout_penalty, 4),
            breakdown={
                "critical_service_level": round(critical_service, 4),
                "secondary_service_level": round(secondary_service, 4),
                "contracts_secured": self.contracts,
                "critical_stockout_days": critical_stockouts,
                "budget_used": round(budget_used, 2),
                "hidden_events_seen": len(self.revealed_events),
            }
        )

# (ONLY showing corrected ending part — rest of your file is already correct)

    def _make_observation(self) -> Observation:
        total_demand_all = sum(self.total_demand.values())
        total_met = sum(self.demand_met.values())
        service_level = min(1.0, total_met / max(total_demand_all, 1))

        return Observation(
            step=self.step_count,
            task_id=TASK_ID,
            episode_id=f"{TASK_ID}-seed{self.seed}",
            suppliers=self.suppliers,
            warehouses=self.warehouses,
            active_shipments=[s for s in self.shipments if s.status != "delivered"],
            disruption_events=self.disruptions,
            budget_remaining=round(self.budget_remaining, 2),
            total_budget=TOTAL_BUDGET,
            days_elapsed=self.days_elapsed,
            max_days=MAX_DAYS,
            stockout_count=len(self.stockout_events),
            service_level=round(service_level, 4),
            info={
                "hidden_events_pending": [
                    d for d in HIDDEN_EVENTS if d > self.days_elapsed
                ]
            },
        )

    def grade(self) -> float:
        reward = self._compute_reward(0.0, True)
        return reward.value