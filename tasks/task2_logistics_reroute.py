"""
Task 2: Logistics Reroute (Medium)
------------------------------------
Scenario: A major port (Rotterdam) is closed due to a strike. 14 days horizon.
Three shipments are blocked. The agent must reroute them via alternative ports,
manage cost overruns, and reallocate existing stock between warehouses to
prevent stockouts at high-demand locations.

Difficulty: MEDIUM
- Multiple simultaneous disruptions (port closure + one supplier at 50%)
- Agent must balance rerouting cost vs stockout risk
- Stock reallocation between warehouses adds complexity
- 14-day horizon with cascading demand changes

Score: 0.0–1.0
- Stockout prevention weight: 40%
- Cost efficiency weight: 30%
- Route optimality weight: 30%
"""
import random
from typing import Any, Dict, List, Tuple
from models import (
    DisruptionEvent, Observation, SupplierStatus, WarehouseStatus,
    ShipmentStatus, Reward, Action
)

TASK_ID = "logistics_reroute"
MAX_DAYS = 14
TOTAL_BUDGET = 120_000.0

SUPPLIERS_TEMPLATE = [
    {
        "supplier_id": "SUP-A",
        "name": "EuroMake GmbH",
        "country": "Germany",
        "capacity_available": 0.5,  # partially disrupted
        "lead_time_days": 5,
        "cost_per_unit": 20.0,
        "reliability_score": 0.85,
        "disruption_active": True,
        "disruption_reason": "Labour shortage — 50% capacity",
    },
    {
        "supplier_id": "SUP-B",
        "name": "AsiaLink Singapore",
        "country": "Singapore",
        "capacity_available": 1.0,
        "lead_time_days": 8,
        "cost_per_unit": 17.0,
        "reliability_score": 0.93,
        "disruption_active": False,
    },
]

WAREHOUSES_TEMPLATE = [
    {
        "warehouse_id": "WH-EU-WEST",
        "location": "Paris, France",
        "current_stock": 1500,
        "capacity": 8000,
        "demand_forecast": 3500,
        "days_of_stock_remaining": 6.0,
    },
    {
        "warehouse_id": "WH-EU-NORTH",
        "location": "Stockholm, Sweden",
        "current_stock": 400,
        "capacity": 4000,
        "demand_forecast": 2200,
        "days_of_stock_remaining": 2.5,
    },
    {
        "warehouse_id": "WH-EU-EAST",
        "location": "Warsaw, Poland",
        "current_stock": 3000,
        "capacity": 6000,
        "demand_forecast": 1000,
        "days_of_stock_remaining": 21.0,
    },
]

SHIPMENTS_TEMPLATE = [
    {
        "shipment_id": "SHP-ROT-001",
        "origin": "Singapore",
        "destination": "WH-EU-NORTH",
        "quantity": 1800,
        "status": "blocked",
        "eta_days": None,
        "delay_days": 0,
    },
    {
        "shipment_id": "SHP-ROT-002",
        "origin": "Germany",
        "destination": "WH-EU-WEST",
        "quantity": 1200,
        "status": "blocked",
        "eta_days": None,
        "delay_days": 0,
    },
    {
        "shipment_id": "SHP-ROT-003",
        "origin": "Singapore",
        "destination": "WH-EU-WEST",
        "quantity": 900,
        "status": "blocked",
        "eta_days": None,
        "delay_days": 0,
    },
]

DISRUPTIONS_TEMPLATE = [
    {
        "event_id": "EVT-PORT-01",
        "event_type": "port_closure",
        "affected_entity": "Port of Rotterdam",
        "severity": 0.95,
        "estimated_duration_days": 10,
        "description": "Dockers' strike — Rotterdam port fully closed. All inbound/outbound halted.",
    },
    {
        "event_id": "EVT-SUP-01",
        "event_type": "supplier_failure",
        "affected_entity": "SUP-A",
        "severity": 0.5,
        "estimated_duration_days": 7,
        "description": "Labour shortage at EuroMake — running at 50% capacity.",
    },
]

# Reroute options: route_name -> {extra_days, extra_cost_per_shipment, works_for}
REROUTE_OPTIONS = {
    "via_antwerp": {"extra_days": 2, "extra_cost": 3000, "available": True},
    "via_hamburg": {"extra_days": 3, "extra_cost": 2500, "available": True},
    "via_le_havre": {"extra_days": 4, "extra_cost": 4000, "available": True},
    "air_freight":  {"extra_days": 1, "extra_cost": 15000, "available": True},
}


class Task2LogisticsReroute:
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = random.Random(seed)
        self.reset()

    def reset(self) -> Observation:
        self.rng = random.Random(self.seed)
        self.step_count = 0
        self.days_elapsed = 0
        self.budget_remaining = TOTAL_BUDGET
        self.stockouts: Dict[str, int] = {"WH-EU-WEST": 0, "WH-EU-NORTH": 0, "WH-EU-EAST": 0}
        self.demand_met: Dict[str, int] = {"WH-EU-WEST": 0, "WH-EU-NORTH": 0, "WH-EU-EAST": 0}
        self.total_demand: Dict[str, int] = {"WH-EU-WEST": 3500, "WH-EU-NORTH": 2200, "WH-EU-EAST": 1000}
        self.reroute_actions: List[Dict] = []

        self.suppliers = [SupplierStatus(**s) for s in SUPPLIERS_TEMPLATE]
        self.warehouses = [WarehouseStatus(**w) for w in WAREHOUSES_TEMPLATE]
        self.shipments = [ShipmentStatus(**s) for s in SHIPMENTS_TEMPLATE]
        self.disruptions = [DisruptionEvent(**d) for d in DISRUPTIONS_TEMPLATE]
        return self._make_observation()

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict]:
        self.step_count += 1
        self.days_elapsed += 1
        error = None
        penalty = 0.0

        if action.action_type == "reroute_shipment":
            result = self._handle_reroute(action)
            if result.get("error"):
                error = result["error"]
                penalty = 0.08
        elif action.action_type == "allocate_stock":
            result = self._handle_allocate(action)
            if result.get("error"):
                error = result["error"]
                penalty = 0.08
        elif action.action_type == "activate_supplier":
            result = self._handle_activate(action)
            if result.get("error"):
                error = result["error"]
                penalty = 0.05
        elif action.action_type == "wait":
            pass
        else:
            penalty = 0.05

        self._simulate_day()

        all_served = all(
            self.demand_met[wh] >= self.total_demand[wh]
            for wh in self.total_demand
        )
        done = self.days_elapsed >= MAX_DAYS or all_served

        reward = self._compute_reward(penalty, done)
        obs = self._make_observation()
        info = {
            "reroutes_done": len(self.reroute_actions),
            "stockouts": self.stockouts,
            "demand_met": self.demand_met,
        }
        return obs, reward, done, info

    def _get_warehouse(self, wh_id: str):
        return next((w for w in self.warehouses if w.warehouse_id == wh_id), None)

    def _handle_reroute(self, action: Action) -> Dict:
        spec = action.reroute_shipment
        if not spec:
            return {"error": "reroute_shipment payload missing"}

        shipment = next((s for s in self.shipments if s.shipment_id == spec.shipment_id), None)
        if not shipment:
            return {"error": f"Shipment {spec.shipment_id} not found"}
        if shipment.status not in ("blocked", "in_transit"):
            return {"error": f"Shipment {spec.shipment_id} cannot be rerouted (status: {shipment.status})"}

        route = REROUTE_OPTIONS.get(spec.new_route)
        if not route:
            return {"error": f"Unknown route '{spec.new_route}'. Options: {list(REROUTE_OPTIONS.keys())}"}
        if not route["available"]:
            return {"error": f"Route '{spec.new_route}' is currently unavailable"}

        extra_cost = route["extra_cost"]
        if spec.expedite:
            extra_cost *= 1.5

        if extra_cost > self.budget_remaining:
            return {"error": f"Insufficient budget for reroute: need ${extra_cost:.0f}"}

        self.budget_remaining -= extra_cost
        shipment.status = "in_transit"
        shipment.eta_days = route["extra_days"] + (1 if not spec.expedite else 0)
        shipment.delay_days = route["extra_days"]
        self.reroute_actions.append({
            "shipment_id": spec.shipment_id,
            "route": spec.new_route,
            "cost": extra_cost,
        })
        return {"ok": True, "new_eta": shipment.eta_days}

    def _handle_allocate(self, action: Action) -> Dict:
        spec = action.allocate_stock
        if not spec:
            return {"error": "allocate_stock payload missing"}

        src = self._get_warehouse(spec.source_warehouse)
        dst = self._get_warehouse(spec.destination_warehouse)
        if not src:
            return {"error": f"Source warehouse {spec.source_warehouse} not found"}
        if not dst:
            return {"error": f"Destination warehouse {spec.destination_warehouse} not found"}
        if spec.quantity <= 0:
            return {"error": "Quantity must be positive"}
        if spec.quantity > src.current_stock:
            return {"error": f"Insufficient stock at {spec.source_warehouse}: have {src.current_stock}"}

        transfer_cost = spec.quantity * 2.0  # $2 per unit transfer cost
        if transfer_cost > self.budget_remaining:
            return {"error": f"Insufficient budget for transfer: need ${transfer_cost:.0f}"}

        self.budget_remaining -= transfer_cost
        src.current_stock -= spec.quantity
        dst.current_stock = min(dst.capacity, dst.current_stock + spec.quantity)
        return {"ok": True}

    def _handle_activate(self, action: Action) -> Dict:
        spec = action.activate_supplier
        if not spec:
            return {"error": "activate_supplier payload missing"}
        supplier = next((s for s in self.suppliers if s.supplier_id == spec.supplier_id), None)
        if not supplier:
            return {"error": "Supplier not found"}

        cost = spec.order_quantity * supplier.cost_per_unit
        if cost > self.budget_remaining:
            return {"error": "Insufficient budget"}

        self.budget_remaining -= cost
        effective_qty = int(spec.order_quantity * supplier.capacity_available * supplier.reliability_score)
        self.shipments.append(ShipmentStatus(
            shipment_id=f"SHP-NEW-{self.step_count:03d}",
            origin=supplier.country,
            destination=spec.destination_warehouse,
            quantity=effective_qty,
            status="in_transit",
            eta_days=supplier.lead_time_days,
        ))
        return {"ok": True}

    def _simulate_day(self):
        daily_demands = {
            "WH-EU-WEST": self.total_demand["WH-EU-WEST"] // MAX_DAYS,
            "WH-EU-NORTH": self.total_demand["WH-EU-NORTH"] // MAX_DAYS,
            "WH-EU-EAST": self.total_demand["WH-EU-EAST"] // MAX_DAYS,
        }
        # Deliver arrived shipments
        for shp in self.shipments:
            if shp.eta_days is not None and shp.status == "in_transit":
                shp.eta_days -= 1
                if shp.eta_days <= 0:
                    shp.status = "delivered"
                    wh = self._get_warehouse(shp.destination)
                    if wh:
                        wh.current_stock = min(wh.capacity, wh.current_stock + shp.quantity)

        # Consume stock
        for wh in self.warehouses:
            daily = daily_demands.get(wh.warehouse_id, 0)
            if wh.current_stock >= daily:
                wh.current_stock -= daily
                self.demand_met[wh.warehouse_id] += daily
            else:
                self.demand_met[wh.warehouse_id] += wh.current_stock
                self.stockouts[wh.warehouse_id] += 1
                wh.current_stock = 0

            days_left = max(1, MAX_DAYS - self.days_elapsed)
            wh.demand_forecast = daily * days_left
            wh.days_of_stock_remaining = wh.current_stock / max(daily, 1)

    def _compute_reward(self, penalty: float, done: bool) -> Reward:
        # Service level per warehouse, weighted by demand
        total_demand_all = sum(self.total_demand.values())
        service_level = sum(
            self.demand_met[wh] / max(self.total_demand[wh], 1) * (self.total_demand[wh] / total_demand_all)
            for wh in self.total_demand
        )

        # Cost efficiency
        budget_used_frac = (TOTAL_BUDGET - self.budget_remaining) / TOTAL_BUDGET
        cost_efficiency = max(0.0, 1.0 - budget_used_frac)

        # Resilience: reward rerouting blocked shipments quickly
        blocked_count = sum(1 for s in self.shipments if s.status == "blocked")
        total_shp = max(len(SHIPMENTS_TEMPLATE), 1)
        resilience = 1.0 - (blocked_count / total_shp)

        # Stockout penalty
        total_stockout_days = sum(self.stockouts.values())
        stockout_penalty = min(0.3, total_stockout_days * 0.03)

        base = (
            0.4 * service_level
            + 0.3 * cost_efficiency
            + 0.3 * resilience
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
                "demand_met": self.demand_met,
                "stockout_days": self.stockouts,
                "blocked_shipments": blocked_count,
                "budget_used": round(TOTAL_BUDGET - self.budget_remaining, 2),
            }
        )

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
            active_shipments=[s for s in self.shipments if s.status not in ("delivered",)],
            disruption_events=self.disruptions,
            budget_remaining=round(self.budget_remaining, 2),
            total_budget=TOTAL_BUDGET,
            days_elapsed=self.days_elapsed,
            max_days=MAX_DAYS,
            stockout_count=sum(1 for v in self.stockouts.values() if v > 0),
            service_level=round(service_level, 4),
        )

    def grade(self) -> float:
        reward = self._compute_reward(0.0, True)
        return reward.value
