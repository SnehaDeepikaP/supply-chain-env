"""
Task 1: Supplier Triage (Easy)
-------------------------------
Scenario: A key supplier (30% of supply) has suddenly gone offline due to a factory fire.
The agent must identify and activate the best available alternative supplier within budget
to prevent a stockout at the main warehouse.

Difficulty: EASY
- Single disruption event
- 3 alternative suppliers clearly ranked by cost/reliability
- 7-day horizon
- Success = stockout avoided AND budget not exceeded by >20%

Score: 0.0–1.0
- 0.0  = stockout occurred, no action taken
- 0.4  = alternative activated but wrong choice (high cost or low reliability)
- 0.7  = good alternative chosen, partial demand met
- 1.0  = optimal supplier chosen, full demand met, within budget
"""
import random
from typing import Any, Dict, List, Tuple
from models import (
    DisruptionEvent, Observation, SupplierStatus, WarehouseStatus,
    ShipmentStatus, Reward, Action
)


TASK_ID = "supplier_triage"
MAX_DAYS = 7
TOTAL_BUDGET = 50_000.0

# Suppliers: primary is down, three alternatives
SUPPLIERS_TEMPLATE = [
    {
        "supplier_id": "SUP-001",
        "name": "PrimeCo Manufacturing",
        "country": "China",
        "capacity_available": 0.0,  # offline
        "lead_time_days": 0,
        "cost_per_unit": 12.0,
        "reliability_score": 0.95,
        "disruption_active": True,
        "disruption_reason": "Factory fire — offline indefinitely",
    },
    {
        "supplier_id": "SUP-002",
        "name": "FastShip Vietnam",
        "country": "Vietnam",
        "capacity_available": 0.9,
        "lead_time_days": 3,
        "cost_per_unit": 15.0,
        "reliability_score": 0.88,
        "disruption_active": False,
    },
    {
        "supplier_id": "SUP-003",
        "name": "QuickMake Mexico",
        "country": "Mexico",
        "capacity_available": 1.0,
        "lead_time_days": 2,
        "cost_per_unit": 18.0,
        "reliability_score": 0.92,
        "disruption_active": False,
    },
    {
        "supplier_id": "SUP-004",
        "name": "BudgetBuild India",
        "country": "India",
        "capacity_available": 0.6,
        "lead_time_days": 6,
        "cost_per_unit": 10.0,
        "reliability_score": 0.65,
        "disruption_active": False,
    },
]

WAREHOUSES_TEMPLATE = [
    {
        "warehouse_id": "WH-MAIN",
        "location": "Chicago, US",
        "current_stock": 800,
        "capacity": 5000,
        "demand_forecast": 2000,
        "days_of_stock_remaining": 2.8,
    }
]

DISRUPTION_TEMPLATE = [
    {
        "event_id": "EVT-001",
        "event_type": "supplier_failure",
        "affected_entity": "SUP-001",
        "severity": 0.9,
        "estimated_duration_days": 30,
        "description": "Factory fire at PrimeCo Manufacturing. Production halted.",
    }
]


class Task1SupplierTriage:
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = random.Random(seed)
        self.reset()

    def reset(self) -> Observation:
        self.rng = random.Random(self.seed)
        self.step_count = 0
        self.days_elapsed = 0
        self.budget_remaining = TOTAL_BUDGET
        self.stockout_occurred = False
        self.demand_met = 0
        self.total_demand = 2000
        self.activated_supplier: Dict[str, Any] = {}
        self.action_history: List[str] = []

        self.suppliers = [SupplierStatus(**s) for s in SUPPLIERS_TEMPLATE]
        self.warehouses = [WarehouseStatus(**w) for w in WAREHOUSES_TEMPLATE]
        self.shipments: List[ShipmentStatus] = []
        self.disruptions = [DisruptionEvent(**d) for d in DISRUPTION_TEMPLATE]

        return self._make_observation()

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict]:
        self.step_count += 1
        self.days_elapsed += 1
        error = None
        penalty = 0.0

        # Process action
        if action.action_type == "activate_supplier":
            result = self._handle_activate(action)
            if result.get("error"):
                error = result["error"]
                penalty = 0.1
        elif action.action_type == "wait":
            pass  # Legitimate but may let stock deplete
        else:
            error = f"Action '{action.action_type}' not relevant to this task"
            penalty = 0.05

        self.action_history.append(action.action_type)

        # Simulate one day passing
        self._simulate_day()

        done = self.days_elapsed >= MAX_DAYS or self.stockout_occurred

        reward = self._compute_reward(penalty, done)
        obs = self._make_observation()
        info = {
            "days_elapsed": self.days_elapsed,
            "demand_met": self.demand_met,
            "stockout": self.stockout_occurred,
            "activated_supplier": self.activated_supplier,
        }
        return obs, reward, done, info

    def _handle_activate(self, action: Action) -> Dict:
        spec = action.activate_supplier
        if not spec:
            return {"error": "activate_supplier payload missing"}

        supplier = next((s for s in self.suppliers if s.supplier_id == spec.supplier_id), None)
        if not supplier:
            return {"error": f"Supplier {spec.supplier_id} not found"}
        if supplier.disruption_active:
            return {"error": f"Supplier {spec.supplier_id} is offline"}
        if spec.order_quantity <= 0:
            return {"error": "Order quantity must be positive"}

        cost = spec.order_quantity * supplier.cost_per_unit
        if cost > self.budget_remaining:
            return {"error": f"Insufficient budget: need ${cost:.0f}, have ${self.budget_remaining:.0f}"}

        # Place order
        self.budget_remaining -= cost
        effective_qty = int(spec.order_quantity * supplier.reliability_score * supplier.capacity_available)
        self.activated_supplier = {
            "supplier_id": spec.supplier_id,
            "quantity": effective_qty,
            "arrive_day": self.days_elapsed + supplier.lead_time_days,
            "cost": cost,
        }
        # Add shipment
        self.shipments.append(ShipmentStatus(
            shipment_id=f"SHP-{self.step_count:03d}",
            origin=supplier.country,
            destination="WH-MAIN",
            quantity=effective_qty,
            status="in_transit",
            eta_days=supplier.lead_time_days,
        ))
        return {"ok": True}

    def _simulate_day(self):
        daily_demand = self.total_demand // MAX_DAYS

        # Deliver shipments that have arrived
        for shp in self.shipments:
            if shp.eta_days is not None:
                shp.eta_days -= 1
                if shp.eta_days <= 0 and shp.status == "in_transit":
                    shp.status = "delivered"
                    self.warehouses[0].current_stock += shp.quantity

        # Consume stock
        stock = self.warehouses[0].current_stock
        if stock >= daily_demand:
            self.warehouses[0].current_stock -= daily_demand
            self.demand_met += daily_demand
        else:
            self.demand_met += stock
            self.warehouses[0].current_stock = 0
            self.stockout_occurred = True

        days_left = MAX_DAYS - self.days_elapsed
        remaining_demand = max(1, daily_demand * days_left)
        self.warehouses[0].demand_forecast = int(remaining_demand)
        self.warehouses[0].days_of_stock_remaining = (
            self.warehouses[0].current_stock / max(daily_demand, 1)
        )

    def _compute_reward(self, penalty: float, done: bool) -> Reward:
        # Service level component
        service_level = min(1.0, self.demand_met / self.total_demand)

        # Cost efficiency component
        budget_used = TOTAL_BUDGET - self.budget_remaining
        cost_efficiency = max(0.0, 1.0 - (budget_used / TOTAL_BUDGET) * 0.5) if budget_used > 0 else 0.5

        # Resilience component: reward speed of activation
        if self.activated_supplier:
            arrive_day = self.activated_supplier.get("arrive_day", MAX_DAYS)
            resilience = max(0.0, 1.0 - (arrive_day / MAX_DAYS))
        else:
            resilience = 0.0

        # Supplier quality bonus
        supplier_quality_bonus = 0.0
        if self.activated_supplier:
            sid = self.activated_supplier.get("supplier_id")
            supplier = next((s for s in self.suppliers if s.supplier_id == sid), None)
            if supplier:
                # SUP-003 is optimal (best lead time + reliability)
                if sid == "SUP-003":
                    supplier_quality_bonus = 0.1
                elif sid == "SUP-002":
                    supplier_quality_bonus = 0.05

        weights = {"service": 0.5, "cost": 0.2, "resilience": 0.3}
        base = (
            weights["service"] * service_level
            + weights["cost"] * cost_efficiency
            + weights["resilience"] * resilience
        )
        value = min(1.0, max(0.0, base + supplier_quality_bonus - penalty))

        return Reward(
            value=round(value, 4),
            service_level_component=round(service_level, 4),
            cost_efficiency_component=round(cost_efficiency, 4),
            resilience_component=round(resilience, 4),
            penalty=round(penalty, 4),
            breakdown={
                "demand_met": self.demand_met,
                "total_demand": self.total_demand,
                "budget_used": round(TOTAL_BUDGET - self.budget_remaining, 2),
                "supplier_quality_bonus": supplier_quality_bonus,
            }
        )

    def _make_observation(self) -> Observation:
        service_level = min(1.0, self.demand_met / max(self.total_demand, 1))
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
            stockout_count=1 if self.stockout_occurred else 0,
            service_level=round(service_level, 4),
        )

    def grade(self) -> float:
        """Final grade 0.0–1.0 for this episode."""
        reward = self._compute_reward(0.0, True)
        return reward.value
