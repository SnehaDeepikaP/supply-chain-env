"""
Pydantic typed models for Supply Chain Disruption Manager.
OpenEnv spec requires: Observation, Action, Reward models.
"""
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ─── Observation ────────────────────────────────────────────────────────────

class SupplierStatus(BaseModel):
    supplier_id: str
    name: str
    country: str
    capacity_available: float = Field(ge=0.0, le=1.0, description="Fraction of capacity available (0=offline, 1=full)")
    lead_time_days: int
    cost_per_unit: float
    reliability_score: float = Field(ge=0.0, le=1.0)
    disruption_active: bool
    disruption_reason: Optional[str] = None


class WarehouseStatus(BaseModel):
    warehouse_id: str
    location: str
    current_stock: int
    capacity: int
    demand_forecast: int = Field(description="Units needed in next 7 days")
    days_of_stock_remaining: float


class ShipmentStatus(BaseModel):
    shipment_id: str
    origin: str
    destination: str
    quantity: int
    status: str  # "in_transit", "delayed", "blocked", "delivered"
    eta_days: Optional[int]
    delay_days: int = 0


class DisruptionEvent(BaseModel):
    event_id: str
    event_type: str  # "port_closure", "supplier_failure", "demand_spike", "weather", "strike"
    affected_entity: str
    severity: float = Field(ge=0.0, le=1.0)
    estimated_duration_days: int
    description: str


class Observation(BaseModel):
    step: int
    task_id: str
    episode_id: str
    suppliers: List[SupplierStatus]
    warehouses: List[WarehouseStatus]
    active_shipments: List[ShipmentStatus]
    disruption_events: List[DisruptionEvent]
    budget_remaining: float
    total_budget: float
    days_elapsed: int
    max_days: int
    stockout_count: int = 0
    service_level: float = Field(ge=0.0, le=1.0, description="Fraction of demand met so far")
    info: Dict[str, Any] = Field(default_factory=dict)


# ─── Action ─────────────────────────────────────────────────────────────────

class ActivateSupplierAction(BaseModel):
    supplier_id: str
    order_quantity: int
    destination_warehouse: str


class RerouteShipmentAction(BaseModel):
    shipment_id: str
    new_route: str
    expedite: bool = False


class AllocateStockAction(BaseModel):
    source_warehouse: str
    destination_warehouse: str
    quantity: int


class NegotiateContractAction(BaseModel):
    supplier_id: str
    contract_type: str  # "emergency", "spot", "long_term"
    max_price_per_unit: float


class Action(BaseModel):
    action_type: str = Field(
        description="One of: activate_supplier, reroute_shipment, allocate_stock, negotiate_contract, wait"
    )
    activate_supplier: Optional[ActivateSupplierAction] = None
    reroute_shipment: Optional[RerouteShipmentAction] = None
    allocate_stock: Optional[AllocateStockAction] = None
    negotiate_contract: Optional[NegotiateContractAction] = None
    reasoning: Optional[str] = Field(None, description="Agent's reasoning for this action")


# ─── Reward ─────────────────────────────────────────────────────────────────

class Reward(BaseModel):
    value: float = Field(ge=0.0, le=1.0, description="Normalized reward for this step")
    service_level_component: float = Field(ge=0.0, le=1.0)
    cost_efficiency_component: float = Field(ge=0.0, le=1.0)
    resilience_component: float = Field(ge=0.0, le=1.0)
    penalty: float = Field(ge=0.0, le=1.0, description="Penalty for bad actions (subtracted)")
    breakdown: Dict[str, Any] = Field(default_factory=dict)


# ─── Step Response ───────────────────────────────────────────────────────────

class StepResponse(BaseModel):
    observation: Observation
    reward: Reward
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None
