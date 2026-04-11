"""
FastAPI application exposing the OpenEnv HTTP API for Supply Chain Disruption Manager.

Endpoints:
  POST /reset          → Initialize episode
  POST /step           → Execute action
  GET  /state          → Current observation
  GET  /tasks          → List available tasks
  GET  /grade          → Episode final score
  GET  /health         → Health check
  GET  /               → Environment info
"""
import os
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from env import SupplyChainEnv
from models import Action, Observation, Reward, StepResponse

app = FastAPI(
    title="Supply Chain Disruption Manager",
    description=(
        "OpenEnv-compliant environment for training RL agents on real-world "
        "supply chain disruption management. "
        "3 tasks: supplier triage (easy), logistics reroute (medium), cascading disruption (hard)."
    ),
    version="1.0.0",
    tags_metadata=[
        {"name": "openenv", "description": "OpenEnv standard endpoints"},
    ],
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Single shared environment instance per process
# For production, use session-based isolation
env = SupplyChainEnv()


# ─── Request/Response schemas ────────────────────────────────────────────────

class ResetRequest(BaseModel):
    task_id: str = "supplier_triage"
    seed: int = 42


class StepRequest(BaseModel):
    action: Action


class ResetResponse(BaseModel):
    observation: Observation
    task_info: Dict[str, Any]


# ─── Endpoints ───────────────────────────────────────────────────────────────

@app.get("/", tags=["info"])
def root():
    return {
        "name": "Supply Chain Disruption Manager",
        "version": "1.0.0",
        "openenv": True,
        "tasks": list(env.list_tasks().keys()),
        "spec": "https://github.com/meta-pytorch/OpenEnv",
    }


@app.get("/health", tags=["info"])
def health():
    return {"status": "ok", "env": "supply_chain_disruption_manager"}


@app.get("/tasks", tags=["openenv"])
def list_tasks():
    """List all available tasks with metadata."""
    return env.list_tasks()


@app.post("/reset", response_model=ResetResponse, tags=["openenv"])
def reset(request: Optional[ResetRequest] = None):
    try:
        if request is None:
            # Default behavior (required for OpenEnv)
            task_id = "supplier_triage"
            seed = 42
        else:
            task_id = request.task_id
            seed = request.seed

        obs = env.reset(task_id=task_id, seed=seed)
        task_info = env.list_tasks().get(task_id, {})

        return ResetResponse(observation=obs, task_info=task_info)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/step", response_model=StepResponse, tags=["openenv"])
def step(request: StepRequest):
    """
    Execute one action.

    **action_type** options:
    - `activate_supplier` — Order from a supplier (requires `activate_supplier` payload)
    - `reroute_shipment` — Reroute a blocked shipment (requires `reroute_shipment` payload)
    - `allocate_stock` — Transfer stock between warehouses (requires `allocate_stock` payload)
    - `negotiate_contract` — Negotiate contract terms (requires `negotiate_contract` payload)
    - `wait` — Do nothing this step

    **Example — activate supplier:**
    ```json
    {
      "action": {
        "action_type": "activate_supplier",
        "activate_supplier": {
          "supplier_id": "SUP-003",
          "order_quantity": 2000,
          "destination_warehouse": "WH-MAIN"
        },
        "reasoning": "SUP-003 has best lead time and reliability"
      }
    }
    ```
    """
    try:
        response = env.step(request.action)
        return response
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/state", response_model=Observation, tags=["openenv"])
def state():
    """Return current environment state without advancing the episode."""
    try:
        return env.state()
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/grade", tags=["openenv"])
def grade():
    """Return the final score for the current episode (0.0–1.0)."""
    try:
        score = env.grade()
        return {"score": score, "range": [0.0, 1.0]}
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 7860))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
