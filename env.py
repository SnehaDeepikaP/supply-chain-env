"""
SupplyChainEnv — Core environment implementing the OpenEnv interface.

API:
  reset(task_id, seed)  → Observation
  step(action)          → (Observation, Reward, done, info)
  state()               → current Observation
"""
import uuid
from typing import Any, Dict, Optional, Tuple

from models import Action, Observation, Reward, StepResponse
from tasks.task1_supplier_triage import Task1SupplierTriage
from tasks.task2_logistics_reroute import Task2LogisticsReroute
from tasks.task3_cascade_disruption import Task3CascadeDisruption

VALID_TASKS = {
    "supplier_triage": Task1SupplierTriage,
    "logistics_reroute": Task2LogisticsReroute,
    "cascade_disruption": Task3CascadeDisruption,
}


class SupplyChainEnv:
    """
    OpenEnv-compliant environment for Supply Chain Disruption Management.

    The agent interacts with a simulated supply chain under disruption.
    It must activate suppliers, reroute shipments, allocate stock, and
    negotiate contracts to maximize service level and cost efficiency.
    """

    def __init__(self):
        self._task = None
        self._task_id: Optional[str] = None
        self._episode_id: Optional[str] = None
        self._current_obs: Optional[Observation] = None
        self._done: bool = False
        self._step_count: int = 0
        self._last_error: Optional[str] = None

    def reset(self, task_id: str = "supplier_triage", seed: int = 42) -> Observation:
        """
        Initialize a new episode.

        Args:
            task_id: One of 'supplier_triage', 'logistics_reroute', 'cascade_disruption'
            seed: Random seed for reproducibility

        Returns:
            Initial Observation
        """
        if task_id not in VALID_TASKS:
            raise ValueError(
                f"Unknown task '{task_id}'. Valid tasks: {list(VALID_TASKS.keys())}"
            )

        self._task_id = task_id
        self._episode_id = str(uuid.uuid4())[:8]
        self._done = False
        self._step_count = 0
        self._last_error = None

        task_cls = VALID_TASKS[task_id]
        self._task = task_cls(seed=seed)
        self._current_obs = self._task.reset()
        return self._current_obs

    def step(self, action: Action) -> StepResponse:
        """
        Execute one action in the environment.

        Args:
            action: Action pydantic model

        Returns:
            StepResponse with observation, reward, done flag, info dict
        """
        if self._task is None:
            raise RuntimeError("Call reset() before step()")
        if self._done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")

        obs, reward, done, info = self._task.step(action)
        self._current_obs = obs
        self._done = done
        self._step_count += 1
        self._last_error = info.get("error")

        return StepResponse(
            observation=obs,
            reward=reward,
            done=done,
            info=info,
            error=self._last_error,
        )

    def state(self) -> Observation:
        """Return current observation without advancing the episode."""
        if self._current_obs is None:
            raise RuntimeError("Call reset() first")
        return self._current_obs

    def grade(self) -> float:
        """Return final episode score 0.0–1.0."""
        if self._task is None:
            raise RuntimeError("No episode to grade. Call reset() + run episode first.")
        return self._task.grade()

    def list_tasks(self) -> Dict[str, Any]:
        """Return metadata about all available tasks."""
        return {
            "supplier_triage": {
                "difficulty": "easy",
                "horizon_days": 7,
                "budget": 50_000,
                "description": "Activate best alternative supplier after primary goes offline.",
                "key_actions": ["activate_supplier", "wait"],
            },
            "logistics_reroute": {
                "difficulty": "medium",
                "horizon_days": 14,
                "budget": 120_000,
                "description": "Reroute blocked shipments past a closed port; reallocate stock between warehouses.",
                "key_actions": ["reroute_shipment", "allocate_stock", "activate_supplier"],
            },
            "cascade_disruption": {
                "difficulty": "hard",
                "horizon_days": 21,
                "budget": 200_000,
                "description": "Manage cascading multi-supplier failure, port closure, demand spike, and hidden mid-episode shocks.",
                "key_actions": ["activate_supplier", "negotiate_contract", "allocate_stock", "reroute_shipment"],
            },
        }
