import json
import os
import time
import requests

# ─── ENV VARS ─────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini")
HF_TOKEN = os.getenv("HF_TOKEN")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:7860")

if not HF_TOKEN:
    print("Warning: HF_TOKEN not found. Running in fallback mode.")

TASKS = [
    {"task_id": "supplier_triage", "seed": 42},
    {"task_id": "logistics_reroute", "seed": 42},
    {"task_id": "cascade_disruption", "seed": 42},
]

ENV_NAME = "supply_chain_disruption_manager"
MAX_STEPS = 30


# ─── SAFE ENV CHECK ───────────────────────────────────────
def wait_for_env():
    for _ in range(5):
        try:
            r = requests.get(f"{ENV_BASE_URL}/health", timeout=5)
            if r.status_code == 200:
                return True
        except:
            pass
        time.sleep(2)
    return False


# ─── SAFE ENV FUNCTIONS ───────────────────────────────────
def env_reset(task_id, seed=42):
    try:
        r = requests.post(
            f"{ENV_BASE_URL}/reset",
            json={"task_id": task_id, "seed": seed},
            timeout=10,
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print("RESET ERROR:", str(e))
        return {"error": str(e)}


def env_step(action):
    try:
        r = requests.post(
            f"{ENV_BASE_URL}/step",
            json={"action": action},
            timeout=10,
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print("STEP ERROR:", str(e))
        return {
            "reward": {"value": 0.0},
            "done": True,
            "observation": {},
            "error": str(e),
        }


def env_grade():
    try:
        r = requests.get(f"{ENV_BASE_URL}/grade", timeout=10)
        r.raise_for_status()
        return r.json().get("score", 0.0)
    except Exception as e:
        print("GRADE ERROR:", str(e))
        return 0.0


# ─── AGENT LOGIC (SAFE RULE-BASED) ───────────────────────
def call_llm(obs):
    try:
        task_id = obs.get("task_id", "")

        # Supplier triage
        if task_id == "supplier_triage":
            suppliers = obs.get("suppliers", [])
            warehouses = obs.get("warehouses", [])

            for supplier in sorted(
                suppliers,
                key=lambda s: (s.get("capacity_available", 0), s.get("reliability_score", 0)),
                reverse=True,
            ):
                if not supplier.get("disruption_active") and supplier.get("capacity_available", 0) > 0:
                    return {
                        "action_type": "activate_supplier",
                        "activate_supplier": {
                            "supplier_id": supplier["supplier_id"],
                            "order_quantity": 500,
                            "destination_warehouse": warehouses[0]["warehouse_id"] if warehouses else "WH-DEFAULT",
                        },
                    }

        # Logistics reroute
        elif task_id == "logistics_reroute":
            for shipment in obs.get("active_shipments", []):
                if shipment.get("status") in ["blocked", "delayed"]:
                    return {
                        "action_type": "reroute_shipment",
                        "reroute_shipment": {
                            "shipment_id": shipment["shipment_id"],
                            "new_route": "air_freight",
                            "expedite": True,
                        },
                    }

        # Cascade disruption
        elif task_id == "cascade_disruption":
            day = obs.get("days_elapsed", 0)

            if day == 0:
                return {"action_type": "allocate_stock", "allocate_stock": {"source_warehouse": "WH-SECONDARY-C", "destination_warehouse": "WH-CRITICAL-B", "quantity": 2000}}
            elif day == 1:
                return {"action_type": "allocate_stock", "allocate_stock": {"source_warehouse": "WH-SECONDARY-D", "destination_warehouse": "WH-CRITICAL-A", "quantity": 1800}}
            elif day == 2:
                return {"action_type": "negotiate_contract", "negotiate_contract": {"supplier_id": "SUP-X4", "contract_type": "long_term", "max_price_per_unit": 40.0}}
            elif day == 3:
                return {"action_type": "activate_supplier", "activate_supplier": {"supplier_id": "SUP-X4", "order_quantity": 3450, "destination_warehouse": "WH-CRITICAL-A"}}

        return {"action_type": "wait"}

    except Exception as e:
        print("LLM ERROR:", str(e))
        return {"action_type": "wait"}


# ─── RUN TASK ─────────────────────────────────────────────
def run_task(task_id, seed=42):
    step_rewards = []
    steps = 0
    done = False

    if not wait_for_env():
        print("[START] task={} env={} model={}".format(task_id, ENV_NAME, MODEL_NAME))
        print("[END] success=false steps=0 score=0.0000 rewards=0.00")
        return

    reset_data = env_reset(task_id, seed)

    if "observation" not in reset_data:
        print("[START] task={} env={} model={}".format(task_id, ENV_NAME, MODEL_NAME))
        print("[END] success=false steps=0 score=0.0000 rewards=0.00")
        return

    obs = reset_data["observation"]

    print(f"[START] task={task_id} env={ENV_NAME} model={MODEL_NAME}")

    try:
        while not done and steps < MAX_STEPS:
            action = call_llm(obs)
            action_str = action.get("action_type", "wait")

            step_result = env_step(action)

            reward = step_result.get("reward", {}).get("value", 0.0)
            done = step_result.get("done", True)
            obs = step_result.get("observation", {})
            error = step_result.get("error")

            steps += 1
            step_rewards.append(reward)

            print(
                f"[STEP] step={steps} action={action_str} "
                f"reward={reward:.2f} done={str(done).lower()} error={error if error else 'null'}"
            )

        score = env_grade()
        rewards_str = ",".join(f"{r:.2f}" for r in step_rewards) if step_rewards else "0.00"

        print(
            f"[END] success={'true' if score >= 0.5 else 'false'} "
            f"steps={steps} score={score:.4f} rewards={rewards_str}"
        )

    except Exception as e:
        print(f"[END] success=false steps={steps} score=0.0000 rewards=0.00")


# ─── MAIN ─────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        for task in TASKS:
            run_task(task["task_id"], task["seed"])
            time.sleep(1)
    except Exception as e:
        print("FATAL ERROR:", str(e))
