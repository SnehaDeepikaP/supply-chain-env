import json
import os
import time
import re
import requests
from openai import OpenAI

# ─── ENV ────────────────────────────────────────────────
API_BASE_URL = os.environ.get("API_BASE_URL")
API_KEY = os.environ.get("API_KEY")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4.1-mini")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:7860")

client = None

TASKS = [
    {"task_id": "supplier_triage", "seed": 42},
    {"task_id": "logistics_reroute", "seed": 42},
    {"task_id": "cascade_disruption", "seed": 42},
]

ENV_NAME = "supply_chain_disruption_manager"
MAX_STEPS = 30


# ─── CLIENT ─────────────────────────────────────────────
def get_client():
    global client
    if client is None:
        try:
            if API_BASE_URL and API_KEY:
                client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
        except:
            client = None
    return client


# ─── ENV HELPERS ────────────────────────────────────────
def wait_for_env():
    for _ in range(6):
        try:
            if requests.get(f"{ENV_BASE_URL}/health", timeout=5).status_code == 200:
                return True
        except:
            pass
        time.sleep(2)
    return False


def env_reset(task_id, seed):
    try:
        r = requests.post(f"{ENV_BASE_URL}/reset", json={"task_id": task_id, "seed": seed}, timeout=10)
        r.raise_for_status()
        return r.json()
    except:
        return {"error": "reset_failed"}


def env_step(action):
    try:
        r = requests.post(f"{ENV_BASE_URL}/step", json={"action": action}, timeout=10)
        r.raise_for_status()
        return r.json()
    except:
        return {"reward": {"value": 0.0}, "done": True, "observation": {}, "error": "step_failed"}


def env_grade():
    try:
        r = requests.get(f"{ENV_BASE_URL}/grade", timeout=10)
        r.raise_for_status()
        return r.json().get("score", 0.0)
    except:
        return 0.0


# ─── SAFETY ─────────────────────────────────────────────
def safe_action(action):
    if not isinstance(action, dict) or "action_type" not in action:
        return {"action_type": "wait"}
    return action


# ─── SMART POLICY ───────────────────────────────────────
def smart_policy(obs):
    task = obs.get("task_id", "")

    # ───── SUPPLIER TRIAGE ─────
    if task == "supplier_triage":
        suppliers = sorted(
            obs.get("suppliers", []),
            key=lambda s: (s.get("capacity_available", 0), s.get("reliability_score", 0)),
            reverse=True,
        )
        warehouses = obs.get("warehouses", [])

        if warehouses:
            target = min(warehouses, key=lambda w: w.get("current_stock", 9999))

            for s in suppliers:
                if not s.get("disruption_active") and s.get("capacity_available", 0) > 0:
                    return {
                        "action_type": "activate_supplier",
                        "activate_supplier": {
                            "supplier_id": s["supplier_id"],
                            "order_quantity": min(600, int(1000 * s["capacity_available"])),
                            "destination_warehouse": target["warehouse_id"],
                        },
                    }

    # ───── LOGISTICS REROUTE ─────
    if task == "logistics_reroute":
        shipments = obs.get("active_shipments", [])
        for s in shipments:
            if s.get("status") in ["blocked", "delayed"]:
                return {
                    "action_type": "reroute_shipment",
                    "reroute_shipment": {
                        "shipment_id": s["shipment_id"],
                        "new_route": "via_hamburg",
                        "expedite": True,
                    },
                }
        return {"action_type": "wait"}

    # ───── CASCADE DISRUPTION (FINAL FIXED) ─────
    if task == "cascade_disruption":
        days = obs.get("days_elapsed", 0)
        warehouses = obs.get("warehouses", [])
        suppliers = obs.get("suppliers", [])

        critical = [w for w in warehouses if "CRITICAL" in w["warehouse_id"]]
        secondary = [w for w in warehouses if "SECONDARY" in w["warehouse_id"]]

        # PHASE 1: protect critical
        if days <= 2:
            if critical and secondary:
                needy = min(critical, key=lambda w: w["current_stock"])
                rich = max(secondary, key=lambda w: w["current_stock"])

                if rich["current_stock"] > 0:
                    return {
                        "action_type": "allocate_stock",
                        "allocate_stock": {
                            "source_warehouse": rich["warehouse_id"],
                            "destination_warehouse": needy["warehouse_id"],
                            "quantity": min(800, rich["current_stock"]),
                        },
                    }

        # PHASE 2: contracts
        if 2 <= days <= 6:
            for s in suppliers:
                if not s.get("disruption_active"):
                    return {
                        "action_type": "negotiate_contract",
                        "negotiate_contract": {
                            "supplier_id": s["supplier_id"],
                            "contract_type": "long_term",
                            "max_price_per_unit": 120.0,
                        },
                    }

        # PHASE 3: activation (ensure ≥4 shipments)
        if days >= 4:
            for s in suppliers:
                if not s.get("disruption_active") and s.get("capacity_available", 0) > 0:
                    if critical:
                        target = min(critical, key=lambda w: w["current_stock"])
                        return {
                            "action_type": "activate_supplier",
                            "activate_supplier": {
                                "supplier_id": s["supplier_id"],
                                "order_quantity": min(700, int(1000 * s["capacity_available"])),
                                "destination_warehouse": target["warehouse_id"],
                            },
                        }

        return {"action_type": "wait"}

    return {"action_type": "wait"}


# ─── LLM CALL ───────────────────────────────────────────
def call_llm(obs):
    client = get_client()

    if client is None:
        return smart_policy(obs)

    try:
        res = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "Return JSON with action_type"},
                {"role": "user", "content": json.dumps(obs)[:800]},
            ],
            temperature=0.1,
            max_tokens=60,
        )

        text = res.choices[0].message.content.strip()
        match = re.search(r"\{.*\}", text, re.DOTALL)

        if match:
            action = json.loads(match.group())
            if "action_type" in action:
                return action

    except:
        pass

    return smart_policy(obs)


# ─── RUN TASK ───────────────────────────────────────────
def run_task(task_id, seed):
    print(f"[START] task={task_id} env={ENV_NAME} model={MODEL_NAME}")

    if not wait_for_env():
        print("[END] success=false steps=0 score=0.0000 rewards=0.00")
        return {"score": 0.0, "success": False}

    reset_data = env_reset(task_id, seed)
    if "observation" not in reset_data:
        print("[END] success=false steps=0 score=0.0000 rewards=0.00")
        return {"score": 0.0, "success": False}

    obs = reset_data["observation"]
    rewards = []
    steps = 0
    done = False

    try:
        while not done and steps < MAX_STEPS:
            action = safe_action(call_llm(obs))
            action_str = action.get("action_type", "wait")

            result = env_step(action)

            reward = result.get("reward", {}).get("value", 0.0)
            done = result.get("done", True)
            obs = result.get("observation", {})
            error = result.get("error")

            steps += 1
            rewards.append(reward)

            print(
                f"[STEP] step={steps} action={action_str} "
                f"reward={reward:.2f} done={str(done).lower()} "
                f"error={error if error else 'null'}"
            )

        score = env_grade()
        rewards_str = ",".join(f"{r:.2f}" for r in rewards)

        print(
            f"[END] success={'true' if score >= 0.5 else 'false'} "
            f"steps={steps} score={score:.4f} rewards={rewards_str}"
        )

        return {"score": score, "success": score >= 0.5}

    except:
        print("[END] success=false steps=0 score=0.0000 rewards=0.00")
        return {"score": 0.0, "success": False}


# ─── MAIN ───────────────────────────────────────────────
if __name__ == "__main__":
    results = []
    scores = []

    for t in TASKS:
        r = run_task(t["task_id"], t["seed"])
        results.append(r)
        scores.append(r["score"])
        time.sleep(1)

    print("\n=== SUMMARY ===")
    for i, t in enumerate(TASKS):
        print(f"{t['task_id']}: score={scores[i]:.4f} success={results[i]['success']}")

    overall = sum(scores) / len(scores) if scores else 0.0
    print(f"overall_score={overall:.4f}")
