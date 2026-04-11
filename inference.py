import json
import os
import time
import requests
from openai import OpenAI

# ─── ENV ─────────────────────────────────────────────────
API_BASE_URL = os.environ.get("API_BASE_URL")
API_KEY = os.environ.get("API_KEY")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4.1-mini")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:7860")

client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

TASKS = [
    {"task_id": "supplier_triage", "seed": 42},
    {"task_id": "logistics_reroute", "seed": 42},
    {"task_id": "cascade_disruption", "seed": 42},
]

ENV_NAME = "supply_chain_disruption_manager"
MAX_STEPS = 30


# ─── SAFE ENV ─────────────────────────────────────────────
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


def env_reset(task_id, seed):
    try:
        r = requests.post(f"{ENV_BASE_URL}/reset", json={"task_id": task_id, "seed": seed}, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print("RESET ERROR:", e)
        return {"error": str(e)}


def env_step(action):
    try:
        r = requests.post(f"{ENV_BASE_URL}/step", json={"action": action}, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print("STEP ERROR:", e)
        return {"reward": {"value": 0.0}, "done": True, "observation": {}, "error": str(e)}


def env_grade():
    try:
        r = requests.get(f"{ENV_BASE_URL}/grade", timeout=10)
        r.raise_for_status()
        return r.json().get("score", 0.0)
    except:
        return 0.0


# ─── LLM + FALLBACK ──────────────────────────────────────
def fallback_logic(obs):
    return {"action_type": "wait"}


def call_llm(obs):
    try:
        prompt = f"""
        You are a supply chain optimizer.

        Observation:
        {json.dumps(obs)[:2000]}

        Decide next best action.
        Return ONLY JSON with action_type and parameters.
        """

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a supply chain optimizer."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=200,
        )

        text = response.choices[0].message.content.strip()

        try:
            action = json.loads(text)
            if "action_type" in action:
                return action
        except:
            pass

        return fallback_logic(obs)

    except Exception as e:
        print("LLM ERROR:", e)
        return fallback_logic(obs)


# ─── RUN TASK ─────────────────────────────────────────────
def run_task(task_id, seed):
    step_rewards = []
    steps = 0
    done = False

    print(f"[START] task={task_id} env={ENV_NAME} model={MODEL_NAME}")

    if not wait_for_env():
        print(f"[END] success=false steps=0 score=0.0000 rewards=0.00")
        return {"task_id": task_id, "score": 0.0, "success": False}

    reset_data = env_reset(task_id, seed)

    if "observation" not in reset_data:
        print(f"[END] success=false steps=0 score=0.0000 rewards=0.00")
        return {"task_id": task_id, "score": 0.0, "success": False}

    obs = reset_data["observation"]

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
                f"reward={reward:.2f} done={str(done).lower()} "
                f"error={error if error else 'null'}"
            )

        score = env_grade()
        rewards_str = ",".join(f"{r:.2f}" for r in step_rewards) if step_rewards else "0.00"

        print(
            f"[END] success={'true' if score >= 0.5 else 'false'} "
            f"steps={steps} score={score:.4f} rewards={rewards_str}"
        )

        return {
            "task_id": task_id,
            "score": score,
            "success": score >= 0.5
        }

    except Exception as e:
        print(f"[END] success=false steps={steps} score=0.0000 rewards=0.00")
        return {"task_id": task_id, "score": 0.0, "success": False}


# ─── MAIN ─────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        results = []

        for task in TASKS:
            res = run_task(task["task_id"], task["seed"])
            results.append(res)
            time.sleep(1)

        # SUMMARY
        print("\n=== SUMMARY ===")
        scores = []

        for r in results:
            print(f"{r['task_id']}: score={r['score']:.4f} success={r['success']}")
            scores.append(r["score"])

        overall = sum(scores) / len(scores) if scores else 0.0
        print(f"overall_score={overall:.4f}")

    except Exception as e:
        print("FATAL ERROR:", e)
