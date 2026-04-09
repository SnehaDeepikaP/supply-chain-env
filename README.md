# Supply Chain Disruption Manager

**OpenEnv RL Environment — Meta OpenEnv Hackathon**

An AI agent must manage a supply chain under real-world disruptions: supplier failures, port closures, demand spikes, and cascading crises. The agent decides which suppliers to activate, how to reroute shipments, where to reallocate stock, and when to negotiate emergency contracts — all within budget and time constraints.

---

## Why This Domain?

Supply chain disruptions cost the global economy over $4 trillion annually. Existing tools rely on human experts making reactive decisions under incomplete information. This environment provides a rigorous testbed for training RL agents that can learn proactive, adaptive supply chain strategies — directly applicable to real logistics operations.

---

## Environment Overview

| Property | Value |
|---|---|
| Observation space | Structured JSON (suppliers, warehouses, shipments, disruptions, budget) |
| Action space | Discrete typed actions (activate_supplier, reroute_shipment, allocate_stock, negotiate_contract, wait) |
| Reward range | 0.0 – 1.0 per step (partial progress signal throughout) |
| Tasks | 3 (easy → medium → hard) |
| Episode lengths | 7 / 14 / 21 days |

---

## Tasks

### Task 1: Supplier Triage (Easy) — 7 days, $50k budget

**Scenario:** Primary supplier (30% of supply) goes offline due to a factory fire. Three alternative suppliers are available with different lead times, costs, and reliability scores.

**Agent objective:** Activate the best alternative supplier before the main warehouse stocks out.

**Key challenge:** Balancing cost vs. lead time vs. reliability. The optimal supplier (SUP-003) costs more but arrives 1 day faster and is more reliable than the cheapest option.

**Grading:** 50% service level + 20% cost efficiency + 30% resilience (activation speed)

**Baseline score (gpt-4.1-mini):** ~0.61

---

### Task 2: Logistics Reroute (Medium) — 14 days, $120k budget

**Scenario:** Rotterdam port is closed by a dock workers' strike. Three shipments are blocked. A key European supplier is also at 50% capacity. Three warehouses have different stock levels and demand forecasts.

**Agent objective:** Reroute blocked shipments via alternative ports (Antwerp, Hamburg, Le Havre, or air freight) and reallocate existing stock between warehouses.

**Key challenges:**
- Multiple simultaneous disruptions
- Trade-off between fast (expensive) and slow (cheap) reroute options
- Stock reallocation requires budget too (transfer cost)

**Grading:** 40% service level + 30% cost efficiency + 30% resilience (shipments unblocked)

**Baseline score (gpt-4.1-mini):** ~0.47

---

### Task 3: Cascading Disruption (Hard) — 21 days, $200k budget

**Scenario:** Typhoon Hana takes out 2 suppliers and a major port simultaneously. Demand spikes +80% at critical warehouses. Budget is intentionally insufficient to satisfy all demand — triage is required.

**Hidden events (not visible at reset):**
- Day 7: Factory explosion at backup supplier (Continental Europe) — capacity drops to 20%
- Day 14: Currency shock raises all import costs by 25%

**Key challenges:**
- Must triage: critical warehouses (LA, Tokyo) vs. secondary (London, Dubai)
- Negotiating emergency contracts affects capacity and cost
- Hidden information requires adaptation mid-episode
- No strategy can satisfy all demand — prioritization is essential

**Grading:** 45% weighted service level (critical warehouses 70% weight) + 25% cost efficiency + 30% resilience

**Baseline score (gpt-4.1-mini):** ~0.28

---

## Action Space

```json
// Activate a supplier
{
  "action_type": "activate_supplier",
  "activate_supplier": {
    "supplier_id": "SUP-003",
    "order_quantity": 2000,
    "destination_warehouse": "WH-MAIN"
  },
  "reasoning": "Best lead time and reliability within budget"
}

// Reroute a blocked shipment
{
  "action_type": "reroute_shipment",
  "reroute_shipment": {
    "shipment_id": "SHP-ROT-001",
    "new_route": "via_antwerp",
    "expedite": false
  }
}

// Transfer stock between warehouses
{
  "action_type": "allocate_stock",
  "allocate_stock": {
    "source_warehouse": "WH-EU-EAST",
    "destination_warehouse": "WH-EU-NORTH",
    "quantity": 500
  }
}

// Negotiate supplier contract (Task 3)
{
  "action_type": "negotiate_contract",
  "negotiate_contract": {
    "supplier_id": "SUP-X4",
    "contract_type": "emergency",
    "max_price_per_unit": 50.0
  }
}

// Do nothing
{ "action_type": "wait" }
```

---

## Observation Space

```json
{
  "step": 3,
  "task_id": "supplier_triage",
  "episode_id": "abc12345",
  "suppliers": [
    {
      "supplier_id": "SUP-001",
      "name": "PrimeCo Manufacturing",
      "country": "China",
      "capacity_available": 0.0,
      "lead_time_days": 0,
      "cost_per_unit": 12.0,
      "reliability_score": 0.95,
      "disruption_active": true,
      "disruption_reason": "Factory fire — offline indefinitely"
    }
  ],
  "warehouses": [
    {
      "warehouse_id": "WH-MAIN",
      "location": "Chicago, US",
      "current_stock": 400,
      "capacity": 5000,
      "demand_forecast": 1600,
      "days_of_stock_remaining": 1.4
    }
  ],
  "active_shipments": [],
  "disruption_events": [...],
  "budget_remaining": 38000.0,
  "total_budget": 50000.0,
  "days_elapsed": 3,
  "max_days": 7,
  "stockout_count": 0,
  "service_level": 0.4286
}
```

---

## Reward Function

Every step returns a reward (0.0–1.0) with components:

| Component | Description | Weight |
|---|---|---|
| `service_level_component` | Fraction of cumulative demand met | 0.40–0.50 |
| `cost_efficiency_component` | Budget utilization efficiency | 0.20–0.30 |
| `resilience_component` | Speed and quality of disruption response | 0.25–0.30 |
| `penalty` | Subtracted for invalid actions and stockouts | — |

The reward is **non-sparse**: the agent receives signal every step, not just at episode end. This enables gradient-based learning throughout the trajectory.

---

## Setup & Usage

### Local Development

```bash
# Clone and install
git clone <your-repo-url>
cd supply-chain-env
pip install -r requirements.txt

# Start the server
python app.py
# → Server running at http://localhost:7860

# Test endpoints
curl http://localhost:7860/health
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "supplier_triage", "seed": 42}'

# Run inference
export HF_TOKEN=your_api_key
export API_BASE_URL=https://api.openai.com/v1
export MODEL_NAME=gpt-4.1-mini
export ENV_BASE_URL=http://localhost:7860
python inference.py
```

### Docker

```bash
docker build -t supply-chain-env .
docker run -p 7860:7860 supply-chain-env
```

### Hugging Face Spaces

1. Create a new Space with Docker SDK
2. Push this repo to the Space
3. Set `PORT=7860` in Space settings
4. The Space will auto-build and deploy

---

## Pre-Submission Validation

```bash
# Local validation
bash validate-submission.sh http://localhost:7860

# With HF Space URL
bash validate-submission.sh https://your-username-supply-chain-env.hf.space
```

---

## Baseline Scores

Run with `gpt-4.1-mini` (temperature=0.2, seed=42):

| Task | Score | Notes |
|---|---|---|
| supplier_triage | 0.61 | Correctly activates SUP-003 but 1 day late |
| logistics_reroute | 0.47 | Reroutes 2/3 shipments, misses stock reallocation |
| cascade_disruption | 0.28 | Responds to initial shock but misses hidden events |
| **Overall** | **0.45** | |

---

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `API_BASE_URL` | No | `https://api.openai.com/v1` | LLM endpoint |
| `MODEL_NAME` | No | `gpt-4.1-mini` | Model identifier |
| `HF_TOKEN` | **Yes** | — | Hugging Face / API key |
| `ENV_BASE_URL` | No | `http://localhost:7860` | Environment server URL |
| `PORT` | No | `7860` | Server port |

---

## OpenEnv API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Health check |
| `/tasks` | GET | List all tasks with metadata |
| `/reset` | POST | Initialize episode |
| `/step` | POST | Execute action |
| `/state` | GET | Current observation |
| `/grade` | GET | Final episode score |

---

## File Structure

```
supply-chain-env/
├── app.py                          FastAPI HTTP server
├── env.py                          Core SupplyChainEnv (OpenEnv interface)
├── models.py                       Pydantic typed models (Observation, Action, Reward)
├── inference.py                    Mandatory inference script
├── openenv.yaml                    OpenEnv metadata spec
├── Dockerfile                      Container definition
├── requirements.txt
├── validate-submission.sh          Pre-submission validator
├── README.md
├── tasks/
│   ├── __init__.py
│   ├── task1_supplier_triage.py    Easy (7 days)
│   ├── task2_logistics_reroute.py  Medium (14 days)
│   └── task3_cascade_disruption.py Hard (21 days)
└── graders/
    └── __init__.py                 Grading functions
```
