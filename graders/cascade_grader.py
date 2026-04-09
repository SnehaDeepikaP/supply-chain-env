def grade(trajectory: dict) -> float:
    """
    Hard Task Grader: Cascade Disruption

    Focus:
    - Critical warehouse survival
    - Handling hidden events
    - Smart contracts + resilience
    """

    final_reward = trajectory.get("final_reward", 0.0)
    steps = trajectory.get("steps", [])

    if not steps:
        return 0.0

    last_step = steps[-1]
    last_obs = last_step["observation"]
    last_info = last_step.get("info", {})

    service = last_obs.get("service_level", 0.0)
    stockouts = last_obs.get("stockout_count", 0)

    budget_remaining = last_obs.get("budget_remaining", 0.0)
    total_budget = last_obs.get("total_budget", 1.0)
    cost_efficiency = budget_remaining / total_budget

    # Hidden event handling (VERY IMPORTANT SIGNAL)
    hidden_events_seen = len(last_info.get("hidden_events_revealed", []))

    # Contracts = resilience signal
    contracts = last_info.get("contracts", {})
    contract_count = len(contracts)

    # --- scoring ---
    resilience_score = min(1.0, (contract_count + hidden_events_seen) / 4)

    stockout_penalty = min(0.4, stockouts * 0.04)

    score = (
        0.4 * service +
        0.2 * cost_efficiency +
        0.3 * resilience_score -
        stockout_penalty
    )

    # Strong reliance on env reward for hard task
    score = 0.75 * final_reward + 0.25 * score

    return round(max(0.0, min(1.0, score)), 4)