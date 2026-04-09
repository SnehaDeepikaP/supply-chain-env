def grade(trajectory: dict) -> float:
    """
    Medium Task Grader: Logistics Reroute

    Focus:
    - Shipment recovery
    - Stock balancing
    - Moderate cost control
    """

    final_reward = trajectory.get("final_reward", 0.0)
    steps = trajectory.get("steps", [])

    if not steps:
        return 0.0

    last_obs = steps[-1]["observation"]

    service = last_obs.get("service_level", 0.0)
    stockouts = last_obs.get("stockout_count", 0)

    budget_remaining = last_obs.get("budget_remaining", 0.0)
    total_budget = last_obs.get("total_budget", 1.0)
    cost_efficiency = budget_remaining / total_budget

    # Penalize stockouts more in medium task
    stockout_penalty = min(0.3, stockouts * 0.05)

    score = (
        0.5 * service +
        0.3 * cost_efficiency -
        stockout_penalty
    )

    # Blend with env reward
    score = 0.6 * final_reward + 0.4 * score

    return round(max(0.0, min(1.0, score)), 4)