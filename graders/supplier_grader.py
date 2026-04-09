def grade(trajectory: dict) -> float:
    """
    Easy Task Grader: Supplier Triage

    Focus:
    - Did agent recover supply quickly?
    - Did it avoid unnecessary cost?
    """

    final_reward = trajectory.get("final_reward", 0.0)
    steps = trajectory.get("steps", [])

    if not steps:
        return 0.0

    # Extract signals
    last_obs = steps[-1]["observation"]

    service = last_obs.get("service_level", 0.0)
    budget_remaining = last_obs.get("budget_remaining", 0.0)
    total_budget = last_obs.get("total_budget", 1.0)

    cost_efficiency = budget_remaining / total_budget

    # Simple weighted score
    score = (
        0.6 * service +
        0.4 * cost_efficiency
    )

    # Blend with environment reward
    score = 0.7 * final_reward + 0.3 * score

    return round(max(0.0, min(1.0, score)), 4)