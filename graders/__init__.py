"""
Graders are standalone wrappers around task.grade() for external evaluation.
Each grader takes a completed task instance and returns 0.0–1.0.
"""

def grade_task1(task) -> float:
    """Grade a completed Task1SupplierTriage episode."""
    return round(task.grade(), 4)


def grade_task2(task) -> float:
    """Grade a completed Task2LogisticsReroute episode."""
    return round(task.grade(), 4)


def grade_task3(task) -> float:
    """Grade a completed Task3CascadeDisruption episode."""
    return round(task.grade(), 4)


def run_all_graders(task1, task2, task3) -> dict:
    """Run all three graders and return combined summary."""
    s1 = grade_task1(task1)
    s2 = grade_task2(task2)
    s3 = grade_task3(task3)
    overall = round((s1 + s2 + s3) / 3, 4)
    return {
        "task1_supplier_triage": s1,
        "task2_logistics_reroute": s2,
        "task3_cascade_disruption": s3,
        "overall_score": overall,
    }
