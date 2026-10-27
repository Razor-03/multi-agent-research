"""
Human-in-the-loop: review + resume.

Trigger: the judge's verdict was "fail" AND no human has reviewed yet
(that check lives in planner.py). This is the sharp trigger condition we
deliberately waited for before building HITL at all -- before we had a
real quality signal, "add HITL" would have meant guessing at when to
pause.

REQUIRES a checkpointer (see graph.py's build_graph) -- interrupt() has
nothing to pause/resume FROM without one. This is also why HITL was kept
out of scope for so long: it's not a small addition, it's a different
execution shape (pause-and-resume vs. one-shot invoke).

CRITICAL DESIGN CONSTRAINT, proven empirically in hitl_toy_test.py: a
node containing interrupt() RE-EXECUTES FROM THE TOP when resumed. So
human_review_node does nothing but READ state (cheap, no side effects) to
build the review payload, call interrupt(), and then act on the result.
All actual mutation happens strictly after the interrupt() call returns.
"""

from __future__ import annotations

from langgraph.types import interrupt

from state import OrchestratorState, Task, TaskStatus


def reset_tasks(tasks: dict[str, Task], task_ids: list[str], reason: str) -> dict[str, Task]:
    """Reset the given tasks back to PENDING so the planner's normal
    readiness resolution picks them up again next pass. Uses `transition()`
    (not a raw field assignment) so the reset itself is a LOGGED event in
    the task's history -- per #10, we don't want a human-triggered retry
    to be an invisible state change."""
    updates = {}
    for tid in task_ids:
        t = tasks[tid].model_copy(deep=True)
        t.result = None
        t.attempts = 0
        t.error = None
        t.transition(TaskStatus.PENDING, reason)
        updates[tid] = t
    return updates


def human_review_node(state: OrchestratorState) -> dict:
    tasks = state["tasks"]

    # --- everything above interrupt() is READ-ONLY. No mutation here. ---
    report_task = next(
        (t for t in tasks.values() if t.assigned_agent == "report_writer"), None
    )
    judge_task = next(
        (t for t in tasks.values() if t.assigned_agent == "judge"), None
    )
    payload = {
        "reason": "the judge flagged this report as FAIL",
        "report": report_task.result.get("report", "") if report_task and report_task.result else None,
        "verdict": judge_task.result if judge_task and judge_task.result else None,
        "options": ["accept", "retry_report", "retry_synthesis", "abort"],
    }

    decision = interrupt(payload)
    # --- everything below this line is the ONLY place mutation happens ---

    action = decision.get("action", "accept") if isinstance(decision, dict) else str(decision)

    synthesis_task = next(
        (t for t in tasks.values() if t.assigned_agent == "synthesis"), None
    )

    if action == "accept":
        return {
            "human_reviewed": True,
            "needs_human_review": False,
            "step_count": 1,
        }

    if action == "retry_report" and report_task and judge_task:
        updates = reset_tasks(
            tasks, [report_task.id, judge_task.id],
            "reset by human review decision: retry_report",
        )
        return {
            "tasks": updates,
            "needs_human_review": False,
            "human_reviewed": False,  # allow review to trigger again if the retry also fails
            "step_count": 1,
        }

    if action == "retry_synthesis" and synthesis_task and report_task and judge_task:
        updates = reset_tasks(
            tasks, [synthesis_task.id, report_task.id, judge_task.id],
            "reset by human review decision: retry_synthesis",
        )
        return {
            "tasks": updates,
            "needs_human_review": False,
            "human_reviewed": False,
            "step_count": 1,
        }

    # action == "abort" (or anything unrecognized -- fail safe, don't loop)
    return {
        "is_complete": True,
        "needs_human_review": False,
        "final_report": "[aborted by human reviewer after a failed evaluation]",
        "step_count": 1,
    }
