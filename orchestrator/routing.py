"""
Dispatch routing -- returns a LIST of Send objects, one per READY task,
instead of a single node name. This is the actual #11 payoff: every READY
task (regardless of how many, regardless of agent type mix) gets
dispatched in the SAME superstep, running concurrently. LangGraph
converges all their results before the planner runs again.
"""

from __future__ import annotations

from langgraph.graph import END
from langgraph.types import Send

from state import OrchestratorState, TaskStatus
from config import KNOWN_AGENTS


def route_after_planner(state: OrchestratorState):
    if state["step_count"] >= state["max_steps"]:
        return END
    if state.get("is_complete"):
        return END
    if state.get("needs_human_review"):
        return "human_review"

    ready = [t for t in state["tasks"].values() if t.status == TaskStatus.READY]
    if not ready:
        # nothing ready right now (e.g. waiting on in-flight deps) -- loop
        # the planner again rather than dead-ending. Guarded by max_steps.
        return "planner"

    sends = []
    for t in ready:
        if t.assigned_agent not in KNOWN_AGENTS:
            raise ValueError(f"planner assigned unknown agent type: {t.assigned_agent}")
        # Scoped dependency results (#6): only THIS task's own upstream
        # results, looked up from the full ledger here in the router (which
        # has global visibility) and handed down -- the worker itself never
        # sees the full ledger, only what it depends on.
        dep_results = {
            dep_id: state["tasks"][dep_id].result
            for dep_id in t.depends_on
            if state["tasks"][dep_id].result is not None
        }
        sends.append(
            Send(t.assigned_agent, {
                "task": t,
                "user_request": state["user_request"],
                "dependency_results": dep_results,
            })
        )
    return sends
