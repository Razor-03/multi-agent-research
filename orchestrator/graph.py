"""
Graph skeleton -- stub version.

Goal of this file: prove the LangGraph WIRING is correct -- the loop, the
conditional routing, the termination condition -- BEFORE we put any real
LLM intelligence inside a single node. Every node here is a dumb stub.

Concept mapping (so nothing here is unexplained magic):

- StateGraph(OrchestratorState)  -> the graph is literally parameterized by
                                     our task ledger from state.py (#5).
- add_node("planner", ...)       -> #1/#2: the planner is just a python
                                     function; the "agent loop" is the
                                     graph's own execution loop, we don't
                                     hand-roll a while-loop ourselves.
- add_conditional_edges(...)     -> #1 convergence: this IS the "while not
                                     done" check, made explicit as a graph
                                     edge instead of buried in a loop body.
- The planner -> worker -> planner cycle -> #1 ReAct shape: reason
                                     (planner) -> act (worker) -> observe
                                     (back to planner) -> repeat.
- step_count / max_steps check   -> #1: the hard iteration cap. Without
                                     this, a buggy conditional edge could
                                     loop forever.

We are NOT using a checkpointer here -- `graph.compile()` with no
checkpointer means state lives only for the duration of one `.invoke()`
call, which matches our explicit single-session scope decision.
"""

from __future__ import annotations

from langgraph.graph import StateGraph, END

from state import OrchestratorState, Task, TaskStatus


# ---------------------------------------------------------------------------
# Stub nodes -- deliberately dumb. No LLM calls yet.
# ---------------------------------------------------------------------------

def planner_node(state: OrchestratorState) -> dict:
    """
    STUB. Real version (next step) will actually look at task statuses and
    decide what's READY to dispatch. Right now it just:
      - on the very first call, seeds one fake task
      - once that task is DONE, marks the plan complete

    This is intentionally trivial -- we're proving the EDGE from planner
    back to itself (via the worker) actually cycles correctly.
    """
    tasks = state["tasks"]

    if not tasks:
        # First entry into the loop: seed a single stub task.
        t = Task(id="t1", description="stub task", assigned_agent="stub_worker")
        t.transition(TaskStatus.READY, "seeded by planner (stub)")
        return {
            "tasks": {"t1": t},
            "step_count": 1,
        }

    # If our one task is done, we're finished.
    if tasks["t1"].status == TaskStatus.DONE:
        return {
            "is_complete": True,
            "final_report": "stub report: task t1 completed successfully.",
            "step_count": 1,
        }

    # Otherwise nothing new to do this pass -- just count the step.
    return {"step_count": 1}


def stub_worker_node(state: OrchestratorState) -> dict:
    """
    STUB for a sub-agent. Real version later will actually dispatch to a
    LangGraph subgraph / tool call per #7. Right now it just flips the one
    task from READY -> RUNNING -> DONE across two visits, so we can watch
    the ledger's `history` accumulate transitions (#10) across loop
    iterations.
    """
    tasks = dict(state["tasks"])
    t = tasks["t1"]

    if t.status == TaskStatus.READY:
        t.transition(TaskStatus.RUNNING, "stub_worker picked up task")
    elif t.status == TaskStatus.RUNNING:
        t.result = {"claims": [{"text": "stub result", "source": "stub", "confidence": 1.0}]}
        t.transition(TaskStatus.DONE, "stub_worker finished")

    return {"tasks": {"t1": t}, "step_count": 1}


# ---------------------------------------------------------------------------
# Routing logic -- this is #1's convergence check, made explicit
# ---------------------------------------------------------------------------

def route_after_planner(state: OrchestratorState) -> str:
    """Conditional edge out of the planner. Three exits:
    - hard step cap hit         -> END (safety valve, #1)
    - planner declared complete -> END (deliberate termination, #1)
    - otherwise                 -> hand off to the worker
    """
    if state["step_count"] >= state["max_steps"]:
        return "end"
    if state.get("is_complete"):
        return "end"
    return "dispatch"


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------

def build_graph():
    graph = StateGraph(OrchestratorState)

    graph.add_node("planner", planner_node)
    graph.add_node("stub_worker", stub_worker_node)

    graph.set_entry_point("planner")

    graph.add_conditional_edges(
        "planner",
        route_after_planner,
        {"dispatch": "stub_worker", "end": END},
    )
    # worker always hands control back to the planner -- this is the
    # "observe" step of ReAct feeding back into "reason" (#1).
    graph.add_edge("stub_worker", "planner")

    return graph.compile()  # NOTE: no checkpointer -- single-session only


if __name__ == "__main__":
    app = build_graph()

    initial_state: OrchestratorState = {
        "user_request": "stub request",
        "tasks": {},
        "step_count": 0,
        "max_steps": 10,
        "final_report": None,
        "is_complete": False,
    }

    result = app.invoke(initial_state)

    print("=== FINAL STATE ===")
    print("is_complete:", result["is_complete"])
    print("step_count:", result["step_count"])
    print("final_report:", result["final_report"])
    print()
    print("=== TASK t1 HISTORY (this is our free observability trace) ===")
    for event in result["tasks"]["t1"].history:
        print(f"  {event.from_status} -> {event.to_status} :: {event.detail}")