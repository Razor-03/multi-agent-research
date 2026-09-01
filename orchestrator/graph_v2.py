"""
Graph v3 -- real decomposition + real dispatch routing.

Two things become real in this version, together (they're coupled: an
LLM-generated DAG is meaningless without routing that respects it):

1. PLANNER DECOMPOSITION (#7 tool calling / structured output)
   The planner's first visit now calls an LLM with a Pydantic schema and
   asks it to break `user_request` into a task DAG -- instead of the
   hardcoded 4-task stub from v2. This is "structured extraction" per our
   #6 discussion: we don't want prose back, we want typed Task-shaped
   objects directly, via `.with_structured_output()`.

2. DISPATCH-BY-AGENT-TYPE (new -- the piece we skipped until now)
   Previously EVERY ready task went to the same stub_worker regardless of
   `assigned_agent`. Now `route_after_planner` inspects the READY tasks'
   `assigned_agent` field and routes to the correspondingly-named node.
   This is what makes `assigned_agent` a meaningful field instead of
   decoration -- #2 (planner/supervisor pattern): the planner's real job
   is deciding WHO does WHAT, and this is that decision made executable.

   IMPORTANT SIMPLIFICATION, stated explicitly: LangGraph's conditional
   edges route to exactly ONE next node per planner visit. With multiple
   READY tasks of DIFFERENT agent types in the same pass, we currently
   pick ONE (first by id) and dispatch just that one -- the rest stay
   READY and get picked up on a later pass. This is CORRECT but not yet
   CONCURRENT: it's still one-task-per-graph-step, just now routed to the
   right specialist instead of a generic stub. Real concurrent fan-out
   across agent types is the `Send` API -- deliberately deferred so we
   don't conflate "correct routing" with "parallel execution" in one
   step.

Only `web_search` is a REAL agent in this version (real tool call via
Tavily-style web search... here we use a simple mock search function
since network/API key availability varies -- swap `mock_web_search` for
a real search tool trivially, the AGENT NODE STRUCTURE doesn't change).
`synthesis` and `report_writer` remain stubs -- same reasoning as before,
prove one piece for real before generalizing.

NOTE ON EXECUTION: this file requires ANTHROPIC_API_KEY to be set to run
the decomposition step for real. The `if __name__ == "__main__"` block
has TWO entry points: `run_live()` (needs the API key) and
`test_routing_only()` (no API key needed -- injects a fake decomposition
to prove the DISPATCH logic works in isolation). Run the latter to
verify plumbing without burning API calls.
"""

from __future__ import annotations

import os
from typing import Any, Literal

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field

from state import OrchestratorState, Task, TaskStatus


TERMINAL_STATUSES = {TaskStatus.DONE, TaskStatus.FAILED, TaskStatus.BLOCKED}

# The set of agent types our planner is allowed to assign work to. This is
# the thing that lets us scale to N agents later -- add a name here, add
# a node with that name, done. No branching logic to touch.
KNOWN_AGENTS = {"web_search", "synthesis", "report_writer"}


# ---------------------------------------------------------------------------
# Structured output schema for decomposition (#7: tool calling / structured
# extraction -- we want typed objects back, not prose to parse)
# ---------------------------------------------------------------------------

class TaskSpec(BaseModel):
    """One task as the LLM should propose it. Deliberately NOT the same
    class as our internal `Task` (which has runtime-only fields like
    `status`, `history`) -- the LLM only proposes the plan shape, our code
    owns the lifecycle state. Keeping these separate is the same principle
    as #6's scoped-context idea: give the LLM exactly the surface it needs,
    nothing about internal bookkeeping it shouldn't be inventing."""
    id: str = Field(description="short unique id, e.g. 't1'")
    description: str = Field(description="what this task should accomplish")
    assigned_agent: Literal["web_search", "synthesis", "report_writer"]
    depends_on: list[str] = Field(
        default_factory=list,
        description="ids of tasks that must complete before this one can run",
    )


class DecompositionPlan(BaseModel):
    tasks: list[TaskSpec]


DECOMPOSITION_SYSTEM_PROMPT = """You are a task planner for a multi-agent \
research system. Break the user's request into a small DAG of tasks.

Available agent types:
- web_search: searches the web for a specific, narrow sub-question
- synthesis: combines multiple web_search results into structured findings
- report_writer: writes the final report from synthesis output

Rules:
- Use web_search for independent, parallelizable sub-questions (no \
  dependencies between them).
- Use exactly one synthesis task that depends on all web_search tasks.
- Use exactly one report_writer task that depends on the synthesis task.
- Keep it small: 2-4 web_search tasks is typical.
- Task ids must be unique short strings.
"""


def decompose_request(user_request: str) -> dict[str, Task]:
    """The real #7 call: LLM + structured output -> typed Task objects.
    This REPLACES the hardcoded seeding from v2."""
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    structured_llm = llm.with_structured_output(DecompositionPlan)

    plan: DecompositionPlan = DecompositionPlan.model_validate(
        structured_llm.invoke(
            [
                ("system", DECOMPOSITION_SYSTEM_PROMPT),
                ("human", user_request),
            ]
        )
    )

    print(f"Decomposition plan: {plan}")

    tasks: dict[str, Task] = {}
    for spec in plan.tasks:
        tasks[spec.id] = Task(
            id=spec.id,
            description=spec.description,
            assigned_agent=spec.assigned_agent,
            depends_on=spec.depends_on,
        )
    return tasks


# ---------------------------------------------------------------------------
# Readiness resolution -- UNCHANGED from v2 (already proven correct)
# ---------------------------------------------------------------------------

def resolve_readiness(tasks: dict[str, Task]) -> dict[str, Task]:
    updates: dict[str, Task] = {}
    for task_id, task in tasks.items():
        if task.status != TaskStatus.PENDING:
            continue
        dep_statuses = [tasks[dep_id].status for dep_id in task.depends_on]
        if any(s in (TaskStatus.FAILED, TaskStatus.BLOCKED) for s in dep_statuses):
            t = task.model_copy(deep=True)
            t.transition(TaskStatus.BLOCKED,
                         f"upstream failed/blocked: {task.depends_on}")
            updates[task_id] = t
            continue
        if all(s == TaskStatus.DONE for s in dep_statuses):
            t = task.model_copy(deep=True)
            t.transition(TaskStatus.READY, "all dependencies satisfied")
            updates[task_id] = t
    return updates


# ---------------------------------------------------------------------------
# Planner node
# ---------------------------------------------------------------------------

def planner_node(state: OrchestratorState) -> dict:
    tasks = state["tasks"]

    if not tasks:
        seeded = decompose_request(state["user_request"])
        updates = resolve_readiness(seeded)
        merged = {**seeded, **updates}
        return {"tasks": merged, "step_count": 1}

    updates = resolve_readiness(tasks)
    latest = {**tasks, **updates}
    all_terminal = all(t.status in TERMINAL_STATUSES for t in latest.values())

    if all_terminal:
        done_count = sum(1 for t in latest.values()
                         if t.status == TaskStatus.DONE)
        return {
            "tasks": updates,
            "is_complete": True,
            "final_report": f"stub report: {done_count}/{len(latest)} tasks completed.",
            "step_count": 1,
        }

    return {"tasks": updates, "step_count": 1}


# ---------------------------------------------------------------------------
# Real agent: web_search (one real tool call, #7)
# ---------------------------------------------------------------------------

def mock_web_search(query: str) -> list[dict]:
    """Stand-in for a real search tool (e.g. Tavily). Swap this one
    function for a real API call later -- nothing else in the agent node
    needs to change, because the agent node's CONTRACT (query in,
    structured results out) doesn't depend on the implementation."""
    return [
        {"title": f"Result for: {query}",
            "snippet": "...", "url": "https://example.com"}
    ]


def web_search_node(state: OrchestratorState) -> dict:
    tasks = state["tasks"]
    ready = [t for t in tasks.values()
             if t.status == TaskStatus.READY and t.assigned_agent == "web_search"]
    if not ready:
        return {"step_count": 1}

    t = ready[0].model_copy(deep=True)
    t.transition(TaskStatus.RUNNING, "web_search agent picked up task")
    try:
        results = mock_web_search(t.description)
        t.result = {
            "claims": [
                {"text": r["snippet"], "source": r["url"], "confidence": 0.7}
                for r in results
            ]
        }
        t.transition(TaskStatus.DONE,
                     f"search returned {len(results)} results")
    except Exception as e:  # #9: real failures become ledger state, not crashes
        t.attempts += 1
        t.error = str(e)
        if t.attempts >= t.failure_policy.max_retries:
            t.transition(TaskStatus.FAILED, f"exhausted retries: {e}")
        else:
            t.transition(TaskStatus.READY,
                         f"attempt {t.attempts} failed, will retry: {e}")

    return {"tasks": {t.id: t}, "step_count": 1}


# ---------------------------------------------------------------------------
# Stub agents: synthesis, report_writer (same pattern as v2's stub_worker,
# just scoped to their own assigned_agent value now)
# ---------------------------------------------------------------------------

def make_stub_agent(agent_name: str):
    def _node(state: OrchestratorState) -> dict:
        tasks = state["tasks"]
        ready = [t for t in tasks.values()
                 if t.status == TaskStatus.READY and t.assigned_agent == agent_name]
        if not ready:
            return {"step_count": 1}
        t = ready[0].model_copy(deep=True)
        t.transition(TaskStatus.RUNNING, f"{agent_name} (stub) picked up task")
        t.result = {"note": f"stub output from {agent_name}"}
        t.transition(TaskStatus.DONE, f"{agent_name} (stub) finished")
        return {"tasks": {t.id: t}, "step_count": 1}
    return _node


synthesis_node = make_stub_agent("synthesis")
report_writer_node = make_stub_agent("report_writer")


# ---------------------------------------------------------------------------
# Dispatch routing -- THE new piece. Inspects READY tasks' assigned_agent
# and routes to the matching node name.
# ---------------------------------------------------------------------------

def route_after_planner(state: OrchestratorState) -> str:
    if state["step_count"] >= state["max_steps"]:
        return "end"
    if state.get("is_complete"):
        return "end"

    ready = [t for t in state["tasks"].values() if t.status ==
             TaskStatus.READY]
    if not ready:
        # nothing ready right now (e.g. waiting on in-flight deps) -- loop
        # the planner again rather than dead-ending. Guarded by max_steps.
        return "planner"

    # Route to whichever agent type the first READY task needs. See the
    # module docstring for the honest caveat: with mixed READY agent
    # types in one pass, only one gets dispatched this step.
    target_agent = ready[0].assigned_agent
    if target_agent not in KNOWN_AGENTS:
        raise ValueError(
            f"planner assigned unknown agent type: {target_agent}")
    return target_agent


def build_graph():
    graph = StateGraph(OrchestratorState)
    graph.add_node("planner", planner_node)
    graph.add_node("web_search", web_search_node)
    graph.add_node("synthesis", synthesis_node)
    graph.add_node("report_writer", report_writer_node)

    graph.set_entry_point("planner")
    graph.add_conditional_edges(
        "planner",
        route_after_planner,
        {
            "planner": "planner",
            "web_search": "web_search",
            "synthesis": "synthesis",
            "report_writer": "report_writer",
            "end": END,
        },
    )
    for agent in ("web_search", "synthesis", "report_writer"):
        graph.add_edge(agent, "planner")

    return graph.compile()


# ---------------------------------------------------------------------------
# Two run modes
# ---------------------------------------------------------------------------

def run_live():
    """Requires ANTHROPIC_API_KEY. Exercises the REAL decomposition call."""
    app = build_graph()
    initial_state: OrchestratorState = {
        "user_request": "What are the tradeoffs between LangGraph and building a multi-agent system from scratch?",
        "tasks": {},
        "step_count": 0,
        "max_steps": 25,
        "final_report": None,
        "is_complete": False,
    }
    result = app.invoke(initial_state)
    _print_result(result)


def test_routing_only():
    """No API key needed. Injects a fake decomposition (bypassing
    decompose_request entirely) to prove the DISPATCH/ROUTING logic works
    in isolation -- this is what we can actually verify in this sandbox."""
    # NOTE: patching the module-level global directly (not via a fresh
    # `import graph as graph_module`) -- since this file runs as __main__,
    # re-importing it as "graph" creates a SECOND, separate module object,
    # and patching that copy silently does nothing to the one executing.
    global decompose_request

    fake_tasks = {
        "s1": Task(id="s1", description="search A", assigned_agent="web_search"),
        "s2": Task(id="s2", description="search B", assigned_agent="web_search"),
        "syn": Task(id="syn", description="synthesize", assigned_agent="synthesis",
                    depends_on=["s1", "s2"]),
        "rep": Task(id="rep", description="write report", assigned_agent="report_writer",
                    depends_on=["syn"]),
    }

    # monkeypatch decompose_request so planner_node's first branch uses our
    # fake plan instead of calling a real LLM
    original = decompose_request
    def decompose_request(user_request): return fake_tasks

    try:
        app = build_graph()
        initial_state: OrchestratorState = {
            "user_request": "fake request (bypassed)",
            "tasks": {},
            "step_count": 0,
            "max_steps": 25,
            "final_report": None,
            "is_complete": False,
        }
        result = app.invoke(initial_state)
        _print_result(result)
    finally:
        decompose_request = original


def _print_result(result):
    print("=== FINAL STATE ===")
    print("is_complete:", result["is_complete"])
    print("step_count:", result["step_count"])
    print("final_report:", result["final_report"])
    print()
    print("=== FULL LEDGER TRACE ===")
    for task_id, task in sorted(result["tasks"].items()):
        print(
            f"\n[{task_id}] agent={task.assigned_agent} depends_on={task.depends_on}")
        for event in task.history:
            print(
                f"    {event.from_status} -> {event.to_status} :: {event.detail}")


if __name__ == "__main__":
    if os.environ.get("GOOGLE_API_KEY"):
        run_live()
    else:
        print("(no GOOGLE_API_KEY found -- running routing-only test instead)\n")
        test_routing_only()
