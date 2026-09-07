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

NOTE ON EXECUTION: this file requires GOOGLE_API_KEY to be set to run
the decomposition step for real. The `if __name__ == "__main__"` block
has TWO entry points: `run_live()` (needs the API key) and
`test_routing_only()` (no API key needed -- injects a fake decomposition
to prove the DISPATCH logic works in isolation). Run the latter to
verify plumbing without burning API calls.
"""

from __future__ import annotations
from langchain_tavily import TavilySearch

import os
from typing import Literal, TypedDict

from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from langgraph.types import Send
from pydantic import BaseModel, Field

from state import OrchestratorState, Task, TaskStatus


TERMINAL_STATUSES = {TaskStatus.DONE, TaskStatus.FAILED, TaskStatus.BLOCKED}

# The set of agent types our planner is allowed to assign work to. This is
# the thing that lets us scale to N agents later -- add a name here, add
# a node with that name, done. No branching logic to touch.
KNOWN_AGENTS = {"web_search", "synthesis", "report_writer"}


# ---------------------------------------------------------------------------
# Concurrency cap (#11 gap #1: rate limiting)
#
# Send fans out to however many tasks are READY, with no upper bound of its
# own. Against a real API with real rate limits, an unbounded fan-out is a
# genuine way to get throttled or burn budget fast -- not hypothetical.
# LangGraph exposes `max_concurrency` on the RunnableConfig passed to
# invoke(), which caps how many node executions run simultaneously across
# the WHOLE graph step, regardless of how many Sends were emitted. This is
# a hard infrastructure limit, separate from and layered UNDER whatever the
# DAG says is theoretically parallelizable -- exactly the "two separate
# constraints" point from our #11 discussion: the DAG says what COULD run
# together, max_concurrency says how much of that we ACTUALLY let run at
# once.
# ---------------------------------------------------------------------------

DEFAULT_MAX_CONCURRENCY = 4


# ---------------------------------------------------------------------------
# Partial-failure policy (#11 gap #2 / #9)
#
# Before: the planner's "are we done" check only asked "has every task
# reached a TERMINAL status" -- DONE, FAILED, and BLOCKED were all treated
# as equally acceptable ways to be terminal. That's wrong: a plan where
# every web_search task FAILED should not silently produce a "stub report:
# 0/5 tasks completed" and call it success. We need an explicit policy for
# what a mix of success/failure MEANS for the overall run, not just what it
# means for each task individually.
#
# This mirrors the #9 distinction between "sub-agent handles its own
# retries" and "planner handles plan-level consequences" -- individual task
# failure was already handled (retry policy on Task). This is the layer
# above: what does failure of one or more tasks mean for the OVERALL run.
# ---------------------------------------------------------------------------

class PlanOutcome:
    SUCCESS = "success"                  # all tasks DONE
    PARTIAL_SUCCESS = "partial_success"  # some DONE, some FAILED/BLOCKED, but
    # enough succeeded to still produce
    # a (degraded) result
    FAILED = "failed"                    # too much failed to produce anything
    # meaningful


def assess_plan_outcome(tasks: dict[str, Task]) -> tuple[str, str]:
    """Returns (outcome, explanation). Policy, stated explicitly rather
    than left implicit:
      - If ALL tasks are DONE -> SUCCESS.
      - If a TERMINAL task (one nothing else depends on -- i.e. an end
        product like report_writer) specifically FAILED or is BLOCKED ->
        FAILED. We can't produce a final answer without it, regardless of
        how many upstream searches succeeded.
      - Otherwise, if at least one task DONE -> PARTIAL_SUCCESS (we have
        SOMETHING to work with, even if degraded).
      - If nothing succeeded at all -> FAILED.
    """
    done = [t for t in tasks.values() if t.status == TaskStatus.DONE]
    failed_or_blocked = [
        t for t in tasks.values() if t.status in (TaskStatus.FAILED, TaskStatus.BLOCKED)
    ]

    if not failed_or_blocked:
        return PlanOutcome.SUCCESS, f"all {len(tasks)} tasks completed"

    # Tasks with no dependents are "terminal" in the DAG sense -- nothing
    # downstream needs them. If a TERMINAL task specifically failed, no
    # amount of upstream success can produce a final result.
    depended_on = {dep for t in tasks.values() for dep in t.depends_on}
    terminal_task_ids = {tid for tid in tasks if tid not in depended_on}
    terminal_failures = [
        t for t in failed_or_blocked if t.id in terminal_task_ids]

    if terminal_failures:
        names = [t.id for t in terminal_failures]
        return (
            PlanOutcome.FAILED,
            f"terminal task(s) {names} did not complete -- no usable output",
        )

    if done:
        return (
            PlanOutcome.PARTIAL_SUCCESS,
            f"{len(done)}/{len(tasks)} tasks completed; "
            f"{len(failed_or_blocked)} failed/blocked but did not block the final output",
        )

    return PlanOutcome.FAILED, "no tasks completed successfully"


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
        outcome, explanation = assess_plan_outcome(latest)

        # Pull the REAL report text out of the report_writer task's result
        # if we have one, rather than always emitting a generic status
        # string. Falls back to the status explanation when there's no
        # actual report to show (e.g. FAILED before report_writer ran).
        report_tasks = [
            t for t in latest.values()
            if t.assigned_agent == "report_writer" and t.status == TaskStatus.DONE
        ]
        if report_tasks and report_tasks[0].result:
            report_body = report_tasks[0].result.get("report", "")
            final_report = (
                f"[{outcome}] {explanation}\n\n{report_body}"
                if outcome == PlanOutcome.PARTIAL_SUCCESS
                else report_body
            )
        else:
            final_report = f"[{outcome}] {explanation}"

        return {
            "tasks": updates,
            "is_complete": True,
            "final_report": final_report,
            "step_count": 1,
        }

    return {"tasks": updates, "step_count": 1}


# ---------------------------------------------------------------------------
# WorkerInput -- the SCOPED context contract (#6), now literal.
#
# Previously every worker node re-scanned the ENTIRE `tasks` ledger to find
# its own work. That was a workaround for not having per-invocation input.
# With Send, each worker invocation gets EXACTLY what it needs: one task,
# plus a compressed pointer to the original ask (per #6's "scoped context
# is not context amnesia" rule -- the worker should still know what the
# overall goal was, even scoped down to one task).
#
# This is a DIFFERENT state shape than OrchestratorState on purpose -- Send
# explicitly supports invoking a node with custom state that differs from
# the main graph state (see the Send docstring). The return value still
# merges back into OrchestratorState via the normal `tasks` reducer.
# ---------------------------------------------------------------------------

class WorkerInput(TypedDict):
    task: Task
    user_request: str  # compressed top-level goal, per #6's anti-drift rule
    dependency_results: dict[str, dict]  # {task_id: task.result} for each
    # id in task.depends_on -- the
    # scoped input for synthesis/
    # report_writer, populated by the
    # router at dispatch time (see
    # route_after_planner below)


# ---------------------------------------------------------------------------
# Real agent: web_search (real tool call via Tavily, #7)
#
# This REPLACES mock_web_search. Note what did NOT need to change:
# web_search_node's structure, its error handling, its Task/ledger
# plumbing -- none of it. Only the function that does the actual network
# call changed. That's the payoff of the "query in, structured results
# out" contract we set up back when this was still a mock: the AGENT
# doesn't care how the search happens, only that it gets a query and
# returns something shaped like search results.
#
# Requires TAVILY_API_KEY to be set. If it's missing, TavilySearch's own
# client raises at call time -- which lands in our existing try/except in
# web_search_node and becomes a ledger FAILED/retry state (#9), not a
# crash. We don't need to add new error handling for "no API key" as a
# special case; the existing failure path already covers it correctly.
# ---------------------------------------------------------------------------


_tavily_tool = None


def _get_tavily_tool() -> TavilySearch:
    """Lazy singleton, NOT instantiated at import time.

    Bug we just avoided: creating TavilySearch() at module load means a
    missing TAVILY_API_KEY crashes the whole program on import -- before
    our #9 error handling in web_search_node ever gets a chance to catch
    it. Lazy init means the failure happens INSIDE the try/except where
    it belongs, and becomes a ledger FAILED/retry state instead of a
    startup crash. Small detail, but it's exactly the kind of thing that
    silently defeats a retry policy if you don't catch it."""
    global _tavily_tool
    if _tavily_tool is None:
        _tavily_tool = TavilySearch(max_results=3)
    return _tavily_tool


def real_web_search(query: str) -> list[dict]:
    """Real search via Tavily. Same contract as the old mock_web_search:
    takes a query string, returns a list of result dicts with at least
    'snippet'-equivalent text and a source URL -- normalized here so
    web_search_node doesn't need to know Tavily's specific response shape."""
    tool = _get_tavily_tool()
    raw = tool.invoke({"query": query})
    results = raw.get("results", [])
    return [
        {
            "title": r.get("title", ""),
            "snippet": r.get("content", ""),
            "url": r.get("url", ""),
        }
        for r in results
    ]


def web_search_node(worker_input: WorkerInput) -> dict:
    t = worker_input["task"].model_copy(deep=True)
    t.transition(TaskStatus.RUNNING, "web_search agent picked up task")
    try:
        results = real_web_search(t.description)
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
# Stub agents: synthesis, report_writer -- same WorkerInput contract.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Real agent: synthesis (#8 result aggregation, #7 tool calling / structured
# output)
#
# Input: dependency_results, scoped to exactly this task's upstream tasks
# (#6). Each web_search task's result is {"claims": [{"text", "source",
# "confidence"}]} -- structured, not prose, per our #6/#8 discussion: the
# map step (extracting claims) already happened at search time, so
# synthesis's job is purely the REDUCE step -- combine structured claims
# into a coherent, deduplicated set of findings.
#
# Explicitly instructed on conflict/redundancy handling (#8's sharp edge):
# corroborating claims get merged with combined attribution, conflicting
# claims get surfaced as a disagreement rather than silently resolved.
# ---------------------------------------------------------------------------

class SynthesizedFinding(BaseModel):
    finding: str = Field(
        description="a single synthesized point, in Claude's own words")
    sources: list[str] = Field(
        description="source URLs supporting this finding")
    corroborated: bool = Field(
        description="true if 2+ independent sources support this finding"
    )


class SynthesisOutput(BaseModel):
    findings: list[SynthesizedFinding]
    conflicts: list[str] = Field(
        default_factory=list,
        description="descriptions of any claims that directly contradicted each other",
    )


SYNTHESIS_SYSTEM_PROMPT = """You are a synthesis agent. You receive raw \
claims extracted from multiple web searches and must combine them into a \
clean set of findings.

Rules:
- If multiple sources support the same point, merge them into ONE finding \
  with all supporting source URLs attached, and mark corroborated=true.
- If sources DIRECTLY CONTRADICT each other, do NOT silently pick one. \
  Add an entry to `conflicts` describing the disagreement, and still \
  include both claims as separate findings with corroborated=false.
- Do not invent claims that aren't present in the input.
- Keep each finding concise -- one clear point per finding.
"""


def synthesis_node(worker_input: WorkerInput) -> dict:
    t = worker_input["task"].model_copy(deep=True)
    t.transition(TaskStatus.RUNNING, "synthesis agent picked up task")

    try:
        # Flatten all upstream claims into one list the LLM can reason over.
        all_claims = []
        for dep_id, dep_result in worker_input["dependency_results"].items():
            for claim in dep_result.get("claims", []):
                all_claims.append({**claim, "from_task": dep_id})

        if not all_claims:
            # No upstream data at all (e.g. every search failed but this
            # task still got dispatched somehow) -- fail loudly rather
            # than fabricate a synthesis from nothing.
            raise ValueError("no upstream claims available to synthesize")

        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
        structured_llm = llm.with_structured_output(SynthesisOutput)
        output: SynthesisOutput = SynthesisOutput.model_validate(
            structured_llm.invoke([
                ("system", SYNTHESIS_SYSTEM_PROMPT),
                ("human",
                 f"Original request: {worker_input['user_request']}\n\n"
                 f"Raw claims:\n{all_claims}"),
            ]))

        t.result = {
            "findings": [f.model_dump() for f in output.findings],
            "conflicts": output.conflicts,
        }
        t.transition(
            TaskStatus.DONE,
            f"synthesized {len(output.findings)} findings "
            f"({len(output.conflicts)} conflicts noted)",
        )
    except Exception as e:  # #9: same pattern as web_search_node
        t.attempts += 1
        t.error = str(e)
        if t.attempts >= t.failure_policy.max_retries:
            t.transition(TaskStatus.FAILED, f"exhausted retries: {e}")
        else:
            t.transition(TaskStatus.READY,
                         f"attempt {t.attempts} failed, will retry: {e}")

    return {"tasks": {t.id: t}, "step_count": 1}


# ---------------------------------------------------------------------------
# Real agent: report_writer (#8, final output generation)
#
# Input: dependency_results from the synthesis task -- findings + any
# noted conflicts. Writes actual prose, surfacing conflicts explicitly
# rather than hiding them (same #8 principle carried through to the final
# artifact the user sees).
# ---------------------------------------------------------------------------

REPORT_SYSTEM_PROMPT = """You are a report-writing agent. You receive \
synthesized findings (and any noted conflicts between sources) and must \
write a clear, well-organized report answering the original request.

Rules:
- Structure the report with short sections/paragraphs, not just a list.
- If any conflicts were noted, explicitly mention the disagreement in the \
  report rather than silently picking a side.
- Cite sources inline where useful (e.g. "(source: ...)").
- Do not introduce claims that weren't in the findings.
- Give response in Markdown (.md) format.
"""


def report_writer_node(worker_input: WorkerInput) -> dict:
    t = worker_input["task"].model_copy(deep=True)
    t.transition(TaskStatus.RUNNING, "report_writer agent picked up task")

    try:
        synthesis_results = list(worker_input["dependency_results"].values())
        if not synthesis_results:
            raise ValueError("no synthesis output available to write from")

        # Only ever depends on exactly one synthesis task per our decomposition
        # prompt's rules -- take the first (and expected-only) result.
        synth = synthesis_results[0]

        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
        response = llm.invoke([
            ("system", REPORT_SYSTEM_PROMPT),
            ("human",
             f"Original request: {worker_input['user_request']}\n\n"
             f"Findings: {synth.get('findings', [])}\n\n"
             f"Conflicts noted: {synth.get('conflicts', [])}"),
        ])

        report_text = response.content
        t.result = {"report": report_text}
        t.transition(TaskStatus.DONE, "report written")
    except Exception as e:
        t.attempts += 1
        t.error = str(e)
        if t.attempts >= t.failure_policy.max_retries:
            t.transition(TaskStatus.FAILED, f"exhausted retries: {e}")
        else:
            t.transition(TaskStatus.READY,
                         f"attempt {t.attempts} failed, will retry: {e}")

    return {"tasks": {t.id: t}, "step_count": 1}


# ---------------------------------------------------------------------------
# Dispatch routing -- NOW returns a LIST of Send objects, one per READY
# task, instead of a single node name. This is the actual #11 payoff: every
# READY task (regardless of how many, regardless of agent type mix) gets
# dispatched in the SAME superstep, running concurrently. LangGraph
# converges all their results before the planner runs again.
# ---------------------------------------------------------------------------

def route_after_planner(state: OrchestratorState):
    if state["step_count"] >= state["max_steps"]:
        return END
    if state.get("is_complete"):
        return END

    ready = [t for t in state["tasks"].values() if t.status ==
             TaskStatus.READY]
    if not ready:
        # nothing ready right now (e.g. waiting on in-flight deps) -- loop
        # the planner again rather than dead-ending. Guarded by max_steps.
        return "planner"

    sends = []
    for t in ready:
        if t.assigned_agent not in KNOWN_AGENTS:
            raise ValueError(
                f"planner assigned unknown agent type: {t.assigned_agent}")
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


def build_graph():
    graph = StateGraph(OrchestratorState)
    graph.add_node("planner", planner_node)
    graph.add_node("web_search", web_search_node)
    graph.add_node("synthesis", synthesis_node)
    graph.add_node("report_writer", report_writer_node)

    graph.set_entry_point("planner")
    # No explicit mapping dict needed here -- route_after_planner returns
    # either a plain node name (str), END, or a list of Send objects, and
    # LangGraph resolves Send targets dynamically at runtime.
    graph.add_conditional_edges("planner", route_after_planner)

    for agent in ("web_search", "synthesis", "report_writer"):
        graph.add_edge(agent, "planner")

    return graph.compile()


# ---------------------------------------------------------------------------
# Two run modes
# ---------------------------------------------------------------------------

def run_live(max_concurrency: int = DEFAULT_MAX_CONCURRENCY):
    """Requires GOOGLE_API_KEY. Exercises the REAL decomposition call.
    `max_concurrency` caps simultaneous node executions per graph step --
    this is the actual rate-limit guard, not just a suggestion."""
    query = input("Enter a query: ")
    app = build_graph()
    initial_state: OrchestratorState = {
        "user_request": query,
        "tasks": {},
        "step_count": 0,
        "max_steps": 25,
        "final_report": None,
        "is_complete": False,
    }
    result = app.invoke(initial_state, config={
                        "max_concurrency": max_concurrency})
    _print_result(result)


def test_concurrency_proof():
    """No API key needed. Proves tasks dispatched via Send actually run
    CONCURRENTLY (not just correctly) by having each worker sleep and
    recording wall-clock start/end times. If truly concurrent, 3 tasks
    each sleeping 1s should take ~1s total, not ~3s."""
    import time
    import threading

    global decompose_request

    timing_log = []
    lock = threading.Lock()

    def slow_web_search_node(worker_input: WorkerInput) -> dict:
        t = worker_input["task"].model_copy(deep=True)
        start = time.monotonic()
        t.transition(TaskStatus.RUNNING, "slow web_search picked up task")
        time.sleep(1.0)  # simulate a real network-bound search call
        end = time.monotonic()
        with lock:
            timing_log.append((t.id, start, end))
        t.result = {"claims": [
            {"text": "slow result", "source": "mock", "confidence": 0.7}]}
        t.transition(TaskStatus.DONE, "slow search finished")
        return {"tasks": {t.id: t}, "step_count": 1}

    # 4 tasks now, and we cap max_concurrency at 2 -- if the cap is
    # respected, this should take ~2s (two batches of 2), NOT ~1s
    # (uncapped, all 4 at once) and NOT ~4s (fully sequential).
    fake_tasks = {
        "s1": Task(id="s1", description="search A", assigned_agent="web_search"),
        "s2": Task(id="s2", description="search B", assigned_agent="web_search"),
        "s3": Task(id="s3", description="search C", assigned_agent="web_search"),
        "s4": Task(id="s4", description="search D", assigned_agent="web_search"),
    }
    concurrency_cap = 2

    original_decompose = decompose_request
    def decompose_request(user_request): return fake_tasks

    graph = StateGraph(OrchestratorState)
    graph.add_node("planner", planner_node)
    graph.add_node("web_search", slow_web_search_node)
    graph.add_node("synthesis", synthesis_node)
    graph.add_node("report_writer", report_writer_node)
    graph.set_entry_point("planner")
    graph.add_conditional_edges("planner", route_after_planner)
    for agent in ("web_search", "synthesis", "report_writer"):
        graph.add_edge(agent, "planner")
    app = graph.compile()

    try:
        wall_start = time.monotonic()
        result = app.invoke(
            {
                "user_request": "fake (bypassed)",
                "tasks": {},
                "step_count": 0,
                "max_steps": 25,
                "final_report": None,
                "is_complete": False,
            },
            config={"max_concurrency": concurrency_cap},
        )
        wall_elapsed = time.monotonic() - wall_start
    finally:
        decompose_request = original_decompose

    print("=== CONCURRENCY CAP PROOF ===")
    print(f"4 tasks x 1.0s sleep each, max_concurrency={concurrency_cap}.")
    print(f"Total wall time: {wall_elapsed:.2f}s\n")
    for task_id, start, end in sorted(timing_log):
        print(f"  {task_id}: started at {start - wall_start:.2f}s, "
              f"finished at {end - wall_start:.2f}s")
    if 1.8 <= wall_elapsed < 3.5:
        print(f"\nPASS: cap respected (~2s for 2 batches of 2, "
              f"not ~1s uncapped or ~4s sequential)")
    else:
        print(f"\nUNEXPECTED timing -- investigate (got {wall_elapsed:.2f}s)")
    print()
    _print_result(result)
    print()
    outcome, explanation = assess_plan_outcome(result["tasks"])
    print(f"=== PLAN OUTCOME ===\n{outcome}: {explanation}")


def test_synthesis_and_report_isolated():
    """Bypasses web_search entirely -- seeds s1/s2 as already-DONE with
    fake claims (one deliberately conflicting pair, to test #8's
    conflict-surfacing rule) -- so synthesis_node and report_writer_node
    actually run, regardless of API key availability for search. If
    GOOGLE_API_KEY is missing, this proves their OWN error handling
    works (same pattern as web_search_node's). If it's present, this is
    the real test of synthesis quality -- does it actually merge
    corroborating claims and surface the conflict rather than picking a
    side silently."""
    global decompose_request

    fake_tasks = {
        "s1": Task(
            id="s1", description="search A", assigned_agent="web_search",
            status=TaskStatus.DONE,
            result={"claims": [
                {"text": "LangGraph uses a Pregel-style superstep execution model",
                 "source": "https://example.com/a", "confidence": 0.9},
                {"text": "LangGraph requires a checkpointer for all use cases",
                 "source": "https://example.com/a", "confidence": 0.6},
            ]},
        ),
        "s2": Task(
            id="s2", description="search B", assigned_agent="web_search",
            status=TaskStatus.DONE,
            result={"claims": [
                {"text": "LangGraph's execution model is based on Pregel/BSP",
                 "source": "https://example.com/b", "confidence": 0.85},
                {"text": "Checkpointers in LangGraph are optional and only needed for persistence/HITL",
                 "source": "https://example.com/b", "confidence": 0.8},
            ]},
        ),
        "syn": Task(id="syn", description="synthesize", assigned_agent="synthesis",
                    depends_on=["s1", "s2"]),
        "rep": Task(id="rep", description="write report", assigned_agent="report_writer",
                    depends_on=["syn"]),
    }

    original_decompose = decompose_request
    def decompose_request(user_request): return fake_tasks

    try:
        app = build_graph()
        result = app.invoke(
            {
                "user_request": "How does LangGraph handle execution and persistence?",
                "tasks": {},
                "step_count": 0,
                "max_steps": 25,
                "final_report": None,
                "is_complete": False,
            },
            config={"max_concurrency": DEFAULT_MAX_CONCURRENCY},
        )
    finally:
        decompose_request = original_decompose

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
    import sys
    if "--concurrency" in sys.argv:
        test_concurrency_proof()
    elif "--synthesis" in sys.argv:
        test_synthesis_and_report_isolated()
    elif os.environ.get("GOOGLE_API_KEY"):
        run_live()
    else:
        print("(no GOOGLE_API_KEY found -- running routing-only test instead)\n")
        test_routing_only()
