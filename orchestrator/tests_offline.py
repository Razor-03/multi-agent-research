"""
Offline test harness. Every test here runs without ANTHROPIC_API_KEY or
TAVILY_API_KEY (some exercise real-but-failing LLM calls deliberately, to
prove our #9 error handling catches them).

Run with: python3 tests_offline.py [--flag]
  (no flag)      routing/dispatch logic, no LLM calls made at all
  --concurrency  proves Send-based fan-out is genuinely concurrent + the
                 max_concurrency cap is respected
  --synthesis    exercises synthesis_node/report_writer_node/judge_node
                 for real (needs API keys to see real output; without
                 keys, proves their error handling)
  --judge        isolates judge_node specifically + the PARTIAL_SUCCESS
                 policy when the judge itself fails
  --hitl         proves the full interrupt/resume mechanics against the
                 REAL graph (not the toy one in hitl_toy_test.py)

IMPORTANT PATCHING NOTE: `decompose_request` lives in planner.py, and
planner_node looks it up from ITS OWN module globals at call time. So
tests that want to bypass the real LLM decomposition must patch the
module ATTRIBUTE (`planner.decompose_request = ...`), not a locally
imported name -- patching a name imported via
`from planner import decompose_request` would only rebind this module's
local variable, leaving planner_node looking at the original. This bit
us once already (see graph.py's git history / earlier conversation) in
a different but related form -- worth remembering as a general rule for
monkeypatching across modules.
"""

from __future__ import annotations

import os

from langgraph.graph import StateGraph
from langgraph.types import Command

import planner  # imported as a module, NOT `from planner import decompose_request`
                 # -- see the patching note above for why this matters
from state import OrchestratorState, Task, TaskStatus
from config import DEFAULT_MAX_CONCURRENCY
from agent_common import WorkerInput
from agent_synthesis import synthesis_node
from agent_report_writer import report_writer_node
from routing import route_after_planner
from graph import build_graph
from utils import print_result


def test_trivial_request_skips_pipeline():
    """Fix 1: proves that a trivial request (forced via monkeypatch to
    return needs_research=false) short-circuits to a direct answer WITHOUT
    dispatching web_search, synthesis, or judge at all. Checks this two
    ways: (a) the final ledger has zero tasks in it -- nothing was ever
    seeded to dispatch, and (b) step_count is exactly 1 -- only the
    planner itself ran, no agent node was ever invoked."""
    original = planner.decompose_request
    planner.decompose_request = lambda user_request: planner.DecompositionResult(
        tasks={}, direct_answer="Hi! How can I help you today?"
    )

    try:
        app = build_graph()
        result = app.invoke(
            {
                "user_request": "hi",
                "tasks": {},
                "step_count": 0,
                "max_steps": 25,
                "final_report": None,
                "is_complete": False,
                "needs_human_review": False,
                "human_reviewed": False,
            },
            config={"configurable": {"thread_id": "test-trivial"}},
        )
    finally:
        planner.decompose_request = original

    print_result(result)
    print()
    assert result["is_complete"] is True
    assert result["final_report"] == "Hi! How can I help you today?"
    assert result["tasks"] == {}, f"expected zero tasks, got {list(result['tasks'].keys())}"
    assert result["step_count"] == 1, (
        f"expected exactly 1 step (planner only, no dispatch), got {result['step_count']}"
    )
    print("PASS: trivial request answered directly -- zero tasks created, "
          "zero agent nodes dispatched, single planner step.")


def test_routing_only():
    """No API key needed. Injects a fake decomposition (bypassing
    decompose_request entirely) to prove the DISPATCH/ROUTING logic works
    in isolation."""
    fake_tasks = {
        "s1": Task(id="s1", description="search A", assigned_agent="web_search"),
        "s2": Task(id="s2", description="search B", assigned_agent="web_search"),
        "syn": Task(id="syn", description="synthesize", assigned_agent="synthesis",
                    depends_on=["s1", "s2"]),
        "rep": Task(id="rep", description="write report", assigned_agent="report_writer",
                    depends_on=["syn"]),
    }

    original = planner.decompose_request
    planner.decompose_request = lambda user_request: planner.DecompositionResult(tasks=fake_tasks)

    try:
        app = build_graph()
        initial_state: OrchestratorState = {
            "user_request": "fake request (bypassed)",
            "tasks": {},
            "step_count": 0,
            "max_steps": 25,
            "final_report": None,
            "is_complete": False,
            "needs_human_review": False,
            "human_reviewed": False,
        }
        result = app.invoke(
            initial_state,
            config={"configurable": {"thread_id": "test-routing"}},
        )
        print_result(result)
    finally:
        planner.decompose_request = original


def test_concurrency_proof():
    """No API key needed. Proves tasks dispatched via Send actually run
    CONCURRENTLY (not just correctly) by having each worker sleep and
    recording wall-clock start/end times. 4 tasks, max_concurrency=2 --
    if the cap is respected, this should take ~2s (two batches of 2), not
    ~1s (uncapped) or ~4s (sequential)."""
    import time
    import threading

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
        t.result = {"claims": [{"text": "slow result", "source": "mock", "confidence": 0.7}]}
        t.transition(TaskStatus.DONE, "slow search finished")
        return {"tasks": {t.id: t}, "step_count": 1}

    fake_tasks = {
        "s1": Task(id="s1", description="search A", assigned_agent="web_search"),
        "s2": Task(id="s2", description="search B", assigned_agent="web_search"),
        "s3": Task(id="s3", description="search C", assigned_agent="web_search"),
        "s4": Task(id="s4", description="search D", assigned_agent="web_search"),
    }
    concurrency_cap = 2

    original = planner.decompose_request
    planner.decompose_request = lambda user_request: planner.DecompositionResult(tasks=fake_tasks)

    graph = StateGraph(OrchestratorState)
    graph.add_node("planner", planner.planner_node)
    graph.add_node("web_search", slow_web_search_node)
    graph.add_node("synthesis", synthesis_node)
    graph.add_node("report_writer", report_writer_node)
    graph.set_entry_point("planner")
    graph.add_conditional_edges("planner", route_after_planner)
    for agent in ("web_search", "synthesis", "report_writer"):
        graph.add_edge(agent, "planner")
    app = graph.compile()  # no checkpointer needed -- no HITL nodes in this mini-graph

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
        planner.decompose_request = original

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
    print_result(result)
    print()
    outcome, explanation = planner.assess_plan_outcome(result["tasks"])
    print(f"=== PLAN OUTCOME ===\n{outcome}: {explanation}")


def test_synthesis_and_report_isolated():
    """Bypasses web_search entirely -- seeds s1/s2 as already-DONE with
    fake claims (one deliberately conflicting pair, to test #8's
    conflict-surfacing rule) -- so synthesis_node and report_writer_node
    actually run, regardless of API key availability for search. If
    ANTHROPIC_API_KEY is missing, this proves their OWN error handling
    works. If it's present, this is the real test of synthesis quality --
    does it actually merge corroborating claims and surface the conflict
    rather than picking a side silently."""
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
    planner._inject_eval_task(fake_tasks)  # so this test also exercises judge_node

    original = planner.decompose_request
    planner.decompose_request = lambda user_request: planner.DecompositionResult(tasks=fake_tasks)

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
                "needs_human_review": False,
                "human_reviewed": False,
            },
            config={
                "configurable": {"thread_id": "test-synthesis"},
                "max_concurrency": DEFAULT_MAX_CONCURRENCY,
            },
        )
    finally:
        planner.decompose_request = original

    print_result(result)


def test_judge_isolated():
    """Bypasses web_search AND synthesis/report_writer -- seeds everything
    through report_writer as already DONE with a fake report, so ONLY
    judge_node actually runs. Proves (a) judge_node's own error handling
    in isolation, without earlier-stage failures masking it, and (b) the
    terminal-task exclusion logic in assess_plan_outcome -- a judge
    failure here should yield PARTIAL_SUCCESS (we have a real report),
    NOT FAILED."""
    fake_tasks = {
        "s1": Task(id="s1", description="search A", assigned_agent="web_search",
                   status=TaskStatus.DONE,
                   result={"claims": [{"text": "fact A", "source": "http://a", "confidence": 0.9}]}),
        "syn": Task(id="syn", description="synthesize", assigned_agent="synthesis",
                    depends_on=["s1"], status=TaskStatus.DONE,
                    result={"findings": [{"finding": "fact A restated", "sources": ["http://a"],
                                          "corroborated": False}], "conflicts": []}),
        "rep": Task(id="rep", description="write report", assigned_agent="report_writer",
                    depends_on=["syn"], status=TaskStatus.DONE,
                    result={"report": "This is a fake but complete report about fact A."}),
    }
    planner._inject_eval_task(fake_tasks)

    original = planner.decompose_request
    planner.decompose_request = lambda user_request: planner.DecompositionResult(tasks=fake_tasks)

    try:
        app = build_graph()
        result = app.invoke(
            {
                "user_request": "fake request (bypassed)",
                "tasks": {},
                "step_count": 0,
                "max_steps": 25,
                "final_report": None,
                "is_complete": False,
                "needs_human_review": False,
                "human_reviewed": False,
            },
            config={
                "configurable": {"thread_id": "test-judge"},
                "max_concurrency": DEFAULT_MAX_CONCURRENCY,
            },
        )
    finally:
        planner.decompose_request = original

    print_result(result)

    outcome, explanation = planner.assess_plan_outcome(result["tasks"])
    print(f"\n=== PLAN OUTCOME: {outcome} -- {explanation} ===")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        assert outcome == planner.PlanOutcome.PARTIAL_SUCCESS, (
            f"expected PARTIAL_SUCCESS when judge fails but report exists, got {outcome}"
        )
        print("PASS: judge failure correctly downgraded to PARTIAL_SUCCESS, not FAILED")


def test_hitl_flow():
    """Fully offline, no API keys needed. Seeds a judge task as already
    DONE with a forced 'fail' verdict, then scripts through the resume
    cycle programmatically (no input() prompt) to prove the actual
    pause/resume mechanics work end-to-end against our REAL graph, not
    just the toy one in hitl_toy_test.py.

    Tests two paths: 'accept' (finalize despite the fail verdict) and
    'retry_report' (reset report_writer + judge, loop back) -- run as two
    separate invocations against two separate thread_ids."""

    def make_fake_tasks():
        return {
            "s1": Task(id="s1", description="search A", assigned_agent="web_search",
                       status=TaskStatus.DONE,
                       result={"claims": [{"text": "fact A", "source": "http://a", "confidence": 0.9}]}),
            "syn": Task(id="syn", description="synthesize", assigned_agent="synthesis",
                        depends_on=["s1"], status=TaskStatus.DONE,
                        result={"findings": [{"finding": "fact A restated", "sources": ["http://a"],
                                              "corroborated": False}], "conflicts": []}),
            "rep": Task(id="rep", description="write report", assigned_agent="report_writer",
                        depends_on=["syn"], status=TaskStatus.DONE,
                        result={"report": "This report is actually pretty bad."}),
            "eval_final": Task(id="eval_final", description="evaluate", assigned_agent="judge",
                                depends_on=["rep", "syn"], status=TaskStatus.DONE,
                                result={"verdict": "fail", "answers_request": False,
                                        "grounded_in_findings": True,
                                        "issues": ["does not actually answer the request"],
                                        "reasoning": "forced fail for testing"}),
        }

    original = planner.decompose_request
    app = build_graph()

    # --- Path 1: human says "accept" ---
    print("=== PATH 1: human ACCEPTS the failed report ===\n")
    planner.decompose_request = lambda user_request: planner.DecompositionResult(tasks=make_fake_tasks())
    config1 = {"configurable": {"thread_id": "hitl-test-accept"}}
    initial_state: OrchestratorState = {
        "user_request": "test request", "tasks": {}, "step_count": 0, "max_steps": 25,
        "final_report": None, "is_complete": False,
        "needs_human_review": False, "human_reviewed": False,
    }
    result = app.invoke(initial_state, config=config1)
    assert "__interrupt__" in result, "expected the graph to pause for review"
    print("Paused for review, as expected. Payload:")
    print(" ", result["__interrupt__"][0].value)

    result = app.invoke(Command(resume={"action": "accept"}), config=config1)
    print("\nAfter resume(accept):")
    print("  is_complete:", result["is_complete"])
    print("  final_report:", result["final_report"])
    assert result["is_complete"] is True
    assert "This report is actually pretty bad" in result["final_report"]
    print("\nPASS: accept -> finalized with the original (flagged) report\n")

    # --- Path 2: human says "retry_report" ---
    print("=" * 60)
    print("=== PATH 2: human requests a RETRY of report_writer ===\n")
    planner.decompose_request = lambda user_request: planner.DecompositionResult(tasks=make_fake_tasks())
    config2 = {"configurable": {"thread_id": "hitl-test-retry"}}
    initial_state2: OrchestratorState = {
        "user_request": "test request", "tasks": {}, "step_count": 0, "max_steps": 25,
        "final_report": None, "is_complete": False,
        "needs_human_review": False, "human_reviewed": False,
    }
    result = app.invoke(initial_state2, config=config2)
    assert "__interrupt__" in result

    result = app.invoke(Command(resume={"action": "retry_report"}), config=config2)
    print("After resume(retry_report):")
    print("  is_complete:", result["is_complete"])
    rep_task = result["tasks"]["rep"]
    eval_task = result["tasks"]["eval_final"]
    print(f"  rep task status: {rep_task.status}, attempts reset: {rep_task.attempts == 0}")
    print(f"  eval_final task status: {eval_task.status}")
    print("  rep history (should show the reset event):")
    for e in rep_task.history:
        print(f"    {e.from_status} -> {e.to_status} :: {e.detail}")

    reset_event = next(
        (e for e in rep_task.history if "reset by human review" in e.detail), None
    )
    assert reset_event is not None, "expected a logged reset event in rep's history"
    print("\nPASS: retry_report correctly reset report_writer+judge and "
          "looped back into the graph (reset event is logged in history)")

    planner.decompose_request = original


if __name__ == "__main__":
    import sys
    if "--concurrency" in sys.argv:
        test_concurrency_proof()
    elif "--synthesis" in sys.argv:
        test_synthesis_and_report_isolated()
    elif "--judge" in sys.argv:
        test_judge_isolated()
    elif "--hitl" in sys.argv:
        test_hitl_flow()
    elif "--trivial" in sys.argv:
        test_trivial_request_skips_pipeline()
    else:
        test_routing_only()