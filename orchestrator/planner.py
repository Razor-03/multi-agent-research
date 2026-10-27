"""
Planner module: everything that decides WHAT work exists and WHETHER the
plan is done. This is the "reason" half of the ReAct loop (#1) and the
supervisor/planner pattern (#2) made concrete.

Contents, in the order they execute during a run:
  1. PlanOutcome / assess_plan_outcome -- policy for what a mix of
     success/failure across tasks MEANS for the plan overall (#9/#11).
  2. TaskSpec / DecompositionPlan / decompose_request -- the real #7 LLM
     call that turns a user request into a task DAG.
  3. _inject_eval_task -- structurally guarantees evaluation happens
     (#10), independent of what the decomposition LLM proposes.
  4. resolve_readiness -- DAG traversal: PENDING -> READY/BLOCKED based
     on dependency status (#11's "what could run together" question).
  5. planner_node -- the LangGraph node itself: seeds the plan on first
     visit, resolves readiness on every visit, and decides convergence
     (#1) including the HITL trigger check (pause instead of finalizing
     when the judge says "fail" and no human has reviewed yet).
"""

from __future__ import annotations

from typing import Literal

from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from state import OrchestratorState, Task, TaskStatus
from config import TERMINAL_STATUSES


# ---------------------------------------------------------------------------
# Partial-failure policy (#9 / #11)
#
# The planner's "are we done" check needs more than "has every task
# reached a TERMINAL status" -- DONE, FAILED, and BLOCKED shouldn't all be
# treated as equally acceptable ways to be terminal. A plan where every
# web_search task FAILED should not silently produce a false "success."
# We need an explicit policy for what a mix of success/failure MEANS for
# the overall run, not just what it means for each task individually.
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
      - If a DELIVERABLE task (one nothing else depends on, EXCLUDING the
        judge -- see below) specifically FAILED or is BLOCKED -> FAILED.
        We can't produce a final answer without it, regardless of how
        many upstream searches succeeded.
      - Otherwise, if at least one task DONE -> PARTIAL_SUCCESS (we have
        SOMETHING to work with, even if degraded).
      - If nothing succeeded at all -> FAILED.

    IMPORTANT: the `judge` task is deliberately EXCLUDED from the
    "terminal = must succeed" check, even though it's the actual DAG leaf
    (nothing depends on the evaluator). A failed evaluation is a missing
    QUALITY SIGNAL, not a missing DELIVERABLE -- the report still exists
    and is still usable if the judge itself errors out. Without this
    exclusion, the judge would become a single point of failure for the
    entire plan, which contradicts its own purpose (a lightweight
    secondary check, not a gate). See agent_judge.py's docstring for the
    matching design note on the other side of this decision.
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
    #
    # Subtlety: a task counts as "deliverable-terminal" if it has NO
    # dependents, OR its only dependents are judge tasks. This correctly
    # identifies report_writer as terminal (its only dependent is the
    # judge) while NOT counting the judge itself as terminal. Naively
    # excluding judge-assigned tasks from the terminal set entirely would
    # be a worse bug: report_writer would stop being "terminal" too
    # (something -- the judge -- still depends on it), and a failed
    # report_writer would then silently fall through to PARTIAL_SUCCESS
    # as long as any web_search task succeeded. Caught this while writing
    # it, worth stating so the reasoning doesn't get silently lost.
    def _is_deliverable_terminal(task_id: str) -> bool:
        dependents = [t for t in tasks.values() if task_id in t.depends_on]
        non_judge_dependents = [t for t in dependents if t.assigned_agent != "judge"]
        return len(non_judge_dependents) == 0

    terminal_task_ids = {
        tid for tid, t in tasks.items()
        if t.assigned_agent != "judge" and _is_deliverable_terminal(tid)
    }
    terminal_failures = [t for t in failed_or_blocked if t.id in terminal_task_ids]

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
    """Triage + decomposition, done in ONE structured call rather than two.

    Fix for a real gap: without this, EVERY request -- including "hi" --
    triggered a full web_search -> synthesis -> report_writer -> judge
    DAG. That's wasted tool calls and wasted cost for anything that
    doesn't actually need research. `needs_research` lets the SAME LLM
    call that would have decomposed the request instead just answer it
    directly, when a full DAG would be overkill.
    """
    needs_research: bool = Field(
        description="true if this genuinely requires web research/multi-step "
                    "work; false if you can answer it directly and confidently "
                    "right now (greetings, simple factual/definitional "
                    "questions with stable answers, basic math, opinions, "
                    "casual conversation)."
    )
    direct_answer: str | None = Field(
        default=None,
        description="REQUIRED if needs_research is false: your direct answer "
                    "to the user's request. Leave null if needs_research is true.",
    )
    tasks: list[TaskSpec] = Field(
        default_factory=list,
        description="REQUIRED (non-empty) if needs_research is true, "
                    "following the DAG rules below. Leave empty if "
                    "needs_research is false.",
    )


DECOMPOSITION_SYSTEM_PROMPT = """You are the entry point for a multi-agent \
research system. For every request, first decide: does this genuinely \
need web research, or can you answer it directly right now?

Answer directly (needs_research=false) for: greetings and casual \
conversation, simple math, well-established facts/definitions that don't \
change over time, requests for your opinion, or anything you can answer \
confidently and completely from what you already know.

Use research (needs_research=true) for: anything involving current \
events, recent/changing information, specific real-world facts you're \
not certain of, or requests that genuinely need multiple independent \
pieces of information gathered and synthesized. When in doubt about \
whether something is time-sensitive or you might be wrong, prefer \
research over guessing.

If needs_research is true, break the request into a small DAG of tasks:

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


class DecompositionResult:
    """What decompose_request actually returns -- either a task DAG to
    execute, or a direct answer with no tasks at all. planner_node
    branches on `tasks` being empty to decide which path this run takes."""
    def __init__(self, tasks: dict[str, Task], direct_answer: str | None = None):
        self.tasks = tasks
        self.direct_answer = direct_answer


def decompose_request(user_request: str) -> DecompositionResult:
    """The real #7 call: LLM + structured output -> either a typed task
    DAG, or a direct answer with an empty task dict (Fix 1: trivial
    requests skip the whole pipeline -- no web_search, no synthesis, no
    judge call, no wasted tool use)."""
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

    if not plan.needs_research:
        return DecompositionResult(
            tasks={},
            direct_answer=plan.direct_answer or "(no answer provided)",
        )

    tasks: dict[str, Task] = {}
    for spec in plan.tasks:
        tasks[spec.id] = Task(
            id=spec.id,
            description=spec.description,
            assigned_agent=spec.assigned_agent,
            depends_on=spec.depends_on,
        )

    _inject_eval_task(tasks)
    return DecompositionResult(tasks=tasks)


def _inject_eval_task(tasks: dict[str, Task]) -> None:
    """#10 Evaluation, made STRUCTURALLY guaranteed rather than optional.

    Deliberately NOT letting the decomposition LLM propose this task
    itself -- if evaluation were just another item the LLM's plan *might*
    include, a bad or lazy decomposition could skip the one thing meant
    to catch bad output. So we inject it programmatically here, after
    decomposition, regardless of what the LLM proposed.

    The judge depends on BOTH report_writer and its own upstream
    synthesis task -- not just the report -- so it can check whether the
    report actually stayed grounded in what synthesis found, rather than
    judging the report in a vacuum where hallucination would be invisible.

    Mutates `tasks` in place; no-op if there's no report_writer task to
    evaluate (shouldn't happen given our decomposition prompt's rules,
    but we don't assume).
    """
    report_tasks = [t for t in tasks.values() if t.assigned_agent == "report_writer"]
    if not report_tasks:
        return

    report_task = report_tasks[0]
    eval_deps = [report_task.id] + list(report_task.depends_on)  # report + its synthesis dep(s)

    eval_id = "eval_final"
    tasks[eval_id] = Task(
        id=eval_id,
        description="Evaluate the final report for grounding and relevance",
        assigned_agent="judge",
        depends_on=eval_deps,
    )


# ---------------------------------------------------------------------------
# Readiness resolution (#11: this IS how we discover which tasks form the
# same "DAG level" / could run concurrently)
# ---------------------------------------------------------------------------

def resolve_readiness(tasks: dict[str, Task]) -> dict[str, Task]:
    """Pure function: given the current ledger, return the set of task
    UPDATES needed this pass (promotions to READY, propagations to
    BLOCKED). Doesn't mutate the input."""
    updates: dict[str, Task] = {}
    for task_id, task in tasks.items():
        if task.status != TaskStatus.PENDING:
            continue
        dep_statuses = [tasks[dep_id].status for dep_id in task.depends_on]
        if any(s in (TaskStatus.FAILED, TaskStatus.BLOCKED) for s in dep_statuses):
            t = task.model_copy(deep=True)
            t.transition(TaskStatus.BLOCKED, f"upstream failed/blocked: {task.depends_on}")
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
        decomposition = decompose_request(state["user_request"])

        if not decomposition.tasks:
            # Fix 1: trivial request, no research needed. Finalize
            # IMMEDIATELY in this same return -- is_complete must be set
            # here, not deferred, or the next planner visit would see an
            # empty `tasks` dict again and re-trigger decomposition in an
            # infinite loop. No web_search/synthesis/judge dispatch ever
            # happens for this path -- route_after_planner sees
            # is_complete=True and goes straight to END.
            return {
                "tasks": {},
                "is_complete": True,
                "final_report": decomposition.direct_answer,
                "step_count": 1,
            }

        seeded = decomposition.tasks
        updates = resolve_readiness(seeded)
        merged = {**seeded, **updates}
        return {"tasks": merged, "step_count": 1}

    updates = resolve_readiness(tasks)
    latest = {**tasks, **updates}
    all_terminal = all(t.status in TERMINAL_STATUSES for t in latest.values())

    if all_terminal:
        outcome, explanation = assess_plan_outcome(latest)

        # --- HITL trigger check ---
        # If the judge's verdict was "fail" and no human has reviewed yet,
        # pause for review INSTEAD OF finalizing. Once a human has
        # reviewed (human_reviewed=True), fall through to normal
        # finalization even if the verdict is still "fail" -- that's the
        # human's call to make, we don't ask twice.
        judge_tasks_check = [t for t in latest.values() if t.assigned_agent == "judge"]
        judge_verdict = None
        if judge_tasks_check and judge_tasks_check[0].status == TaskStatus.DONE:
            judge_verdict = (judge_tasks_check[0].result or {}).get("verdict")

        if judge_verdict == "fail" and not state.get("human_reviewed", False):
            return {
                "tasks": updates,
                "needs_human_review": True,
                "step_count": 1,
            }

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

            # #10: prepend the eval verdict as a visible banner, when the
            # judge ran successfully. If the judge itself failed, we say
            # so explicitly rather than silently omitting the banner --
            # "no quality signal available" is different from "verdict:
            # pass," and hiding that difference would be misleading.
            judge_tasks = [t for t in latest.values() if t.assigned_agent == "judge"]
            if judge_tasks:
                jt = judge_tasks[0]
                if jt.status == TaskStatus.DONE and jt.result:
                    verdict = jt.result.get("verdict", "unknown")
                    issues = jt.result.get("issues", [])
                    banner = f"[eval: {verdict.upper()}]"
                    if issues:
                        banner += " Issues noted: " + "; ".join(issues)
                    final_report = f"{banner}\n\n{final_report}"
                else:
                    final_report = (
                        f"[eval: NOT AVAILABLE -- judge failed to run]\n\n{final_report}"
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