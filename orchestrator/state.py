"""
Core state schema for the multi-agent orchestrator.

This IS the task ledger from our conceptual deep dive (#5 State Management),
now formalized as a LangGraph state schema. Every concept we discussed hangs
off this file:

- #5 State Management  -> TaskLedger is the shared, centralized part of our
                           hybrid state model.
- #6 Context Management -> `result` fields are structured (dict), not raw
                           conversation dumps, so scoping context to a
                           sub-agent later is just "pull this task's result."
- #9 Error Handling      -> `status` + `error` fields carry failure state
                           through the SAME channel as success state.
- #10 Observability      -> `history` is an append-only log of every status
                           transition -> this is your trace, for free.
- #11 Concurrency        -> `depends_on` is the DAG. Tasks with no unmet
                           dependency are the same "superstep level" that
                           LangGraph's Send API will later fan out over.

IMPORTANT SCOPE NOTE: this state is designed for SINGLE-SESSION execution.
We are deliberately NOT using LangGraph's checkpointer/persistence layer.
State lives only in memory for the duration of one graph.invoke() call.
"""

from __future__ import annotations

import operator
from datetime import datetime, timezone
from enum import Enum
from typing import Annotated, Any, Optional, TypedDict

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Task-level types
# ---------------------------------------------------------------------------

class TaskStatus(str, Enum):
    """
    Explicit state machine for a task's lifecycle (#1 loop convergence,
    #9 error handling). Every transition MUST go through one of these —
    no implicit states, no magic strings scattered through the codebase.
    """
    PENDING = "pending"      # created, dependencies not yet satisfied
    READY = "ready"          # dependencies satisfied, not yet dispatched
    RUNNING = "running"      # dispatched to an agent, awaiting result
    DONE = "done"            # completed successfully
    FAILED = "failed"        # failed after exhausting retry policy
    BLOCKED = "blocked"      # a dependency failed; this task cannot run


class FailurePolicy(BaseModel):
    """
    Per-task-type failure handling (#9). Attached to each task at creation
    time rather than hardcoded per-agent-type in the control loop — this is
    what lets us scale to N agent types without new branching logic every
    time we add one.
    """
    max_retries: int = 2
    on_exhausted: str = "blocked"  # "blocked" | "skip" | "abort_plan"


class TaskEvent(BaseModel):
    """A single entry in a task's audit trail. This is #10 observability,
    literally implemented as data rather than as a logging side-channel —
    the trace IS the state."""
    timestamp: str
    from_status: Optional[TaskStatus]
    to_status: TaskStatus
    detail: str = ""


class Task(BaseModel):
    id: str
    description: str
    assigned_agent: str  # e.g. "web_search", "synthesis", "report_writer"

    # The DAG (#11 concurrency, #6 context scoping input contract)
    depends_on: list[str] = Field(default_factory=list)

    # Lifecycle (#1, #9)
    status: TaskStatus = TaskStatus.PENDING
    attempts: int = 0
    failure_policy: FailurePolicy = Field(default_factory=FailurePolicy)
    error: Optional[str] = None

    # Structured, not narrative (#6 context management, #8 synthesis input).
    # e.g. {"claims": [{"text": ..., "source": ..., "confidence": ...}]}
    result: Optional[dict[str, Any]] = None

    # Audit trail (#10)
    history: list[TaskEvent] = Field(default_factory=list)

    def transition(self, to_status: TaskStatus, detail: str = "") -> None:
        """The ONLY sanctioned way to change a task's status. Guarantees
        every transition is logged -> no silent state changes."""
        self.history.append(
            TaskEvent(
                timestamp=datetime.now(timezone.utc).isoformat(),
                from_status=self.status,
                to_status=to_status,
                detail=detail,
            )
        )
        self.status = to_status


# ---------------------------------------------------------------------------
# Reducers
# ---------------------------------------------------------------------------
#
# In LangGraph, every state field needs a reducer: a function that says
# "when two nodes both produce an update for this field in the same
# superstep, how do they merge?" This is the SAME question we raised in
# #5/#11 (what happens when two agents write concurrently) -- LangGraph
# just forces you to answer it explicitly per-field instead of letting it
# be implicit and buggy.
#
# For `tasks`, we don't want "last write wins" (that would let one parallel
# branch silently clobber another's task updates). We want a merge-by-id.

def merge_tasks(
    existing: dict[str, Task], updates: dict[str, Task]
) -> dict[str, Task]:
    """Merge task updates into the existing ledger, keyed by task id.
    This is what makes concurrent sub-agents SAFE to update the ledger
    simultaneously (#11): each agent only ever produces updates for its
    OWN task id, so merges never collide on a field within one task."""
    merged = dict(existing)
    merged.update(updates)
    return merged


# ---------------------------------------------------------------------------
# Top-level graph state
# ---------------------------------------------------------------------------

class OrchestratorState(TypedDict):
    """
    The state object LangGraph threads through every node in the graph.
    This is what gets passed to `StateGraph(OrchestratorState)`.

    Two fields deliberately have reducers (Annotated[...]) because they can
    be updated by concurrent/parallel nodes. Everything else uses the
    default "last write wins" because only one place (the planner) ever
    writes to it.
    """

    # The original ask -- per #6, every sub-agent needs at least a
    # compressed pointer back to this, to avoid scope drift.
    user_request: str

    # THE task ledger. Keyed by task id for O(1) lookup + safe merging.
    tasks: Annotated[dict[str, Task], merge_tasks]

    # Global step counter -- our hard cap on loop iterations (#1: nothing
    # else guarantees the planner loop terminates).
    step_count: Annotated[int, operator.add]
    max_steps: int

    # Final output, set once by the report-writer / synthesis step.
    final_report: Optional[str]

    # Terminal flag the planner sets explicitly -- convergence must be a
    # deliberate decision, never inferred.
    is_complete: bool