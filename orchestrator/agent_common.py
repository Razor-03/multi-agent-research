"""
WorkerInput -- the SCOPED context contract (#6), now literal, shared by
every agent node (web_search, synthesis, report_writer, judge).

Before this existed, every worker node re-scanned the ENTIRE `tasks`
ledger to find its own work. That was a workaround for not having
per-invocation input. With LangGraph's `Send` (see routing.py), each
worker invocation gets EXACTLY what it needs: one task, plus a compressed
pointer to the original ask (per #6's "scoped context is not context
amnesia" rule -- the worker should still know what the overall goal was,
even scoped down to one task), plus its upstream dependency results.

This is a DIFFERENT state shape than OrchestratorState on purpose -- Send
explicitly supports invoking a node with custom state that differs from
the main graph state. The return value still merges back into
OrchestratorState via the normal `tasks` reducer.
"""

from __future__ import annotations

from typing import TypedDict

from state import Task


class WorkerInput(TypedDict):
    task: Task
    user_request: str  # compressed top-level goal, per #6's anti-drift rule
    dependency_results: dict[str, dict]  # {task_id: task.result} for each
                                          # id in task.depends_on -- the
                                          # scoped input for synthesis/
                                          # report_writer/judge, populated
                                          # by the router at dispatch time
                                          # (see routing.py)
