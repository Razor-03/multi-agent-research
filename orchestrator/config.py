"""
Shared constants used across the orchestrator.

Kept in their own module because both `planner.py` (which validates
decomposition output) and `routing.py` (which dispatches Send calls) need
KNOWN_AGENTS, and both `routing.py`/`run.py` need DEFAULT_MAX_CONCURRENCY --
putting them here avoids either module importing the other just to reach a
constant.
"""

from __future__ import annotations

from state import TaskStatus

TERMINAL_STATUSES = {TaskStatus.DONE, TaskStatus.FAILED, TaskStatus.BLOCKED}

# The set of agent types our planner is allowed to assign work to. This is
# the thing that lets us scale to N agents later -- add a name here, add
# a node with that name in graph.py, done. No branching logic to touch.
KNOWN_AGENTS = {"web_search", "synthesis", "report_writer", "judge"}


# ---------------------------------------------------------------------------
# Concurrency cap (#11: rate limiting)
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
