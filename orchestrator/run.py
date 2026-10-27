"""
Live entry point. Requires GOOGLE_API_KEY (and TAVILY_API_KEY for real
search). Run with: python3 run.py

Prompts for a query, runs the full pipeline, and -- if the judge flags
the report FAIL -- pauses for human review right here in the terminal
before finishing.
"""

from __future__ import annotations

import os
import uuid

from state import OrchestratorState
from config import DEFAULT_MAX_CONCURRENCY
from graph import build_graph
from utils import print_result

from langgraph.types import Command


def run_live(max_concurrency: int = DEFAULT_MAX_CONCURRENCY):
    """`max_concurrency` caps simultaneous node executions per graph step
    -- this is the actual rate-limit guard, not just a suggestion."""
    app = build_graph()

    user_request = input("Enter a query: ").strip()
    if not user_request:
        print("No query entered, exiting.")
        return

    # thread_id identifies THIS run to the checkpointer -- required for
    # interrupt()/resume to work. A fresh uuid per run is fine for a CLI
    # tool; a real app would likely tie this to a user/session id instead.
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id},
              "max_concurrency": max_concurrency}

    initial_state: OrchestratorState = {
        "user_request": user_request,
        "tasks": {},
        "step_count": 0,
        "max_steps": 25,
        "final_report": None,
        "is_complete": False,
        "needs_human_review": False,
        "human_reviewed": False,
    }

    result = app.invoke(initial_state, config=config)

    # Keep resuming as long as the graph keeps pausing for review -- a
    # single run could in principle hit review more than once (e.g.
    # retry_report -> still fails -> review again).
    while "__interrupt__" in result:
        payload = result["__interrupt__"][0].value
        print("\n" + "=" * 60)
        print("HUMAN REVIEW REQUIRED")
        print("=" * 60)
        print(f"Reason: {payload['reason']}")
        if payload.get("verdict"):
            print(f"Judge verdict: {payload['verdict']}")
        print(f"\nReport:\n{payload.get('report', '(no report available)')}")
        print(f"\nOptions: {payload['options']}")

        action = input("\nYour decision: ").strip()
        if action not in payload["options"]:
            print(f"'{action}' not recognized, defaulting to 'accept'")
            action = "accept"

        result = app.invoke(Command(resume={"action": action}), config=config)

    print_result(result)


if __name__ == "__main__":
    if not os.environ.get("GOOGLE_API_KEY"):
        print("GOOGLE_API_KEY not set. Set it and (optionally) "
              "TAVILY_API_KEY, then re-run. For offline testing without "
              "API keys, use tests_offline.py instead.")
    else:
        run_live()
