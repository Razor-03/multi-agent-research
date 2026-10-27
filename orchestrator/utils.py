"""Shared output formatting -- used by both run.py (live) and
tests_offline.py (offline test harness) so the trace format stays
consistent everywhere it's printed."""

from __future__ import annotations


def print_result(result):
    print("=== FINAL STATE ===")
    print("is_complete:", result["is_complete"])
    print("step_count:", result["step_count"])
    print("final_report:", result["final_report"])
    print()
    print("=== FULL LEDGER TRACE ===")
    for task_id, task in sorted(result["tasks"].items()):
        print(f"\n[{task_id}] agent={task.assigned_agent} depends_on={task.depends_on}")
        for event in task.history:
            print(f"    {event.from_status} -> {event.to_status} :: {event.detail}")
