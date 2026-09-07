"""
Throwaway toy graph -- proves HITL mechanics in isolation before we touch
the real pipeline. Two things to verify empirically, not just read about:

1. How does app.invoke() actually signal "I've paused, waiting on a
   human"? (vs. raising an exception, vs. something else)
2. The re-execution gotcha: does code BEFORE interrupt() in a node
   actually re-run when we resume? If so, that node needs to be designed
   with that in mind (idempotent, or interrupt() called first thing).
"""

from typing import TypedDict

from langgraph.graph import StateGraph, END
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import InMemorySaver


class ToyState(TypedDict):
    counter: int
    log: list[str]


side_effect_count = {"n": 0}  # tracks how many times the pre-interrupt code actually runs


def before_review(state: ToyState) -> dict:
    return {"log": state["log"] + ["before_review ran"]}


def human_review(state: ToyState) -> dict:
    # THIS is the line we're testing: does everything above this line
    # in THIS node re-run when we resume?
    side_effect_count["n"] += 1
    print(f"  [inside human_review node -- entered for the {side_effect_count['n']}(th) time]")

    decision = interrupt({"question": "approve or reject?", "counter": state["counter"]})
    print(f"  [resumed with decision: {decision}]")

    return {"log": state["log"] + [f"human said: {decision}"]}


def after_review(state: ToyState) -> dict:
    return {"log": state["log"] + ["after_review ran"], "counter": state["counter"] + 1}


graph = StateGraph(ToyState)
graph.add_node("before_review", before_review)
graph.add_node("human_review", human_review)
graph.add_node("after_review", after_review)
graph.set_entry_point("before_review")
graph.add_edge("before_review", "human_review")
graph.add_edge("human_review", "after_review")
graph.add_edge("after_review", END)

# THE KEY DIFFERENCE from everything we've built so far: interrupt() only
# works with a checkpointer attached. Without one, there's no mechanism to
# pause execution and resume it later -- there'd be nothing to resume FROM.
app = graph.compile(checkpointer=InMemorySaver())


if __name__ == "__main__":
    # Every invoke/resume pair needs a thread_id -- this is how the
    # checkpointer knows which paused run you're resuming.
    config = {"configurable": {"thread_id": "test-thread-1"}}

    print("=== FIRST INVOKE (should pause at interrupt) ===")
    result = app.invoke({"counter": 0, "log": []}, config=config)
    print("\nReturned from invoke(). Result keys:", list(result.keys()))
    print("Full result:", result)

    if "__interrupt__" in result:
        interrupt_info = result["__interrupt__"]
        print(f"\nGot an interrupt! Payload: {interrupt_info[0].value}")

        print("\n=== RESUMING with a decision ===")
        result2 = app.invoke(Command(resume="approved"), config=config)
        print("\nFinal result:", result2)

        print(f"\n=== side_effect_count: {side_effect_count['n']} ===")
        if side_effect_count["n"] == 2:
            print("CONFIRMED: the node DID re-execute from the top on resume "
                  "(entered twice for one logical pause) -- the gotcha is real.")
        else:
            print(f"UNEXPECTED: entered {side_effect_count['n']} times, investigate")
    else:
        print("\nNo interrupt occurred -- something's wrong with the setup")