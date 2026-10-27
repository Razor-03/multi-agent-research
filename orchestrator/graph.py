"""
Graph assembly -- the ONE place that knows the full topology. Every other
module owns a single concern (planning, one agent, routing, HITL); this
file just wires them together into a compiled LangGraph app.

If you're trying to understand the system for the first time, this is a
reasonable place to start reading -- it's a map of every other module.
"""

from __future__ import annotations

from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import InMemorySaver

from state import OrchestratorState
from planner import planner_node
from agent_search import web_search_node
from agent_synthesis import synthesis_node
from agent_report_writer import report_writer_node
from agent_judge import judge_node
from human_review import human_review_node
from routing import route_after_planner


def build_graph():
    graph = StateGraph(OrchestratorState)
    graph.add_node("planner", planner_node)
    graph.add_node("web_search", web_search_node)
    graph.add_node("synthesis", synthesis_node)
    graph.add_node("report_writer", report_writer_node)
    graph.add_node("judge", judge_node)
    graph.add_node("human_review", human_review_node)

    graph.set_entry_point("planner")
    # No explicit mapping dict needed here -- route_after_planner returns
    # either a plain node name (str), END, or a list of Send objects, and
    # LangGraph resolves Send targets dynamically at runtime.
    graph.add_conditional_edges("planner", route_after_planner)

    for agent in ("web_search", "synthesis", "report_writer", "judge", "human_review"):
        graph.add_edge(agent, "planner")

    # Checkpointer required for interrupt()/Command(resume=...) to work at
    # all -- see human_review.py. InMemorySaver is dev-only -- state is
    # lost when the process exits. A real deployment would swap this for
    # a durable backend (Postgres, Redis, etc.), but that's a deliberate
    # LATER decision, not something to guess at now.
    #
    # KNOWN ISSUE: InMemorySaver's default serializer warns about our
    # custom Task/TaskStatus types ("Deserializing unregistered type...
    # This will be blocked in a future version"). Works today, will break
    # in a future langgraph version -- needs registering these types with
    # the checkpointer's allowed list, or switching checkpointed state to
    # plain dicts. Not yet addressed.
    return graph.compile(checkpointer=InMemorySaver())
