"""
Real agent: web_search (real tool call via Tavily, #7).

Note what this node's STRUCTURE doesn't depend on: the actual search
implementation. `real_web_search` takes a query string and returns
normalized result dicts -- the node itself only cares about that
contract, not about Tavily specifically. Swapping search providers later
means touching only `real_web_search`, nothing here.

Requires TAVILY_API_KEY to be set. If it's missing, TavilySearch's own
client raises at call time -- which lands in the try/except below and
becomes a ledger FAILED/retry state (#9), not a crash. See `_get_tavily_tool`
for why the client is lazily instantiated rather than created at import
time (a real bug we hit and fixed during development).
"""

from __future__ import annotations

from langchain_tavily import TavilySearch

from state import TaskStatus
from agent_common import WorkerInput

_tavily_tool = None


def _get_tavily_tool() -> TavilySearch:
    """Lazy singleton, NOT instantiated at import time.

    Bug we hit and fixed: creating TavilySearch() at module load means a
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
    """Real search via Tavily. Takes a query string, returns a list of
    result dicts with at least 'snippet'-equivalent text and a source
    URL -- normalized here so web_search_node doesn't need to know
    Tavily's specific response shape."""
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
        t.transition(TaskStatus.DONE, f"search returned {len(results)} results")
    except Exception as e:  # #9: real failures become ledger state, not crashes
        t.attempts += 1
        t.error = str(e)
        if t.attempts >= t.failure_policy.max_retries:
            t.transition(TaskStatus.FAILED, f"exhausted retries: {e}")
        else:
            t.transition(TaskStatus.READY, f"attempt {t.attempts} failed, will retry: {e}")

    return {"tasks": {t.id: t}, "step_count": 1}
