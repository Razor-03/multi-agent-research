"""
Real agent: report_writer (#8, final output generation).

Input: dependency_results from the synthesis task -- findings + any noted
conflicts. Writes actual prose, surfacing conflicts explicitly rather
than hiding them (same #8 principle carried through to the final
artifact the user sees).
"""

from __future__ import annotations

from langchain_google_genai import ChatGoogleGenerativeAI

from state import TaskStatus
from agent_common import WorkerInput


REPORT_SYSTEM_PROMPT = """You are a report-writing agent. You receive \
synthesized findings (and any noted conflicts between sources) and must \
write a clear, well-organized report answering the original request.

Rules:
- Structure the report with short sections/paragraphs, not just a list.
- If any conflicts were noted, explicitly mention the disagreement in the \
  report rather than silently picking a side.
- Cite sources inline where useful (e.g. "(source: ...)").
- Do not introduce claims that weren't in the findings.
"""


def report_writer_node(worker_input: WorkerInput) -> dict:
    t = worker_input["task"].model_copy(deep=True)
    t.transition(TaskStatus.RUNNING, "report_writer agent picked up task")

    try:
        synthesis_results = list(worker_input["dependency_results"].values())
        if not synthesis_results:
            raise ValueError("no synthesis output available to write from")

        # Only ever depends on exactly one synthesis task per our decomposition
        # prompt's rules -- take the first (and expected-only) result.
        synth = synthesis_results[0]

        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
        response = llm.invoke([
            ("system", REPORT_SYSTEM_PROMPT),
            ("human",
             f"Original request: {worker_input['user_request']}\n\n"
             f"Findings: {synth.get('findings', [])}\n\n"
             f"Conflicts noted: {synth.get('conflicts', [])}"),
        ])

        report_text = response.content
        t.result = {"report": report_text}
        t.transition(TaskStatus.DONE, "report written")
    except Exception as e:
        t.attempts += 1
        t.error = str(e)
        if t.attempts >= t.failure_policy.max_retries:
            t.transition(TaskStatus.FAILED, f"exhausted retries: {e}")
        else:
            t.transition(TaskStatus.READY,
                         f"attempt {t.attempts} failed, will retry: {e}")

    return {"tasks": {t.id: t}, "step_count": 1}
