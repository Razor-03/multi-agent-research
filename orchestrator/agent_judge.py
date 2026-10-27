"""
Real agent: judge (#10 Evaluation, distinct from Observability).

Observability (the ledger's `history`/trace) answers "what happened."
This answers "was what happened actually GOOD." Two different questions.

Deliberately lightweight: one LLM-as-judge call, structured output, not
a full eval harness. Checks the report against (a) the original request
-- did it actually answer what was asked, and (b) the synthesis findings
-- did it stay grounded, or does it contain claims that weren't in the
findings (a proxy for hallucination in this pipeline, since findings are
the only source of truth the report_writer had access to).

NOTE: this is a QUALITY signal, not a gate on its own. A "fail" verdict
is the trigger condition for human_review.py's HITL pause -- see that
module for what happens next. Everything here just produces the signal.
"""

from __future__ import annotations

from typing import Literal

from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from state import TaskStatus
from agent_common import WorkerInput


class EvaluationResult(BaseModel):
    verdict: Literal["pass", "weak", "fail"] = Field(
        description="pass: answers the request and is well-grounded. "
                    "weak: partially answers it or has minor grounding issues. "
                    "fail: does not answer the request, or contains claims "
                    "not supported by the findings."
    )
    answers_request: bool
    grounded_in_findings: bool = Field(
        description="false if the report contains claims not present in "
                    "the synthesis findings it was given"
    )
    issues: list[str] = Field(
        default_factory=list,
        description="specific problems found, if any -- empty if verdict is pass",
    )
    reasoning: str = Field(description="brief justification for the verdict")


JUDGE_SYSTEM_PROMPT = """You are a skeptical, strict evaluator reviewing a \
report before it is shown to a user. You are not the report's author --
do not be charitable to it.

Check:
1. Does the report actually answer the original request, or does it \
   drift off-topic / answer a different question?
2. Is every claim in the report traceable to the provided findings? Flag \
   anything that looks fabricated or added beyond what the findings support.
3. If the findings noted any conflicts, does the report handle them \
   honestly (mentioning the disagreement) rather than silently picking a side?

Be concrete in `issues` -- vague verdicts aren't useful. If the report is \
genuinely solid, say so plainly; don't invent nitpicks to seem thorough.
"""


def judge_node(worker_input: WorkerInput) -> dict:
    t = worker_input["task"].model_copy(deep=True)
    t.transition(TaskStatus.RUNNING, "judge agent picked up task")

    try:
        deps = worker_input["dependency_results"]
        # deps contains BOTH the report_writer result ({"report": ...}) and
        # the synthesis result ({"findings": ..., "conflicts": ...}) --
        # distinguish them by shape rather than by id, since the judge
        # shouldn't need to know task ids, only result shapes (#6 scoping).
        report_text = None
        findings = None
        conflicts = None
        for dep_result in deps.values():
            if dep_result is None:
                continue
            if "report" in dep_result:
                report_text = dep_result["report"]
            if "findings" in dep_result:
                findings = dep_result["findings"]
                conflicts = dep_result.get("conflicts", [])

        if report_text is None:
            raise ValueError("no report found among dependencies to evaluate")

        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
        structured_llm = llm.with_structured_output(EvaluationResult)
        result: EvaluationResult = EvaluationResult.model_validate(
            structured_llm.invoke([
                ("system", JUDGE_SYSTEM_PROMPT),
                ("human",
                f"Original request: {worker_input['user_request']}\n\n"
                f"Report to evaluate:\n{report_text}\n\n"
                f"Findings it was based on: {findings}\n\n"
                f"Conflicts noted in findings: {conflicts}"),
            ])
        )

        t.result = result.model_dump()
        t.transition(TaskStatus.DONE, f"evaluation verdict: {result.verdict}")
    except Exception as e:
        t.attempts += 1
        t.error = str(e)
        if t.attempts >= t.failure_policy.max_retries:
            # Judge failing is NOT the same severity as report_writer
            # failing -- we still have a report, just no quality signal
            # on it. Handled explicitly in planner.py's final assembly
            # rather than treated as a plan-level failure (#9: different
            # failure types warrant different consequences).
            t.transition(TaskStatus.FAILED, f"exhausted retries: {e}")
        else:
            t.transition(TaskStatus.READY,
                         f"attempt {t.attempts} failed, will retry: {e}")

    return {"tasks": {t.id: t}, "step_count": 1}
