"""
Real agent: synthesis (#8 result aggregation, #7 tool calling / structured
output).

Input: dependency_results, scoped to exactly this task's upstream tasks
(#6). Each web_search task's result is {"claims": [{"text", "source",
"confidence"}]} -- structured, not prose, per our #6/#8 discussion: the
map step (extracting claims) already happened at search time, so
synthesis's job is purely the REDUCE step -- combine structured claims
into a coherent, deduplicated set of findings.

Explicitly instructed on conflict/redundancy handling (#8's sharp edge):
corroborating claims get merged with combined attribution, conflicting
claims get surfaced as a disagreement rather than silently resolved.
"""

from __future__ import annotations

from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from state import TaskStatus
from agent_common import WorkerInput


class SynthesizedFinding(BaseModel):
    finding: str = Field(
        description="a single synthesized point, in Claude's own words")
    sources: list[str] = Field(
        description="source URLs supporting this finding")
    corroborated: bool = Field(
        description="true if 2+ independent sources support this finding"
    )


class SynthesisOutput(BaseModel):
    findings: list[SynthesizedFinding]
    conflicts: list[str] = Field(
        default_factory=list,
        description="descriptions of any claims that directly contradicted each other",
    )


SYNTHESIS_SYSTEM_PROMPT = """You are a synthesis agent. You receive raw \
claims extracted from multiple web searches and must combine them into a \
clean set of findings.

Rules:
- If multiple sources support the same point, merge them into ONE finding \
  with all supporting source URLs attached, and mark corroborated=true.
- If sources DIRECTLY CONTRADICT each other, do NOT silently pick one. \
  Add an entry to `conflicts` describing the disagreement, and still \
  include both claims as separate findings with corroborated=false.
- Do not invent claims that aren't present in the input.
- Keep each finding concise -- one clear point per finding.
"""


def synthesis_node(worker_input: WorkerInput) -> dict:
    t = worker_input["task"].model_copy(deep=True)
    t.transition(TaskStatus.RUNNING, "synthesis agent picked up task")

    try:
        # Flatten all upstream claims into one list the LLM can reason over.
        all_claims = []
        for dep_id, dep_result in worker_input["dependency_results"].items():
            for claim in dep_result.get("claims", []):
                all_claims.append({**claim, "from_task": dep_id})

        if not all_claims:
            # No upstream data at all (e.g. every search failed but this
            # task still got dispatched somehow) -- fail loudly rather
            # than fabricate a synthesis from nothing.
            raise ValueError("no upstream claims available to synthesize")

        llm = llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash", temperature=0)
        structured_llm = llm.with_structured_output(SynthesisOutput)
        output: SynthesisOutput = SynthesisOutput.model_validate(
            structured_llm.invoke([
                ("system", SYNTHESIS_SYSTEM_PROMPT),
                ("human",
                 f"Original request: {worker_input['user_request']}\n\n"
                 f"Raw claims:\n{all_claims}"),
            ])
        )

        t.result = {
            "findings": [f.model_dump() for f in output.findings],
            "conflicts": output.conflicts,
        }
        t.transition(
            TaskStatus.DONE,
            f"synthesized {len(output.findings)} findings "
            f"({len(output.conflicts)} conflicts noted)",
        )
    except Exception as e:  # #9: same pattern as web_search_node
        t.attempts += 1
        t.error = str(e)
        if t.attempts >= t.failure_policy.max_retries:
            t.transition(TaskStatus.FAILED, f"exhausted retries: {e}")
        else:
            t.transition(TaskStatus.READY,
                         f"attempt {t.attempts} failed, will retry: {e}")

    return {"tasks": {t.id: t}, "step_count": 1}
