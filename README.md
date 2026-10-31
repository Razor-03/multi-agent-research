# Multi-Agent Research Orchestrator

A command-line research assistant built with LangGraph, Google Gemini, and Tavily. It turns a question into a dependency graph of tasks, runs independent searches in parallel, and produces an evaluated report with source references.

Simple requests can receive a direct answer from the planner without running the research pipeline.

## How it works

1. **Plan:** Gemini decides whether research is needed and decomposes the request into tasks.
2. **Search:** Independent web-search tasks retrieve results from Tavily.
3. **Synthesize:** An agent combines search claims into structured findings and records disagreements between sources.
4. **Write:** A report-writing agent answers the original question using those findings.
5. **Evaluate:** A judge checks relevance, grounding in the findings, and handling of conflicts. A `fail` verdict pauses execution for human review.

The planner coordinates each stage through a shared task ledger. Workers receive their own task, the original request, and only their dependencies' results. LangGraph's `Send` API dispatches ready tasks concurrently.

## Requirements

- Python **3.13 or newer** (`.python-version` selects 3.13).
- [uv](https://docs.astral.sh/uv/) for the setup commands below.
- A Google API key for Gemini calls.
- A Tavily API key for live web research.

The Gemini agents currently use `gemini-2.5-flash`. Dependencies are declared in `pyproject.toml` and locked in `uv.lock`.

## Setup

From the repository root, install the locked dependencies:

```powershell
uv sync --locked
```

Create a `.env` file in the repository root:

```dotenv
GOOGLE_API_KEY=your_google_api_key
TAVILY_API_KEY=your_tavily_api_key
```

`.env` is excluded from Git. The application does not load this file itself; use the `--env-file` option shown below, or set the keys in your shell environment.

## Run

```powershell
uv run --env-file .env python orchestrator/run.py
```

At the `Enter a query:` prompt, enter a research question, for example:

```text
Compare the approaches to long-term memory used by current AI agent frameworks, citing sources.
```

The CLI prints the final report, completion state, step count, and each task's transition history. Research reports include an evaluation banner with a verdict of `PASS`, `WEAK`, or `FAIL`; if the judge cannot complete, the banner states that evaluation is unavailable.

Run scripts by their file paths as shown: the current source uses imports between sibling modules.

### Human review

When the judge returns `fail`, the CLI displays the report and evaluation, then accepts one of these actions:

| Action | Effect |
| --- | --- |
| `accept` | Finish with the current report and its evaluation. |
| `retry_report` | Reset the report writer and judge, then rerun them. |
| `retry_synthesis` | Reset synthesis, report writing, and evaluation, then rerun them using the existing search results. |
| `abort` | End the run with an abort message. |

A retry can trigger another review if evaluation fails again. An unrecognized response at the CLI prompt defaults to `accept`.

## Checks and demonstrations

These checks use seeded or mocked inputs and require no API keys:

```powershell
uv run python orchestrator/tests_offline.py
uv run python orchestrator/tests_offline.py --trivial
uv run python orchestrator/tests_offline.py --concurrency
uv run python orchestrator/tests_offline.py --hitl
uv run python orchestrator/hitl_toy_test.py
```

They cover routing, the direct-answer path, concurrent dispatch and its cap, human-review resume/reset behavior, and a minimal interrupt/resume demonstration.

The `--synthesis` and `--judge` modes in `tests_offline.py` exercise real Gemini-backed nodes. They can make API calls when credentials are available. Some test comments and the `--judge` failure assertion still refer to `ANTHROPIC_API_KEY`, although the implementation uses Google Gemini; these modes need that mismatch addressed before being treated as reliable live checks.

## Project structure

```text
orchestrator/
  run.py                  Interactive CLI and review/resume loop
  graph.py                LangGraph topology and in-memory checkpointer
  planner.py              Request triage, decomposition, readiness, and completion
  routing.py              Dispatch of ready tasks with scoped dependency results
  state.py                Task models, status history, and state reducers
  config.py               Agent registry and default concurrency limit
  agent_common.py         Shared worker input contract
  agent_search.py         Tavily search and normalized claims
  agent_synthesis.py      Structured findings and source conflicts
  agent_report_writer.py  Final report generation
  agent_judge.py          Relevance and grounding evaluation
  human_review.py         Interrupt payloads and task resets
  utils.py                Report and ledger trace formatting
  tests_offline.py        Routing, concurrency, review, and agent checks
  hitl_toy_test.py        Minimal interrupt/resume demonstration
pyproject.toml            Python requirements and dependencies
uv.lock                   Dependency lockfile
```

## Configuration and current limits

- **Concurrency:** `DEFAULT_MAX_CONCURRENCY` in `orchestrator/config.py` defaults to `4`. `run_live(max_concurrency=...)` passes the cap to LangGraph; there is no CLI flag for it.
- **Execution budget:** The initial state in `orchestrator/run.py` sets `max_steps` to `25`. The counter accumulates planner and worker updates, including parallel workers; it is not simply the number of graph rounds. Reaching the cap can end a run before a final report is assembled.
- **Failure handling:** Workers record exceptions in the ledger and return tasks to `READY` until the failure threshold is reached. The default `FailurePolicy.max_retries` is `2`; current workers mark a task failed on its second failed attempt. Failed dependencies block downstream tasks.
- **Persistence:** `InMemorySaver` supports pause/resume within the running process. Checkpoints are lost when the process exits. The source also notes serializer warnings for custom task types that need attention for future compatibility.
- **Evaluation:** The judge compares the report with the synthesis findings. Its verdict is a quality signal rather than independent verification of the original sources.
