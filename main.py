import os
from typing import TypedDict

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END


# 1. Define the shape of the state that flows through the graph
class GraphState(TypedDict):
    question: str
    answer: str


# 2. Load .env and set up the LLM
load_dotenv()  # reads .env in the current working directory into os.environ

if not os.environ.get("GOOGLE_API_KEY"):
    raise RuntimeError(
        "GOOGLE_API_KEY not found. Create a .env file with:\n"
        "  GOOGLE_API_KEY=your-key-here"
    )

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")


# 3. Define a node — just a function that takes state, returns updated state
def call_gemini(state: GraphState) -> GraphState:
    response = llm.invoke(state["question"])
    return {"question": state["question"], "answer": response.content}


# 4. Build the graph
builder = StateGraph(GraphState)
builder.add_node("ask_gemini", call_gemini)
builder.set_entry_point("ask_gemini")
builder.add_edge("ask_gemini", END)

graph = builder.compile()


# 5. Run it
if __name__ == "__main__":
    result = graph.invoke({"question": "In one sentence, what is LangGraph?"})
    print("\n--- RESULT ---")
    print(result["answer"])