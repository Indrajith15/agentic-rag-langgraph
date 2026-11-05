from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from retrieval import retrieve_documents
import os

load_dotenv()
llm = ChatGoogleGenerativeAI(
    model = "gemini-3.5-flash",
    google_api_key = os.getenv("GOOGLE_API_KEY"),
    temperature = 0
)

class AgenticRAGState(TypedDict):
    question : str
    decision : str
    documents : list
    generation : str
    retry_count: int

def analyze_question(state: AgenticRAGState):
    question = state["question"]
    prompt = f"""
    You are a question analysis assistant.
    
    Decide whether the user's question requires information from the technical knowledge
    documents.
    
    If the question is about technical topics coverd by a knowledge base. choose retrieve.
    
    If it is general conversation, simple mathematics, or unrelated general knowledge, choose
    direct.
    
    Return only one word:
    retrieve
    direct
    
    Question: {question}
    """
    response = llm.invoke(prompt)

    decision = response.text.lower().strip()
    if decision not in ["retrieve", "direct"]:
        decision = "retrieve"

    return {"decision": decision}



def route_question(state: AgenticRAGState):
    decision = state["decision"]

    if decision == "retrieve":
        return "retrieve"
    else:
        return "direct"



def retrieve_node(state: AgenticRAGState):

    question = state["question"]
    retry_count = state["retry_count"]

    documents = retrieve_documents(question)

    return {
        "documents": documents,
        "retry_count": retry_count + 1
    }



def get_response_text(response):
    content = response.content

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        return "".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
        )

    return str(content)


def grade_documents(state: AgenticRAGState):
    question = state["question"]
    documents = state["documents"]

    relevant_documents = []

    for document in documents:
        prompt = f"""
        You are a document relevance grader.

        Determine whether the following document is relevant
        to answering the user's question.

        Question: {question}

        Document:
        {document.page_content}

        Return ONLY one word:
        yes
        no
        """

        response = llm.invoke(prompt)
        grade = get_response_text(response).strip().lower()

        if grade.startswith("yes"):
            relevant_documents.append(document)

    print(f"Relevant documents: {len(relevant_documents)}")

    return {"documents": relevant_documents}




def route_after_grading(state: AgenticRAGState):

    if state["documents"]:
        return "generate"

    if state["retry_count"] < 2:
        return "retrieve"

    return "fallback"





def generate_answer(state: AgenticRAGState):

    question = state["question"]
    documents = state["documents"]

    context = "\n\n".join(
        document.page_content for document in documents
    )

    prompt = f"""
    Answer the user's question using ONLY the provided context.

    If the answer is not available in the context,
    say that you don't have enough information.

    Context:
    {context}

    Question:
    {question}
    """

    response = llm.invoke(prompt)

    return {
        "generation": get_response_text(response)
    }


def fallback_node(state: AgenticRAGState):

    return {
        "generation": "I couldn't find sufficiently relevant information in the knowledge base to answer your question."
    }




def direct_answer_node(state: AgenticRAGState):

    question = state["question"]

    response = llm.invoke(question)

    return {
        "generation": get_response_text(response)
    }



graph = StateGraph(AgenticRAGState)

# Register nodes
graph.add_node("analyze", analyze_question)
graph.add_node("retrieve", retrieve_node)
graph.add_node("direct", direct_answer_node)
graph.add_node("grade", grade_documents)
graph.add_node("generate", generate_answer)
graph.add_node("fallback", fallback_node)

# Start with question analysis
graph.add_edge(START, "analyze")

# Conditional routing
graph.add_conditional_edges(
    "analyze",
    route_question,
    {
        "retrieve": "retrieve",
        "direct": "direct"
    }
)

# End after these nodes for now
graph.add_edge("retrieve", "grade")

graph.add_conditional_edges(
    "grade",
    route_after_grading,
    {
        "generate": "generate",
        "retrieve": "retrieve",
        "fallback": "fallback"
    }
)

graph.add_edge("generate", END)
graph.add_edge("fallback", END)
graph.add_edge("direct", END)

app = graph.compile()


if __name__ == "__main__":

    initial_state = {
        "question": "What is the monthly cost of Amazon EC2?",
        "decision": "",
        "documents": [],
        "generation": "",
        "retry_count": 0
    }

    result = app.invoke(initial_state)

    print("\nDECISION:", result["decision"])
    print("\nRESULT:", result)
    print("\nFINAL ANSWER:", result["generation"])