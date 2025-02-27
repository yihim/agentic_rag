from typing import TypedDict, List, Union
from langchain_core.messages import HumanMessage, AIMessage, RemoveMessage
from torch.cuda import graph

from agents.engines.initial_answer_crafter import (
    InitialAnswerCrafterOutput,
    craft_initial_answer,
)
from agents.engines.final_answer_crafter import (
    FinalAnswerCrafterOutput,
    craft_final_answer,
)
from agents.engines.task_router import TaskRouterAction, router_action
from agents.engines.query_rewriter import QueryWriterOutput, rewrite_query
from agents.engines.response_checker import ResponseCheckerOutput, check_response
from agents.engines.conversation_summarizer import (
    ConversationSummarizerOutput,
    summarize_conversation,
)
from agents.engines.query_classifier import QueryClassifierOutput, classify_query
from agents.engines.conversation_responder import (
    ConversationalResponderOutput,
    response_conversation,
)
from agents.tools.web_search import tavily_search, TavilySearchInput
from agents.tools.vectorstore_retriever import milvus_retriever
from agents.utils.models import load_chat_model
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, END
from langchain_core.runnables.graph import MermaidDrawMethod
from IPython.display import Image
import asyncio
import langchain

# langchain.debug = True


class AgentState(TypedDict):
    messages: List[Union[HumanMessage, AIMessage]]
    query: str
    rewritten_query: str
    kb_context: str
    web_context: str
    answer: str
    response_check: str
    conversation_summary: str
    current_step: str
    task_history: List[str]
    is_conversational: str


def create_multi_agents():
    memory = MemorySaver()

    llm = load_chat_model()

    def execute_classify_query(state: AgentState):

        state["rewritten_query"] = ""
        state["kb_context"] = ""
        state["web_context"] = ""
        state["answer"] = ""
        state["response_check"] = ""
        state["current_step"] = ""
        state["task_history"] = [""]
        state["is_conversational"] = ""

        classify_result = classify_query(
            llm=llm.with_structured_output(QueryClassifierOutput), query=state["query"]
        )

        print(classify_result)

        return {"is_conversational": classify_result}

    def execute_respond_conversation(state: AgentState):
        response = response_conversation(
            llm=llm.with_structured_output(ConversationalResponderOutput),
            query=state["query"],
            chat_history=state["messages"],
            conversation_summary=state["conversation_summary"],
        )

        print(response)

        messages = state["messages"].copy()
        messages.append(AIMessage(content=response))

        return {"messages": messages}

    def task_router_node(state: AgentState):
        """Central task router that determines the next action"""

        router_result = router_action(
            llm=llm.with_structured_output(TaskRouterAction),
            query=state["query"],
            rewritten_query=state["rewritten_query"],
            current_step=state["current_step"],
            kb_context=state["kb_context"],
            web_context=state["web_context"],
            answer=state["answer"],
            response_check=state["response_check"],
            task_history=state["task_history"],
        )

        task_history = state["task_history"].copy()
        current_step = state["current_step"] if state["current_step"] else "Initial"
        task_history.append(f"{current_step} → {router_result.action}")

        print(router_result.action)
        print(router_result.reasoning)

        return {"current_step": router_result.action, "task_history": task_history}

    def execute_rewrite_query(state: AgentState):
        rewritten = rewrite_query(
            llm=llm.with_structured_output(QueryWriterOutput),
            query=state["query"],
            chat_history=state["messages"],
            conversation_summary=state["conversation_summary"],
        )

        print(rewritten)

        return {"rewritten_query": rewritten}

    def execute_milvus_retrieve(state: AgentState):
        context = milvus_retriever(query=state["rewritten_query"])
        if context is not None:
            formatted_context = "Local Knowledge Base Results:\n\n"
            for index, (item, score) in enumerate(context):
                formatted_context += (
                    f"{index + 1}.\n{item}\nConfidence Score: {score * 100:.2f}%\n\n"
                )

            print(formatted_context)
        else:
            print("No data found in Milvus vector store.")
            formatted_context = ""

        return {"kb_context": formatted_context}

    def execute_tavily_search(state: AgentState):
        search_results = tavily_search(
            query=TavilySearchInput(query=state["rewritten_query"])
        )
        formatted_search_results = f"Tavily Search Results:\n\nContent: {search_results.answer}\nSources: {search_results.sources}\nConfidence Score: {search_results.confidence_score}"

        print(formatted_search_results)

        return {"web_context": formatted_search_results}

    def execute_craft_initial_answer(state: AgentState):
        query = state["rewritten_query"]
        context = state["kb_context"] if state["kb_context"] else state["web_context"]

        response = craft_initial_answer(
            llm=llm.with_structured_output(InitialAnswerCrafterOutput),
            query=query,
            context=context,
        )

        print(response)

        return {"answer": response}

    def execute_check_response(state: AgentState):
        check_result = check_response(
            llm=llm.with_structured_output(ResponseCheckerOutput),
            query=state["rewritten_query"],
            answer=state["answer"],
        )

        print(check_result)

        return {"response_check": check_result}

    def execute_craft_final_answer(state: AgentState):
        answer = state["answer"]

        response = craft_final_answer(
            llm=llm.with_structured_output(FinalAnswerCrafterOutput), answer=answer
        )

        print(response)

        messages = state["messages"].copy()
        messages.append(AIMessage(content=response))

        return {"messages": messages}

    def execute_summarize_conversation(state: AgentState):
        messages = state["messages"].copy()
        task_history = state["task_history"].copy()

        if len(messages) > 10:
            summary = state["conversation_summary"]
            summarized_conversation = summarize_conversation(
                llm=llm.with_structured_output(ConversationSummarizerOutput),
                summary=summary,
                messages=messages,
            )

            print(summarized_conversation)

            delete_messages = [RemoveMessage(id=m.id) for m in messages]
            print(f"\n\nTask history: {task_history}")

            return {
                "conversation_summary": summarized_conversation,
                "messages": delete_messages,
            }

        else:
            print(f"\n\nTask history: {task_history}")
            return {"messages": messages}

    def initial_routing(state: AgentState) -> str:
        if state["is_conversational"].lower() == "true":
            return "conversation_responder"
        else:
            return "task_router"

    def execute_retry(state: AgentState):
        state["task_history"].append("retry -> query_rewriter")
        state["current_step"] = "query_rewriter"
        return {}

    def get_next_task(state: AgentState):
        current_step = state["current_step"]

        if current_step == "query_rewriter":
            return "query_rewriter"
        elif current_step == "milvus_retriever":
            return "milvus_retriever"
        elif current_step == "tavily_searcher":
            return "tavily_searcher"
        elif current_step == "initial_answer_crafter":
            return "initial_answer_crafter"
        elif current_step == "response_checker":
            return "response_checker"
        elif current_step == "final_answer_crafter":
            return "final_answer_crafter"
        elif current_step == "conversation_summarizer":
            return "conversation_summarizer"

    def route_to_knowledge_source(state: AgentState):
        if state["kb_context"]:
            return "initial_answer_crafter"
        else:
            return "tavily_searcher"

    def evaluate_response_quality(state: AgentState):
        if state["response_check"].lower() == "yes":
            return "final_answer_crafter"
        else:
            return "retry"

    workflow = StateGraph(AgentState)

    workflow.add_node("task_router", task_router_node)
    workflow.add_node("query_classifier", execute_classify_query)
    workflow.add_node("conversation_responder", execute_respond_conversation)
    workflow.add_node("query_rewriter", execute_rewrite_query)
    workflow.add_node("milvus_retriever", execute_milvus_retrieve)
    workflow.add_node("tavily_searcher", execute_tavily_search)
    workflow.add_node("initial_answer_crafter", execute_craft_initial_answer)
    workflow.add_node("response_checker", execute_check_response)
    workflow.add_node("final_answer_crafter", execute_craft_final_answer)
    workflow.add_node("conversation_summarizer", execute_summarize_conversation)
    workflow.add_node("retry", execute_retry)

    workflow.add_conditional_edges(
        "query_classifier",
        initial_routing,
        {
            "conversation_responder": "conversation_responder",
            "task_router": "task_router",
        },
    )

    workflow.add_conditional_edges(
        "task_router",
        get_next_task,
        {
            "query_rewriter": "query_rewriter",
            "milvus_retriever": "milvus_retriever",
            "initial_answer_crafter": "initial_answer_crafter",
            "response_checker": "response_checker",
        },
    )

    workflow.add_conditional_edges(
        "milvus_retriever",
        route_to_knowledge_source,
        {
            "initial_answer_crafter": "initial_answer_crafter",
            "tavily_searcher": "tavily_searcher",
        },
    )

    workflow.add_conditional_edges(
        "response_checker",
        evaluate_response_quality,
        {
            "final_answer_crafter": "final_answer_crafter",
            "retry": "retry",
        },
    )

    workflow.add_edge("conversation_responder", "conversation_summarizer")
    workflow.add_edge("query_rewriter", "task_router")
    workflow.add_edge("tavily_searcher", "task_router")
    workflow.add_edge("initial_answer_crafter", "task_router")
    workflow.add_edge("final_answer_crafter", "conversation_summarizer")
    workflow.add_edge("retry", "query_rewriter")
    workflow.add_edge("conversation_summarizer", END)

    workflow.set_entry_point("query_classifier")

    app = workflow.compile(checkpointer=memory)

    return app


async def main():
    import uuid

    session_id = uuid.uuid4().hex[:8]
    config = {"configurable": {"thread_id": session_id}}

    graph = create_multi_agents()

    query = "what is agentic ai?"

    async for msg, metadata in graph.astream(
        {
            "messages": [HumanMessage(content=query)],
            "query": query,
            "rewritten_query": "",
            "kb_context": "",
            "web_context": "",
            "answer": "",
            "response_check": "",
            "conversation_summary": "",
            "current_step": "",
            "task_history": [],
            "is_conversational": "",
        },
        config=config,
        stream_mode="messages",
    ):
        if (
            msg.content
            and not isinstance(msg, HumanMessage)
            and isinstance(msg, AIMessage)
        ):
            print(msg.content, end="")


if __name__ == "__main__":
    asyncio.run(main())

    # # Visualize the graph
    # graph = create_multi_agents()
    # try:
    #     img = Image(
    #         graph.get_graph().draw_mermaid_png(
    #             draw_method=MermaidDrawMethod.API,
    #         )
    #     )
    #
    #     with open("./graph_visualization.png", "wb") as f:
    #         f.write(img.data)
    # except Exception:
    #     pass