from typing import TypedDict, List, Union, Literal
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
)
from sentence_transformers import SentenceTransformer
from constants.models import EMBEDDING_MODEL
import torch
from engines.initial_answer_crafter import (
    InitialAnswerCrafterOutput,
    craft_initial_answer,
)
from engines.final_answer_crafter import (
    craft_final_answer,
)
from engines.task_router import TaskRouterAction, router_action
from engines.query_rewriter import QueryWriterOutput, rewrite_query
from engines.response_checker import ResponseCheckerOutput, check_response
from engines.query_classifier import QueryClassifierOutput, classify_query
from engines.conversation_responder import (
    response_conversation,
)
from tools.web_search import tavily_search, TavilySearchInput
from tools.vectorstore_retriever import milvus_retriever
from utils.models import load_chat_model
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableConfig

# from langchain_core.runnables.graph import MermaidDrawMethod
# from IPython.display import Image
import asyncio
import langchain
import logging

logger = logging.getLogger(__name__)

# langchain.debug = True

device = "cuda" if torch.cuda.is_available() else "cpu"


class AgentState(TypedDict):
    messages: List[Union[HumanMessage, AIMessage]]
    query: str
    rewritten_query: str
    kb_context: str
    web_context: str
    answer: str
    response_check: str
    current_step: str
    task_action_history: List[str]
    task_action_reason_history: List[str]
    is_conversational: str


def create_multi_agents(embedding_model: SentenceTransformer) -> StateGraph.compile:
    memory = MemorySaver()

    llm = load_chat_model()

    def execute_classify_query(state: AgentState) -> dict:

        classify_result = classify_query(
            llm=llm.with_structured_output(QueryClassifierOutput), query=state["query"]
        )

        logger.info(f"Is conversational? {classify_result}")

        return {
            "rewritten_query": "",
            "kb_context": "",
            "web_context": "",
            "answer": "",
            "response_check": "",
            "current_step": "",
            "task_action_history": [],
            "task_action_reason_history": [],
            "is_conversational": classify_result,
        }

    async def execute_respond_conversation(
        state: AgentState, config: RunnableConfig
    ) -> dict:
        response = await response_conversation(
            llm=llm,
            query=state["query"],
            chat_history=state["messages"],
            config=config,
        )

        logger.info(f"\n\nConversational Response: {response}")

        return {"answer": response}

    def task_router_node(state: AgentState) -> dict:
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
            task_action_history=state["task_action_history"],
        )

        task_action_history = state["task_action_history"].copy()
        task_action_reason_history = state["task_action_reason_history"].copy()
        current_step = state["current_step"] if state["current_step"] else "Initial"
        task_action_history.append(f"{current_step} → {router_result.action}")
        task_action_reason_history.append(router_result.reasoning)

        logger.info(
            f"Action: {router_result.action}\nReason: {router_result.reasoning}"
        )

        return {
            "current_step": router_result.action,
            "task_action_history": task_action_history,
            "task_action_reason_history": task_action_reason_history,
        }

    def execute_rewrite_query(state: AgentState) -> dict:
        query = state["query"]
        rewritten = rewrite_query(
            llm=llm.with_structured_output(QueryWriterOutput),
            query=query,
            chat_history=state["messages"],
        )

        logger.info(f"Rewritten query from '{query}' to '{rewritten}'")

        return {"rewritten_query": rewritten}

    def execute_milvus_retrieve(state: AgentState) -> dict:
        context = milvus_retriever(
            query=state["rewritten_query"], embedding_model=embedding_model
        )
        if context is not None:
            formatted_context = "Local Knowledge Base Results:\n\n"
            for text, score in context:
                formatted_context += text + "\n\n"
            logger.info(formatted_context)
        else:
            logger.info("No data found in Local Knowledge Base.")
            formatted_context = ""

        return {"kb_context": formatted_context}

    def execute_tavily_search(state: AgentState) -> dict:
        search_results = tavily_search(
            query=TavilySearchInput(query=state["rewritten_query"])
        )
        formatted_search_results = f"Tavily Search Results:\n\nContent: {search_results.answer}\nSources: {search_results.sources}"

        logger.info(formatted_search_results)

        return {"web_context": formatted_search_results}

    def execute_craft_initial_answer(state: AgentState) -> dict:
        query = state["rewritten_query"]
        context = state["web_context"] if state["web_context"] else state["kb_context"]

        response = craft_initial_answer(
            llm=llm.with_structured_output(InitialAnswerCrafterOutput),
            query=query,
            context=context,
        )

        logger.info(f"Initial Answer:\n{response}")

        return {"answer": response}

    def execute_check_response(state: AgentState):
        check_result = check_response(
            llm=llm.with_structured_output(ResponseCheckerOutput),
            query=state["rewritten_query"],
            answer=state["answer"],
        )

        logger.info(f"Is the initial answer fully addressed the query? {check_result}.")

        return {"response_check": check_result}

    async def execute_craft_final_answer(
        state: AgentState, config: RunnableConfig
    ) -> dict:
        response = await craft_final_answer(
            llm=llm, answer=state["answer"], config=config
        )

        logger.info(f"\n\nFinal Answer:\n{response}")

        # Check action and reason history
        task_action_history = state["task_action_history"].copy()
        task_action_reason_history = state["task_action_reason_history"].copy()
        task_router_action_history = "\n\nTask router decisions:\n\n"
        for action, reason in zip(task_action_history, task_action_reason_history):
            task_router_action_history += f"Action: {action}\nReason: {reason}\n\n"
        logger.info(task_router_action_history)

        return {"answer": response}

    def initial_routing(
        state: AgentState,
    ) -> Literal["conversation_responder", "task_router"]:
        if state["is_conversational"].lower() == "true":
            return "conversation_responder"
        else:
            return "task_router"

    def get_next_task(
        state: AgentState,
    ) -> Literal[
        "query_rewriter",
        "milvus_retriever",
        "tavily_searcher",
        "initial_answer_crafter",
        "response_checker",
        "final_answer_crafter",
    ]:
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
            "tavily_searcher": "tavily_searcher",
            "initial_answer_crafter": "initial_answer_crafter",
            "response_checker": "response_checker",
            "final_answer_crafter": "final_answer_crafter",
        },
    )

    workflow.add_edge("query_rewriter", "task_router")
    workflow.add_edge("milvus_retriever", "task_router")
    workflow.add_edge("tavily_searcher", "task_router")
    workflow.add_edge("initial_answer_crafter", "task_router")
    workflow.add_edge("response_checker", "task_router")

    workflow.add_edge("conversation_responder", END)
    workflow.add_edge("final_answer_crafter", END)

    workflow.set_entry_point("query_classifier")

    app = workflow.compile(checkpointer=memory)

    return app


async def main():
    import uuid

    session_id = uuid.uuid4().hex[:8]
    config = {"configurable": {"thread_id": session_id}}

    embedding_model = SentenceTransformer(
        model_name_or_path=EMBEDDING_MODEL, device=device, trust_remote_code=True
    )

    graph = create_multi_agents(embedding_model=embedding_model)

    session_messages = []

    while True:

        query = input("Query: ").strip()

        if query.lower() == "q":
            logger.info(graph.get_state(config).values)
            break

        session_messages.append(HumanMessage(content=query))

        state = {
            "messages": session_messages,
            "query": query,
            "rewritten_query": "",
            "kb_context": "",
            "web_context": "",
            "answer": "",
            "response_check": "",
            "current_step": "",
            "task_action_history": [],
            "task_action_reason_history": [],
            "is_conversational": "",
        }

        full_response = ""
        async for msg, metadata in graph.astream(
            input=state, config=config, stream_mode="messages"
        ):
            if msg.content:
                full_response += msg.content
                print(msg.content, end="", flush=True)
        print()

        session_messages.append(AIMessage(content=full_response))


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
    #     with open("./assets/graph_visualization.png", "wb") as f:
    #         f.write(img.data)
    # except Exception:
    #     pass
