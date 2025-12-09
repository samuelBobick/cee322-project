"""Agent creation and compilation"""

from langgraph.graph import StateGraph

from .state import DataEngState
from .cleaning import clean_node
from .imputation import impute_node
from .reflection import (
    reflect_clean_node,
    reflect_impute_node,
    route_after_clean_reflection,
    route_after_impute_reflection,
)


def create_agent(chat_model):
    """
    Create and compile the clean-impute agent with reflection nodes.

    Flow: clean -> reflect_clean -> (clean if failed | impute if passed)
          -> reflect_impute -> (impute if failed | end if passed)

    Args:
        chat_model: The LLM chat model to use for assumptions and reflection

    Returns:
        Compiled LangGraph agent
    """
    workflow = StateGraph(DataEngState)

    def clean_wrapper(state: DataEngState) -> DataEngState:
        return clean_node(state, chat_model)

    def impute_wrapper(state: DataEngState) -> DataEngState:
        return impute_node(state, chat_model)

    def reflect_clean_wrapper(state: DataEngState) -> DataEngState:
        return reflect_clean_node(state, chat_model)

    def reflect_impute_wrapper(state: DataEngState) -> DataEngState:
        return reflect_impute_node(state, chat_model)

    # Add nodes
    workflow.add_node("clean", clean_wrapper)
    workflow.add_node("reflect_clean", reflect_clean_wrapper)
    workflow.add_node("impute", impute_wrapper)
    workflow.add_node("reflect_impute", reflect_impute_wrapper)
    workflow.set_entry_point("clean")
    workflow.add_edge("clean", "reflect_clean")
    workflow.add_edge("impute", "reflect_impute")

    # Conditional edge after clean reflection: go back to clean if failed, else proceed to impute
    workflow.add_conditional_edges(
        "reflect_clean",
        route_after_clean_reflection,
        {
            "clean": "clean",  # Re-clean if reflection failed
            "impute": "impute",  # Proceed to impute if reflection passed
        },
    )

    # Same for reflect_clean
    workflow.add_conditional_edges(
        "reflect_impute",
        route_after_impute_reflection,
        {
            "impute": "impute",  # Re-impute if reflection failed
            "end": "__end__",  # End if reflection passed
        },
    )

    # Compile the agent
    agent = workflow.compile()
    print("4-node agent (clean -> reflect_clean -> impute -> reflect_impute) compiled")

    return agent
