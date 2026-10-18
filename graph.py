"""Builds the LangGraph workflow."""
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from agents import (ClaimState, auto_approve_node, auto_reject_node, decision_engine_node,
                    document_verification_node, eligibility_check_node, human_approval_node,
                    intake_node, make_fraud_node, make_summary_node, route_after_intake,
                    route_after_summary)

PARALLEL = ["document_verification", "eligibility_check", "fraud_detection"]


def build_graph(llm=None, checkpointer=None):
    """llm=None runs the workflow with rules/templates only (no API calls)."""
    b = StateGraph(ClaimState)
    b.add_node("intake", intake_node)
    b.add_node("document_verification", document_verification_node)
    b.add_node("eligibility_check", eligibility_check_node)
    b.add_node("fraud_detection", make_fraud_node(llm))
    b.add_node("decision_engine", decision_engine_node)
    b.add_node("claim_summary", make_summary_node(llm))
    b.add_node("auto_approve", auto_approve_node)
    b.add_node("auto_reject", auto_reject_node)
    b.add_node("human_approval", human_approval_node)

    b.add_edge(START, "intake")
    # conditional edge 1: invalid claim -> reject, otherwise fan out to the 3 agents IN PARALLEL
    b.add_conditional_edges("intake", route_after_intake, PARALLEL + ["auto_reject"])
    # fan-in: decision engine waits for all three parallel agents
    b.add_edge(PARALLEL, "decision_engine")
    b.add_edge("decision_engine", "claim_summary")
    # conditional edge 2: route by decision
    b.add_conditional_edges("claim_summary", route_after_summary, {
        "AUTO_APPROVE": "auto_approve",
        "AUTO_REJECT": "auto_reject",
        "HUMAN_REVIEW": "human_approval",
    })
    b.add_edge("auto_approve", END)
    b.add_edge("auto_reject", END)
    b.add_edge("human_approval", END)
    return b.compile(checkpointer=checkpointer or MemorySaver())
