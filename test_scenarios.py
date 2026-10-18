"""Runs the 5 assignment scenarios (rules-only, no API key needed) plus the human-in-the-loop resume."""
import uuid

from langgraph.types import Command

from graph import build_graph
from policies import SCENARIOS, make_claim

graph = build_graph(llm=None)


def run(claim):
    cid = f"CLM-{uuid.uuid4().hex[:6].upper()}"
    cfg = {"configurable": {"thread_id": cid}}
    list(graph.stream({"claim": {**claim, "claim_id": cid}}, cfg, stream_mode="updates"))
    return cfg, graph.get_state(cfg)


if __name__ == "__main__":
    pending = []
    for s in SCENARIOS:
        cfg, snap = run(make_claim(s["claim"]))
        actual = snap.values["decision"]
        status = "PENDING_HUMAN" if snap.next else snap.values["final_status"]
        ok = actual == s["expected"]
        print(f"{'PASS' if ok else 'FAIL'}  {s['name']:<42} expected={s['expected']:<13} actual={actual:<13} status={status}")
        for r in snap.values["decision_reasons"]:
            print("        -", r)
        assert ok, s["name"]
        if snap.next:
            pending.append(cfg)

    # human-in-the-loop: resume the paused claims
    assert len(pending) == 2
    list(graph.stream(Command(resume={"action": "approve", "reviewer": "Asha", "notes": "Documents verified"}),
                      pending[0], stream_mode="updates"))
    s0 = graph.get_state(pending[0]).values
    assert s0["final_status"] == "APPROVED" and "Asha" in s0["decided_by"], s0
    list(graph.stream(Command(resume={"action": "reject", "reviewer": "Asha", "notes": "Inflated"}),
                      pending[1], stream_mode="updates"))
    s1 = graph.get_state(pending[1]).values
    assert s1["final_status"] == "REJECTED" and s1["payable_amount"] == 0.0, s1

    # invalid input takes the early-exit conditional edge
    bad = make_claim(SCENARIOS[0]["claim"]); bad["amount"] = 0
    _, snap = run(bad)
    assert snap.values["decision"] == "AUTO_REJECT" and snap.values["final_status"] == "REJECTED"
    print("\nALL CHECKS PASSED")
