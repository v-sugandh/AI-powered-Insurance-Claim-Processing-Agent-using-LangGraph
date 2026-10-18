"""Insurance Claim Processing Agent - Streamlit app (LangGraph + Gemini)."""
import html
import os
import uuid
from contextlib import nullcontext
from datetime import date, timedelta

import pandas as pd
import streamlit as st
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from agents import inr
from graph import build_graph
from llm_factory import DEFAULT_LLM_MODEL, get_llm
from policies import (ALL_DOCUMENTS, CLAIM_TYPES, HIGH_VALUE_THRESHOLD, POLICIES, REQUIRED_DOCUMENTS,
                      SCENARIOS, make_claim)

st.set_page_config(page_title="ClaimFlow · AI Claims Agent", page_icon="🛡️", layout="wide",
                   initial_sidebar_state="expanded")

# ====================================================================== styling: dark command center
st.markdown(
    """
    <style>
    :root { --bg:#0b1020; --panel:#121a3a; --line:#26336b; --text:#e6ebff; --muted:#9aa6d1; --cyan:#00d4ff; --violet:#7c5cff; }
    .stApp { background: radial-gradient(1200px 600px at 85% -10%, #1a2358 0%, var(--bg) 55%); color: var(--text); }
    [data-testid="stHeader"] { background: transparent; }
    .block-container { padding-top: 1.6rem; max-width: 1320px; }
    h1, h2, h3, h4 { color: #fff; letter-spacing: -0.02em; }
    p, label, [data-testid="stCaptionContainer"] { color: var(--muted); }
    [data-testid="stSidebar"] { background: #0e1530; border-right: 1px solid var(--line); }
    [data-testid="stSidebar"] [role="radiogroup"] label { padding: .55rem .8rem; border-radius: 10px; margin-bottom: .2rem; width: 100%; }
    [data-testid="stSidebar"] [role="radiogroup"] label:hover { background: #18214a; }
    [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) { background: linear-gradient(90deg, rgba(0,212,255,.18), rgba(124,92,255,.18)); border: 1px solid var(--line); }
    [data-testid="stSidebar"] [role="radiogroup"] label > div:first-child { display: none; }
    div.stButton > button, div.stDownloadButton > button { border-radius: 10px; border: 1px solid var(--line); background: var(--panel); color: var(--text); font-weight: 650; min-height: 2.7rem; }
    div.stButton > button:hover { border-color: var(--cyan); color: var(--cyan); }
    div.stButton > button[kind="primary"], button[data-testid="stBaseButton-primary"] {
        background: linear-gradient(90deg, #00d4ff, #7c5cff); border: 0; color: #06101f; font-weight: 800; }
    div.stButton > button[kind="primary"]:hover, button[data-testid="stBaseButton-primary"]:hover { filter: brightness(1.12); color: #06101f; }
    [data-testid="stVerticalBlockBorderWrapper"] { background: rgba(18,26,58,.72); border-color: var(--line) !important; border-radius: 16px; }
    [data-testid="stExpander"] { background: rgba(18,26,58,.72); border: 1px solid var(--line); border-radius: 14px; }

    .banner { border: 1px solid var(--line); border-radius: 20px; padding: 2rem 2.2rem; margin-bottom: 1.4rem;
              background: linear-gradient(120deg, rgba(0,212,255,.14), rgba(124,92,255,.18)), var(--panel); }
    .banner .tag { color: var(--cyan); font-weight: 800; letter-spacing: .18em; font-size: .75rem; text-transform: uppercase; }
    .banner .ttl { font-size: clamp(1.8rem, 3.4vw, 2.7rem); font-weight: 800; color: #fff; line-height: 1.1; margin: .4rem 0 .5rem; }
    .banner .sub { color: var(--muted); max-width: 720px; }
    .page-head { margin-bottom: 1rem; }
    .page-head .tag { color: var(--cyan); font-weight: 800; letter-spacing: .16em; font-size: .72rem; text-transform: uppercase; }
    .page-head .ttl { font-size: 2rem; font-weight: 800; color: #fff; }
    .tile { background: var(--panel); border: 1px solid var(--line); border-top: 3px solid var(--a, var(--cyan)); border-radius: 14px; padding: 1rem 1.2rem; }
    .tile .k { color: var(--muted); font-size: .78rem; text-transform: uppercase; letter-spacing: .1em; font-weight: 700; }
    .tile .v { color: #fff; font-size: 1.9rem; font-weight: 800; line-height: 1.15; }
    .tile .s { color: var(--muted); font-size: .8rem; }
    .pipe { display: flex; flex-wrap: wrap; gap: .5rem; align-items: center; margin: .3rem 0 1.2rem; }
    .pipe .box { background: var(--panel); border: 1px solid var(--line); border-radius: 12px; padding: .6rem .9rem; color: #fff; font-weight: 650; font-size: .9rem; }
    .pipe .par { border-color: var(--cyan); box-shadow: 0 0 14px rgba(0,212,255,.25); }
    .pipe .arr { color: var(--cyan); font-weight: 800; }
    .timeline { border-left: 2px solid var(--line); margin-left: .5rem; padding-left: 1.1rem; }
    .tl-row { position: relative; padding-bottom: .9rem; color: #fff; }
    .tl-row::before { content: ""; position: absolute; left: -1.62rem; top: .3rem; width: .7rem; height: .7rem; border-radius: 50%;
                      background: var(--cyan); box-shadow: 0 0 10px var(--cyan); }
    .tl-row span { color: var(--muted); font-size: .85rem; }
    .chip { display: inline-block; padding: .2rem .7rem; margin: 0 .3rem .35rem 0; border-radius: 999px; font-size: .8rem; font-weight: 650; }
    .chip-ok { background: rgba(0,230,160,.14); color: #4af0b8; }
    .chip-miss { background: rgba(255,92,120,.15); color: #ff7a93; }
    .chip-neutral { background: #1b2550; color: #b8c3f0; }
    .pill { display: inline-block; padding: .28rem .8rem; border-radius: 999px; font-weight: 750; font-size: .85rem; }
    .pill-strong { background: rgba(0,230,160,.2); color: #4af0b8; }
    .pill-hire { background: rgba(0,230,160,.13); color: #4af0b8; }
    .pill-maybe { background: rgba(255,184,0,.16); color: #ffc83d; }
    .pill-no { background: rgba(255,92,120,.16); color: #ff7a93; }
    .lbl { color: var(--muted); font-size: .78rem; text-transform: uppercase; letter-spacing: .1em; font-weight: 700; margin-bottom: .3rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ====================================================================== constants
PAGES = {"dashboard": "📊  Dashboard", "claim": "📝  Submit claim", "workflow": "🔀  Claim result",
         "review": "🧑‍⚖️  Review queue", "scenarios": "🧪  Test scenarios"}
NODE_LABELS = {
    "intake": "Intake", "document_verification": "Document Verification Agent",
    "eligibility_check": "Eligibility Check Agent", "fraud_detection": "Fraud Detection Agent",
    "decision_engine": "Decision Engine", "claim_summary": "Claim Summary Agent",
    "auto_approve": "Auto Approve", "auto_reject": "Auto Reject", "human_approval": "Human Approval Agent",
}
DECISION_UI = {"AUTO_APPROVE": ("Auto-approve", "pill-hire"), "AUTO_REJECT": ("Auto-reject", "pill-no"),
               "HUMAN_REVIEW": ("Human review", "pill-maybe")}
STATUS_UI = {"APPROVED": ("Approved", "pill-strong"), "REJECTED": ("Rejected", "pill-no"),
             "PENDING": ("Awaiting human review", "pill-maybe")}
RISK_CLASS = {"Low": "pill-hire", "Medium": "pill-maybe", "High": "pill-no"}
CUSTOM = "✏️ Custom claim"
FORM_KEYS = ("f_policy", "f_name", "f_type", "f_incident", "f_amount", "f_desc", "f_docs", "preset")


# ====================================================================== helpers
def resolve_api_key() -> str:
    try:
        key = st.secrets.get("GOOGLE_API_KEY")
    except Exception:
        key = None
    return key or os.getenv("GOOGLE_API_KEY", "")


def go(page: str):
    st.session_state.goto = page


def pill(label: str, css: str) -> str:
    return f'<span class="pill {css}">{html.escape(label)}</span>'


def chip(text: str, kind: str) -> str:
    return f'<span class="chip chip-{kind}">{html.escape(str(text))}</span>'


def tile(label: str, value: str, sub: str = "", accent: str = "#00d4ff") -> str:
    return (f'<div class="tile" style="--a:{accent}"><div class="k">{html.escape(label)}</div>'
            f'<div class="v">{html.escape(str(value))}</div><div class="s">{html.escape(sub)}</div></div>')


def page_head(tag: str, title: str, sub: str = ""):
    st.markdown(f'<div class="page-head"><div class="tag">{tag}</div><div class="ttl">{title}</div></div>', unsafe_allow_html=True)
    if sub:
        st.caption(sub)


def timeline(log) -> str:
    rows = "".join(f'<div class="tl-row"><b>{html.escape(NODE_LABELS.get(e["node"], e["node"]))}</b><br>'
                   f'<span>{html.escape(e["message"])}</span></div>' for e in log)
    return f'<div class="timeline">{rows}</div>'


def apply_claim_to_form(c: dict):
    st.session_state.f_policy = c["policy_number"]
    st.session_state.f_name = c["claimant_name"]
    st.session_state.f_type = c["claim_type"]
    st.session_state.f_incident = date.today() - timedelta(days=c.get("incident_days_ago", 7))
    st.session_state.f_amount = float(c["amount"])
    st.session_state.f_desc = c["description"]
    st.session_state.f_docs = list(c["documents_provided"])


def load_preset():
    name = st.session_state.get("preset", CUSTOM)
    for s in SCENARIOS:
        if s["name"] == name:
            apply_claim_to_form(s["claim"])


def get_graph():
    cfg = (bool(use_llm and api_key), api_key if use_llm else "", model_name)
    if st.session_state.get("graph_cfg") != cfg:
        llm = get_llm(api_key, model_name) if cfg[0] else None
        saver = st.session_state.setdefault("saver", MemorySaver())
        st.session_state.graph = build_graph(llm, saver)
        st.session_state.graph_cfg = cfg
    return st.session_state.graph


def cfg_for(cid: str) -> dict:
    return {"configurable": {"thread_id": cid}}


def snapshot(cid: str):
    return get_graph().get_state(cfg_for(cid))


def status_of(snap) -> str:
    return "PENDING" if snap.next else snap.values.get("final_status", "-")


def pending_ids() -> list:
    return [cid for cid in st.session_state.get("claims", {}) if snapshot(cid).next]


def stream_graph(inp, cid: str, box=None):
    for chunk in get_graph().stream(inp, cfg_for(cid), stream_mode="updates"):
        for node in chunk:
            if box is not None:
                box.write("⏸️ Paused: waiting for a human reviewer" if node == "__interrupt__"
                          else f"✅ {NODE_LABELS.get(node, node)} finished")


def run_claim(claim: dict, label: str, expected=None, live: bool = True) -> str:
    cid = f"CLM-{uuid.uuid4().hex[:6].upper()}"
    st.session_state.setdefault("claims", {})[cid] = {"label": label, "expected": expected}
    ctx = st.status(f"Running agents for {cid}...", expanded=True) if live else nullcontext()
    with ctx as box:
        stream_graph({"claim": {**claim, "claim_id": cid}}, cid, box if live else None)
        if live:
            box.update(label=f"Claim {cid} processed", state="complete")
    return cid


def stats() -> dict:
    out = {"APPROVED": 0, "REJECTED": 0, "PENDING": 0}
    for cid in st.session_state.get("claims", {}):
        s = status_of(snapshot(cid))
        out[s] = out.get(s, 0) + 1
    return out


def graph_dot(done: set, pending) -> str:
    labels = {"intake": "Intake", "document_verification": "Document\\nVerification", "eligibility_check": "Eligibility\\nCheck",
              "fraud_detection": "Fraud\\nDetection", "decision_engine": "Decision\\nEngine", "claim_summary": "Claim\\nSummary",
              "auto_approve": "Auto\\nApprove", "auto_reject": "Auto\\nReject", "human_approval": "Human\\nApproval"}

    def node(n, label):
        if n == pending:
            return f'{n} [label="{label}", fillcolor="#6b4e00", color="#ffc83d", penwidth=2];'
        if n in done:
            return f'{n} [label="{label}", fillcolor="#0d5c4a", color="#4af0b8"];'
        return f'{n} [label="{label}"];'
    nodes = "\n".join(node(n, l) for n, l in labels.items())
    return f"""digraph G {{
      rankdir=LR; bgcolor="transparent"; nodesep=0.3; ranksep=0.45;
      node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11, color="#26336b",
            fillcolor="#121a3a", fontcolor="#e6ebff", margin="0.14,0.07"];
      edge [color="#5a6aa8", arrowsize=0.7, fontname="Helvetica", fontsize=9, fontcolor="#9aa6d1"];
      {nodes}
      intake -> document_verification; intake -> eligibility_check; intake -> fraud_detection;
      intake -> auto_reject [style=dashed, label="invalid"];
      document_verification -> decision_engine; eligibility_check -> decision_engine; fraud_detection -> decision_engine;
      decision_engine -> claim_summary;
      claim_summary -> auto_approve [label="approve"]; claim_summary -> human_approval [label="review"];
      claim_summary -> auto_reject [label="reject"];
    }}"""


# ====================================================================== pages
def page_dashboard():
    st.markdown(
        '<div class="banner"><div class="tag">ClaimFlow · LangGraph agents</div>'
        '<div class="ttl">Claims decided in seconds,<br>risky ones escalated to people.</div>'
        '<div class="sub">Documents, eligibility and fraud are checked in parallel. Clean claims are decided automatically; '
        "anything uncertain or high-value waits for a human reviewer.</div></div>",
        unsafe_allow_html=True,
    )
    s = stats()
    k = st.columns(4)
    k[0].markdown(tile("Claims processed", len(st.session_state.get("claims", {})), "this session"), unsafe_allow_html=True)
    k[1].markdown(tile("Approved", s.get("APPROVED", 0), "auto + human", "#4af0b8"), unsafe_allow_html=True)
    k[2].markdown(tile("Rejected", s.get("REJECTED", 0), "auto + human", "#ff7a93"), unsafe_allow_html=True)
    k[3].markdown(tile("Awaiting review", s.get("PENDING", 0), "human-in-the-loop", "#ffc83d"), unsafe_allow_html=True)

    st.markdown('<div class="lbl" style="margin-top:1.4rem">Agent pipeline</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="pipe"><div class="box">Intake</div><span class="arr">→</span>'
        '<div class="box par">Documents</div><div class="box par">Eligibility</div><div class="box par">Fraud</div>'
        '<span class="arr">→</span><div class="box">Decision</div><span class="arr">→</span><div class="box">Summary</div>'
        '<span class="arr">→</span><div class="box">Approve · Reject · Human review</div></div>',
        unsafe_allow_html=True)
    st.caption("Glowing boxes run in parallel.")

    c1, c2 = st.columns(2, gap="large")
    with c1.container(border=True):
        st.markdown("#### How a claim is routed")
        st.markdown(pill("Auto-approve", "pill-hire") + f" documents complete, policy valid, low fraud risk, under {inr(HIGH_VALUE_THRESHOLD)}", unsafe_allow_html=True)
        st.markdown(pill("Auto-reject", "pill-no") + " documents missing, or policy expired / not found / not covering / not the holder", unsafe_allow_html=True)
        st.markdown(pill("Human review", "pill-maybe") + f" fraud risk medium/high or LLM concern, {inr(HIGH_VALUE_THRESHOLD)}+ claims, or above coverage limit", unsafe_allow_html=True)
    with c2.container(border=True):
        st.markdown("#### Get started")
        st.button("📝 Submit a claim", type="primary", on_click=go, args=("claim",), key="d_claim")
        st.button("🧪 Run the 5 test scenarios", on_click=go, args=("scenarios",), key="d_scen")
        st.markdown("".join(chip(t, "neutral") for t in ["LangGraph", "Parallel agents", "Conditional edges", "Human-in-the-loop", "Gemini"]),
                    unsafe_allow_html=True)


def page_claim():
    page_head("Intake", "Submit a claim", "Load a ready-made scenario or enter your own details.")
    st.selectbox("Quick-load a test scenario", [CUSTOM] + [s["name"] for s in SCENARIOS], key="preset", on_change=load_preset)

    with st.container(border=True):
        a, b, c = st.columns(3, gap="large")
        a.text_input("Policy number", key="f_policy", placeholder="e.g. POL-1001")
        b.text_input("Claimant name", key="f_name")
        c.selectbox("Claim type", CLAIM_TYPES, key="f_type")
        d, e = st.columns(2, gap="large")
        d.date_input("Incident date", key="f_incident", max_value=date.today())
        e.number_input("Claim amount (₹)", min_value=0.0, step=1000.0, key="f_amount")
        st.text_area("What happened?", key="f_desc", height=100)
        st.multiselect("Documents provided", ALL_DOCUMENTS, key="f_docs")
        st.caption(f"Required for {st.session_state.f_type}: " + ", ".join(REQUIRED_DOCUMENTS[st.session_state.f_type]))

    with st.expander("Known demo policies"):
        st.dataframe(pd.DataFrame([{"Policy": k, "Holder": v["holder"], "Covers": ", ".join(v["covers"]),
                                    "Valid until": v["end"], "Coverage limit": inr(v["coverage_limit"])}
                                   for k, v in POLICIES.items()]), hide_index=True)

    if st.button("🚀 Run agents", type="primary", key="run_claim"):
        claim = {
            "policy_number": st.session_state.f_policy, "claimant_name": st.session_state.f_name,
            "claim_type": st.session_state.f_type, "incident_date": st.session_state.f_incident.isoformat(),
            "claim_date": date.today().isoformat(), "amount": st.session_state.f_amount,
            "description": st.session_state.f_desc, "documents_provided": st.session_state.f_docs,
        }
        label = st.session_state.preset if st.session_state.preset != CUSTOM else "Custom claim"
        cid = run_claim(claim, label)
        st.session_state.current = cid
        st.session_state.goto = "workflow"
        st.rerun()


def page_workflow():
    cid = st.session_state.get("current")
    if not cid or cid not in st.session_state.get("claims", {}):
        page_head("Nothing yet", "No claim processed")
        st.button("📝 Submit a claim", type="primary", on_click=go, args=("claim",), key="wf_empty")
        return

    snap = snapshot(cid)
    v = snap.values
    claim = v["claim"]
    status = status_of(snap)
    page_head(f"Claim {cid}", st.session_state.claims[cid]["label"])
    st.markdown(pill(*STATUS_UI.get(status, (status, "pill-maybe"))) + f' &nbsp; <span style="color:#9aa6d1">{html.escape(v.get("decided_by", "Awaiting reviewer"))}</span>',
                unsafe_allow_html=True)

    fr = v.get("fraud_result")
    k = st.columns(4)
    k[0].markdown(tile("Claim amount", inr(claim["amount"]), f"{claim['claim_type']} · {claim['policy_number']}"), unsafe_allow_html=True)
    k[1].markdown(tile("Payable", inr(v.get("payable_amount", 0)) if status != "PENDING" else "Pending", "after decision", "#4af0b8"), unsafe_allow_html=True)
    k[2].markdown(tile("Fraud risk", fr["risk_level"] if fr else "-", f"rule score {fr['rule_score']}" if fr else "", "#ffc83d"), unsafe_allow_html=True)
    k[3].markdown(tile("Routing", DECISION_UI[v["decision"]][0] if "decision" in v else "-", "decision engine", "#7c5cff"), unsafe_allow_html=True)
    st.markdown("&nbsp;", unsafe_allow_html=True)

    left, right = st.columns([1, 1.5], gap="large")
    with left.container(border=True):
        st.markdown("#### Execution timeline")
        st.markdown(timeline(v.get("audit_log", [])), unsafe_allow_html=True)
    with right.container(border=True):
        st.markdown("#### Workflow path")
        st.graphviz_chart(graph_dot({e["node"] for e in v.get("audit_log", [])}, snap.next[0] if snap.next else None))
        st.caption("Green = executed, amber = waiting for a human. The three checks run in parallel.")

    if "doc_result" in v:
        doc, el = v["doc_result"], v["eligibility_result"]
        c1, c2, c3 = st.columns(3, gap="medium")
        with c1.container(border=True):
            st.markdown("#### 📄 Documents")
            st.markdown(pill("Complete" if doc["ok"] else "Incomplete", "pill-hire" if doc["ok"] else "pill-no"), unsafe_allow_html=True)
            st.markdown("".join(chip(("✓ " if d in doc["provided"] else "✕ ") + d, "ok" if d in doc["provided"] else "miss")
                                for d in doc["required"]), unsafe_allow_html=True)
        with c2.container(border=True):
            st.markdown("#### 🧾 Eligibility")
            st.markdown(pill("Eligible" if el["eligible"] else "Not eligible", "pill-hire" if el["eligible"] else "pill-no"), unsafe_allow_html=True)
            for ch in el["checks"]:
                st.markdown(f"{'✅' if ch['passed'] else '❌'} {ch['detail']}")
        with c3.container(border=True):
            st.markdown("#### 🕵️ Fraud check")
            st.markdown(pill(f"{fr['risk_level']} risk · {fr['rule_score']}", RISK_CLASS[fr["risk_level"]]), unsafe_allow_html=True)
            for i in fr["indicators"]:
                st.markdown(f"⚠️ **{i['indicator']}**: {i['detail']}")
            if not fr["indicators"]:
                st.markdown("No rule-based indicators.")
            if fr["llm_used"]:
                st.markdown(f"🤖 **LLM:** {'concern raised' if fr['llm_concern'] else 'no concern'}. {fr['llm_reasoning']}")
                for f in fr["llm_findings"]:
                    st.markdown(f"- {f}")
            elif fr["llm_error"]:
                st.caption("LLM unavailable, rules only: " + fr["llm_error"])
            else:
                st.caption("Rules only (LLM off).")

    if "decision" in v:
        with st.container(border=True):
            st.markdown("#### Decision & summary")
            st.markdown("Routing: " + pill(*DECISION_UI[v["decision"]]), unsafe_allow_html=True)
            for r in v["decision_reasons"]:
                st.markdown(f"- {r}")
            if v.get("summary"):
                st.write(v["summary"])
            if v.get("human_decision", {}).get("notes"):
                st.info(f"Reviewer notes: {v['human_decision']['notes']}")
    if snap.next:
        st.warning("This claim is paused and needs a human decision.")
        st.button("🧑‍⚖️ Open review queue", type="primary", on_click=go, args=("review",), key="wf_to_review")


def page_review():
    pend = pending_ids()
    page_head("Human-in-the-loop", "Review queue", "Escalated claims wait here until a reviewer approves or rejects them.")
    if not pend:
        st.success("Queue is empty: no claims are waiting for review.")
        return
    for cid in pend:
        v = snapshot(cid).values
        claim = v["claim"]
        with st.container(border=True):
            h1, h2 = st.columns([3, 1])
            h1.markdown(f"### {cid} · {claim['claimant_name']}")
            h2.markdown(tile("Amount", inr(claim["amount"]), claim["claim_type"], "#ffc83d"), unsafe_allow_html=True)
            st.markdown("**Why it was escalated**")
            for r in v["decision_reasons"]:
                st.markdown(f"- {r}")
            st.write(v.get("summary", ""))
            st.caption(f"Incident ({claim['incident_date']}): {claim['description']}")
            c1, c2 = st.columns([1, 2])
            reviewer = c1.text_input("Reviewer name", key=f"rev_{cid}")
            notes = c2.text_input("Notes (optional)", key=f"notes_{cid}")
            b1, b2, _ = st.columns([1, 1, 3])
            decision = None
            if b1.button("✅ Approve", type="primary", key=f"appr_{cid}"):
                decision = "approve"
            if b2.button("⛔ Reject", key=f"rej_{cid}"):
                decision = "reject"
            if decision:
                with st.spinner("Resuming workflow..."):
                    stream_graph(Command(resume={"action": decision, "reviewer": reviewer, "notes": notes}), cid)
                st.session_state.current = cid
                st.session_state.goto = "workflow"
                st.rerun()


def page_scenarios():
    page_head("Assignment", "Test scenarios", "The five assignment scenarios, run end to end through the graph.")
    for col, s in zip(st.columns(len(SCENARIOS), gap="small"), SCENARIOS):
        col.markdown(tile(s["name"].split(". ")[0] + " · expected", DECISION_UI[s["expected"]][0], s["name"].split(". ", 1)[1], "#7c5cff"),
                     unsafe_allow_html=True)
    st.markdown("&nbsp;", unsafe_allow_html=True)

    if st.button("▶ Run all 5 scenarios", type="primary", key="run_all"):
        with st.status("Running scenarios...", expanded=True) as box:
            for s in SCENARIOS:
                box.write(f"Processing {s['name']}")
                cid = run_claim(make_claim(s["claim"]), s["name"], expected=s["expected"], live=False)
                st.session_state.scen_runs = {**st.session_state.get("scen_runs", {}), s["name"]: cid}
            box.update(label="All scenarios processed", state="complete")

    runs = st.session_state.get("scen_runs", {})
    if runs:
        passed = 0
        for s in SCENARIOS:
            cid = runs.get(s["name"])
            if not cid or cid not in st.session_state.get("claims", {}):
                continue
            snap = snapshot(cid)
            actual = snap.values.get("decision", "-")
            ok = actual == s["expected"]
            passed += ok
            with st.container(border=True):
                a, b, c, d, e = st.columns([3, 1.5, 1.5, 1.6, 1])
                a.markdown(f"**{s['name']}**  \n<span style='color:#9aa6d1;font-size:.85rem'>{cid} · {inr(snap.values['claim']['amount'])}</span>", unsafe_allow_html=True)
                b.markdown('<div class="lbl">Expected</div>' + pill(*DECISION_UI[s["expected"]]), unsafe_allow_html=True)
                c.markdown('<div class="lbl">Actual</div>' + pill(*DECISION_UI.get(actual, (actual, "pill-maybe"))), unsafe_allow_html=True)
                d.markdown('<div class="lbl">Status</div>' + pill(*STATUS_UI.get(status_of(snap), ("-", "pill-maybe"))), unsafe_allow_html=True)
                e.button("Open", key=f"open_{cid}", on_click=lambda c=cid: (st.session_state.update(current=c), go("workflow")))
        st.success(f"{passed} of {len(SCENARIOS)} scenarios routed as expected.")


# ====================================================================== sidebar + routing
st.session_state.setdefault("page", "dashboard")
if "goto" in st.session_state:  # navigation requests are applied before the radio is drawn
    st.session_state.page = st.session_state.pop("goto")
if "f_policy" not in st.session_state:
    apply_claim_to_form(SCENARIOS[0]["claim"])
for _k in FORM_KEYS:  # keep form state alive while another page is shown
    if _k in st.session_state:
        st.session_state[_k] = st.session_state[_k]


def reset_session():
    for k in ("claims", "current", "saver", "graph", "graph_cfg", "scen_runs"):
        st.session_state.pop(k, None)
    go("dashboard")


with st.sidebar:
    st.markdown("## 🛡️ ClaimFlow")
    st.caption("AI claims agent · LangGraph")
    st.radio("Navigate", list(PAGES.keys()), format_func=lambda k: PAGES[k], key="page", label_visibility="collapsed")
    st.divider()
    st.markdown("**⚙️ Settings**")
    api_key = resolve_api_key()
    if not api_key:
        api_key = st.text_input("Google API key", type="password", help="Get one free at aistudio.google.com")
    else:
        st.success("API key loaded")
    model_name = st.text_input("Gemini model", value=DEFAULT_LLM_MODEL)
    use_llm = st.toggle("Use Gemini (fraud review + summaries)", value=True, help="Off = rules and templates only")
    st.caption("Mode: LLM-assisted" if (use_llm and api_key) else "Mode: rules only" + (" (add an API key for the LLM)" if use_llm else ""))
    if st.session_state.get("claims"):
        st.divider()
        n_pend = len(pending_ids())
        st.caption(f"{len(st.session_state.claims)} claim(s) processed · {n_pend} awaiting review")
        st.button("↺ Reset session", key="reset", on_click=reset_session)

{"dashboard": page_dashboard, "claim": page_claim, "workflow": page_workflow,
 "review": page_review, "scenarios": page_scenarios}[st.session_state.page]()