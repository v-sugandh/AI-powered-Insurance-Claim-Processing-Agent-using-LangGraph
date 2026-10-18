"""Graph state and the agent nodes for the insurance claim workflow.

Agents:  Document Verification, Eligibility Check, Fraud Detection (run in parallel),
         Decision Engine, Claim Summary, Human Approval (+ auto approve / auto reject finalisers).
The LLM (Gemini) powers the Fraud Detection reasoning and the Claim Summary. If no LLM is
available, both fall back to deterministic rules/templates so the workflow still runs.
"""
import json
import operator
import time
from datetime import date
from typing import Annotated, Any, Dict, List, TypedDict

from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.prompts import PromptTemplate
from langgraph.types import interrupt

from models import FraudAnalysis
from policies import (CLAIM_TYPES, HIGH_VALUE_THRESHOLD, MINOR_KEYWORDS, POLICIES,
                      REQUIRED_DOCUMENTS, TYPICAL_AMOUNT)


class ClaimState(TypedDict, total=False):
    claim: Dict[str, Any]
    intake_valid: bool
    doc_result: Dict[str, Any]          # written by Document Verification
    eligibility_result: Dict[str, Any]  # written by Eligibility Check
    fraud_result: Dict[str, Any]        # written by Fraud Detection (3 agents write different keys -> safe in parallel)
    decision: str                       # AUTO_APPROVE | AUTO_REJECT | HUMAN_REVIEW
    decision_reasons: List[str]
    summary: str
    human_decision: Dict[str, Any]
    final_status: str                   # APPROVED | REJECTED
    decided_by: str
    payable_amount: float
    audit_log: Annotated[List[Dict[str, str]], operator.add]  # reducer: parallel nodes can all append


def _log(node: str, message: str) -> Dict[str, str]:
    return {"node": node, "message": message}


def inr(x) -> str:
    return f"₹{float(x):,.0f}"


def _invoke(chain, inputs: dict, retries: int = 2):
    for attempt in range(retries):
        try:
            return chain.invoke(inputs)
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2)


# ------------------------------------------------------------------ 1. intake
def intake_node(state: ClaimState):
    claim = dict(state["claim"])
    errors = []
    if not str(claim.get("policy_number", "")).strip():
        errors.append("Policy number is missing")
    if not str(claim.get("claimant_name", "")).strip():
        errors.append("Claimant name is missing")
    if claim.get("claim_type") not in CLAIM_TYPES:
        errors.append("Claim type is invalid")
    try:
        if float(claim.get("amount", 0)) <= 0:
            errors.append("Claim amount must be greater than zero")
    except (TypeError, ValueError):
        errors.append("Claim amount is not a number")
    try:
        if date.fromisoformat(claim.get("incident_date", "")) > date.today():
            errors.append("Incident date is in the future")
    except ValueError:
        errors.append("Incident date is invalid")
    if not str(claim.get("description", "")).strip():
        errors.append("Incident description is missing")

    if errors:
        return {"claim": claim, "intake_valid": False, "decision": "AUTO_REJECT",
                "decision_reasons": ["Invalid claim submission: " + "; ".join(errors)],
                "audit_log": [_log("intake", "Claim failed validation: " + "; ".join(errors))]}
    return {"claim": claim, "intake_valid": True,
            "audit_log": [_log("intake", "Claim validated, starting parallel checks")]}


def route_after_intake(state: ClaimState):
    """Conditional edge #1: invalid claims skip straight to rejection; valid ones fan out in parallel."""
    if state.get("intake_valid"):
        return ["document_verification", "eligibility_check", "fraud_detection"]
    return "auto_reject"


# ------------------------------------------------------------------ 2a. document verification
def document_verification_node(state: ClaimState):
    claim = state["claim"]
    required = REQUIRED_DOCUMENTS.get(claim["claim_type"], [])
    provided = {d.strip().lower() for d in claim.get("documents_provided", [])}
    missing = [d for d in required if d.lower() not in provided]
    result = {"ok": not missing, "required": required,
              "provided": claim.get("documents_provided", []), "missing": missing}
    msg = "All required documents present" if not missing else "Missing documents: " + ", ".join(missing)
    return {"doc_result": result, "audit_log": [_log("document_verification", msg)]}


# ------------------------------------------------------------------ 2b. eligibility
def eligibility_check_node(state: ClaimState):
    claim = state["claim"]
    pn = str(claim["policy_number"]).strip().upper()
    policy = POLICIES.get(pn)
    amount = float(claim["amount"])

    if policy is None:
        result = {"eligible": False, "policy_found": False, "coverage_limit": None, "exceeds_limit": False,
                  "checks": [{"check": "Policy exists", "passed": False, "detail": f"No policy {pn} on record"}],
                  "hard_failures": [f"Policy {pn} not found"]}
        return {"eligibility_result": result, "audit_log": [_log("eligibility_check", result["hard_failures"][0])]}

    incident = date.fromisoformat(claim["incident_date"])
    start, end = date.fromisoformat(policy["start"]), date.fromisoformat(policy["end"])
    checks, failures = [], []

    def add(name, passed, detail, hard=True):
        checks.append({"check": name, "passed": passed, "detail": detail})
        if hard and not passed:
            failures.append(detail)

    add("Policy exists", True, f"Policy {pn} found")
    same_name = claim["claimant_name"].strip().lower() == policy["holder"].lower()
    add("Policyholder matches", same_name, "Claimant matches policyholder" if same_name
        else f"Claimant does not match policyholder on {pn}")
    if incident < start:
        add("Policy active on incident date", False, f"Policy was not yet active on the incident date ({incident})")
    elif incident > end:
        add("Policy active on incident date", False, f"Policy expired on {end}, before the incident on {incident}")
    else:
        add("Policy active on incident date", True, f"Incident {incident} is within {start} to {end}")
    covered = claim["claim_type"] in policy["covers"]
    add("Claim type covered", covered, f"{claim['claim_type']} claims are covered" if covered
        else f"{claim['claim_type']} claims are not covered by this policy")
    exceeds = amount > policy["coverage_limit"]
    add("Within coverage limit", not exceeds,
        f"{inr(amount)} is within the limit of {inr(policy['coverage_limit'])}" if not exceeds
        else f"{inr(amount)} exceeds the coverage limit of {inr(policy['coverage_limit'])}", hard=False)

    result = {"eligible": not failures, "policy_found": True, "coverage_limit": policy["coverage_limit"],
              "exceeds_limit": exceeds, "checks": checks, "hard_failures": failures}
    msg = "Eligible" if not failures else "Not eligible: " + "; ".join(failures)
    return {"eligibility_result": result, "audit_log": [_log("eligibility_check", msg)]}


# ------------------------------------------------------------------ 2c. fraud detection
FRAUD_TEMPLATE = """You are an insurance fraud analyst reviewing a claim. Below are the claim and the indicators
already found by rule-based checks.

Set concern=true ONLY for a concrete red flag, for example: the described damage does not justify the amount,
details are contradictory or implausible, or the wording suggests a staged or inflated claim.
A large but plausible claim (for example major surgery) is NOT a concern by itself.
If nothing stands out, set concern=false and leave findings empty. Use only the information given.

CLAIM
{claim}

TYPICAL CLAIM AMOUNT FOR {claim_type}: {typical}

RULE-BASED INDICATORS
{indicators}

{format_instructions}
"""
fraud_parser = PydanticOutputParser(pydantic_object=FraudAnalysis)
FRAUD_PROMPT = PromptTemplate(
    template=FRAUD_TEMPLATE,
    input_variables=["claim", "claim_type", "typical", "indicators"],
    partial_variables={"format_instructions": fraud_parser.get_format_instructions()},
)


def make_fraud_node(llm):
    def fraud_detection_node(state: ClaimState):
        claim = state["claim"]
        ctype = claim["claim_type"]
        amount = float(claim["amount"])
        typical = TYPICAL_AMOUNT.get(ctype, 100_000)
        desc = claim["description"].strip()
        policy = POLICIES.get(str(claim["policy_number"]).strip().upper())
        incident = date.fromisoformat(claim["incident_date"])
        filed = date.fromisoformat(claim.get("claim_date", date.today().isoformat()))

        indicators: List[Dict[str, Any]] = []

        def add(name, points, detail):
            indicators.append({"indicator": name, "points": points, "detail": detail})

        if any(k in desc.lower() for k in MINOR_KEYWORDS) and amount > 1.5 * typical:
            add("Amount inconsistent with described damage", 40,
                f"Description suggests minor damage but the claim is {inr(amount)} (typical {ctype} claim: {inr(typical)})")
        if amount > 10 * typical:
            add("Amount far above typical", 30, f"{inr(amount)} is more than 10x the typical {ctype} claim")
        if policy and amount >= 0.9 * policy["coverage_limit"]:
            add("Claim close to coverage limit", 20, f"{inr(amount)} is at least 90% of the {inr(policy['coverage_limit'])} limit")
        if policy:
            days_since_start = (incident - date.fromisoformat(policy["start"])).days
            if 0 <= days_since_start < 30:
                add("Loss soon after policy start", 25, f"Incident only {days_since_start} days after the policy began")
            if policy["prior_claims_12m"] >= 2:
                add("Multiple recent claims", 20, f"{policy['prior_claims_12m']} claims in the last 12 months")
        if (filed - incident).days > 60:
            add("Late filing", 10, f"Claim filed {(filed - incident).days} days after the incident")
        if len(desc) < 25:
            add("Vague description", 15, "Incident description is very short")

        score = min(100, sum(i["points"] for i in indicators))
        level = "High" if score >= 60 else "Medium" if score >= 30 else "Low"
        result = {"rule_score": score, "risk_level": level, "indicators": indicators, "llm_used": False,
                  "llm_concern": False, "llm_findings": [], "llm_reasoning": "", "llm_error": ""}

        if llm is not None:
            try:
                analysis: FraudAnalysis = _invoke(
                    FRAUD_PROMPT | llm | fraud_parser,
                    {"claim": json.dumps({k: v for k, v in claim.items() if k != "documents_provided"}, indent=2),
                     "claim_type": ctype, "typical": inr(typical),
                     "indicators": json.dumps(indicators, indent=2) if indicators else "None"},
                )
                result.update(llm_used=True, llm_concern=bool(analysis.concern),
                              llm_findings=analysis.findings, llm_reasoning=analysis.reasoning)
            except Exception as e:  # never let an LLM hiccup crash the claim
                result["llm_error"] = str(e)[:200]

        result["needs_review"] = level in ("Medium", "High") or result["llm_concern"]
        suffix = " (LLM flagged a concern)" if result["llm_concern"] else (
            " (LLM unavailable, rules only)" if llm is not None and not result["llm_used"] else "")
        return {"fraud_result": result,
                "audit_log": [_log("fraud_detection", f"Risk {level} (rule score {score}){suffix}")]}
    return fraud_detection_node


# ------------------------------------------------------------------ 3. decision engine
def decision_engine_node(state: ClaimState):
    doc, el, fr = state["doc_result"], state["eligibility_result"], state["fraud_result"]
    amount = float(state["claim"]["amount"])

    reject = list(el["hard_failures"])
    if not doc["ok"]:
        reject.append("Missing required documents: " + ", ".join(doc["missing"]))
    if reject:
        decision, reasons = "AUTO_REJECT", reject
    else:
        review = []
        if fr["needs_review"]:
            why = [i["indicator"] for i in fr["indicators"]] + fr["llm_findings"]
            review.append(f"Fraud risk {fr['risk_level']}" + (": " + "; ".join(why) if why else " (flagged by LLM review)"))
        if amount >= HIGH_VALUE_THRESHOLD:
            review.append(f"High-value claim ({inr(amount)} is at or above {inr(HIGH_VALUE_THRESHOLD)})")
        if el["exceeds_limit"]:
            review.append(f"Amount exceeds the coverage limit of {inr(el['coverage_limit'])}")
        if review:
            decision, reasons = "HUMAN_REVIEW", review
        else:
            decision = "AUTO_APPROVE"
            reasons = ["All required documents verified", "Policy active and claim type covered",
                       f"Low fraud risk and amount below {inr(HIGH_VALUE_THRESHOLD)}"]
    return {"decision": decision, "decision_reasons": reasons,
            "audit_log": [_log("decision_engine", f"Decision: {decision}")]}


# ------------------------------------------------------------------ 4. claim summary
SUMMARY_TEMPLATE = """Write a concise claim summary for an insurance adjuster, in plain prose of at most 120 words
and with no headings or bullet points. Cover: who is claiming what and how much, document status,
policy eligibility, fraud risk, and the recommended decision with the reasons. Use only the data below.

DATA
{data}
"""
SUMMARY_PROMPT = PromptTemplate(template=SUMMARY_TEMPLATE, input_variables=["data"])


def _template_summary(state: ClaimState) -> str:
    c, doc, el, fr = state["claim"], state["doc_result"], state["eligibility_result"], state["fraud_result"]
    return (f"{c['claimant_name']} filed a {c['claim_type']} claim of {inr(c['amount'])} under {c['policy_number']}. "
            f"Documents: {'complete' if doc['ok'] else 'missing ' + ', '.join(doc['missing'])}. "
            f"Eligibility: {'eligible' if el['eligible'] else '; '.join(el['hard_failures'])}. "
            f"Fraud risk: {fr['risk_level']}. Recommended decision: {state['decision'].replace('_', ' ').title()} "
            f"({'; '.join(state['decision_reasons'])}).")


def make_summary_node(llm):
    def claim_summary_node(state: ClaimState):
        data = {"claim": {k: v for k, v in state["claim"].items() if k != "documents_provided"},
                "documents": state["doc_result"], "eligibility": state["eligibility_result"],
                "fraud": state["fraud_result"], "decision": state["decision"],
                "reasons": state["decision_reasons"]}
        summary, source = None, "template"
        if llm is not None:
            try:
                summary = _invoke(SUMMARY_PROMPT | llm | StrOutputParser(),
                                  {"data": json.dumps(data, indent=2, default=str)}).strip()
                source = "LLM"
            except Exception:
                summary = None
        if not summary:
            summary = _template_summary(state)
        return {"summary": summary, "audit_log": [_log("claim_summary", f"Summary generated ({source})")]}
    return claim_summary_node


def route_after_summary(state: ClaimState) -> str:
    """Conditional edge #2: route by the decision."""
    return state["decision"]


# ------------------------------------------------------------------ 5. finalisers
def _payable(state: ClaimState) -> float:
    claim = state["claim"]
    policy = POLICIES.get(str(claim["policy_number"]).strip().upper(), {})
    return min(float(claim["amount"]), policy.get("coverage_limit", float(claim["amount"])))


def auto_approve_node(state: ClaimState):
    pay = _payable(state)
    return {"final_status": "APPROVED", "decided_by": "System (automatic)", "payable_amount": pay,
            "audit_log": [_log("auto_approve", f"Claim auto-approved, payable {inr(pay)}")]}


def auto_reject_node(state: ClaimState):
    update = {"final_status": "REJECTED", "decided_by": "System (automatic)", "payable_amount": 0.0,
              "audit_log": [_log("auto_reject", "Claim auto-rejected: " + "; ".join(state.get("decision_reasons", [])))]}
    if not state.get("summary"):
        update["summary"] = "Claim rejected automatically. " + "; ".join(state.get("decision_reasons", []))
    return update


def human_approval_node(state: ClaimState):
    """Human-in-the-loop: pauses the graph until a reviewer approves or rejects (LangGraph interrupt)."""
    answer = interrupt({
        "claim_id": state["claim"].get("claim_id"),
        "amount": state["claim"]["amount"],
        "reasons": state.get("decision_reasons", []),
        "summary": state.get("summary", ""),
    })
    approved = answer.get("action") == "approve"
    pay = _payable(state) if approved else 0.0
    reviewer = (answer.get("reviewer") or "").strip() or "unnamed reviewer"
    note = (answer.get("notes") or "").strip()
    return {"human_decision": answer, "final_status": "APPROVED" if approved else "REJECTED",
            "decided_by": f"Human reviewer ({reviewer})", "payable_amount": pay,
            "audit_log": [_log("human_approval", f"{'Approved' if approved else 'Rejected'} by {reviewer}"
                               + (f": {note}" if note else ""))]}
