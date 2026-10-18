"""Mock policy database, business rules and the five demo scenarios.

All dates are generated relative to today so the demo scenarios keep working
whenever the app is run (a fixed 'active until 2026-12-31' policy would eventually expire).
"""
from datetime import date, timedelta

CLAIM_TYPES = ["Health", "Motor", "Home"]

REQUIRED_DOCUMENTS = {
    "Health": ["Claim form", "Hospital bills", "Doctor's report / discharge summary", "ID proof"],
    "Motor": ["Claim form", "Police report (FIR)", "Repair estimate", "Photos of damage", "Driving licence & RC"],
    "Home": ["Claim form", "Photos of damage", "Repair estimate", "ID proof"],
}
ALL_DOCUMENTS = sorted({d for docs in REQUIRED_DOCUMENTS.values() for d in docs})

TYPICAL_AMOUNT = {"Health": 80_000, "Motor": 80_000, "Home": 100_000}  # INR, used by fraud rules
HIGH_VALUE_THRESHOLD = 500_000  # INR: claims at/above this always need a human
MINOR_KEYWORDS = ["minor", "small", "slight", "scratch", "scratches", "dent", "tiny", "cosmetic"]


def _d(offset_days: int) -> str:
    return (date.today() + timedelta(days=offset_days)).isoformat()


POLICIES = {
    "POL-1001": {"holder": "Ananya Rao", "covers": ["Health"], "start": _d(-200), "end": _d(165),
                 "coverage_limit": 1_000_000, "prior_claims_12m": 0},
    "POL-1002": {"holder": "Vikram Singh", "covers": ["Motor"], "start": _d(-300), "end": _d(65),
                 "coverage_limit": 500_000, "prior_claims_12m": 0},
    "POL-1003": {"holder": "Rohit Malhotra", "covers": ["Motor"], "start": _d(-340), "end": _d(25),
                 "coverage_limit": 600_000, "prior_claims_12m": 1},
    "POL-1004": {"holder": "Meera Nair", "covers": ["Motor", "Home"], "start": _d(-485), "end": _d(-120),
                 "coverage_limit": 500_000, "prior_claims_12m": 0},  # expired
    "POL-1005": {"holder": "Sanjay Gupta", "covers": ["Health"], "start": _d(-150), "end": _d(215),
                 "coverage_limit": 1_000_000, "prior_claims_12m": 0},
    "POL-1006": {"holder": "Priya Kapoor", "covers": ["Home"], "start": _d(-90), "end": _d(275),
                 "coverage_limit": 800_000, "prior_claims_12m": 0},
}

ALL_HEALTH = REQUIRED_DOCUMENTS["Health"]
ALL_MOTOR = REQUIRED_DOCUMENTS["Motor"]

# expected: AUTO_APPROVE | AUTO_REJECT | HUMAN_REVIEW
SCENARIOS = [
    {
        "name": "1. Complete claim, valid documents",
        "expected": "AUTO_APPROVE",
        "claim": {
            "policy_number": "POL-1001", "claimant_name": "Ananya Rao", "claim_type": "Health",
            "incident_days_ago": 12, "amount": 45_000,
            "description": "Hospitalised for 3 days with dengue fever. Treatment at City Care Hospital, bills and discharge summary attached.",
            "documents_provided": ALL_HEALTH,
        },
    },
    {
        "name": "2. Missing required documents",
        "expected": "AUTO_REJECT",
        "claim": {
            "policy_number": "POL-1002", "claimant_name": "Vikram Singh", "claim_type": "Motor",
            "incident_days_ago": 9, "amount": 65_000,
            "description": "Rear-ended at a traffic signal while stopped. Bumper and tail lamp damaged, car drivable.",
            "documents_provided": ["Claim form", "Photos of damage", "Driving licence & RC"],
        },
    },
    {
        "name": "3. Suspicious claim amount",
        "expected": "HUMAN_REVIEW",
        "claim": {
            "policy_number": "POL-1003", "claimant_name": "Rohit Malhotra", "claim_type": "Motor",
            "incident_days_ago": 6, "amount": 420_000,
            "description": "Minor scratch on the rear bumper from a parking incident. Requesting full repaint and replacement of several parts.",
            "documents_provided": ALL_MOTOR,
        },
    },
    {
        "name": "4. Expired policy",
        "expected": "AUTO_REJECT",
        "claim": {
            "policy_number": "POL-1004", "claimant_name": "Meera Nair", "claim_type": "Motor",
            "incident_days_ago": 30, "amount": 55_000,
            "description": "Side collision at an intersection damaged the left door and mirror. Police report filed.",
            "documents_provided": ALL_MOTOR,
        },
    },
    {
        "name": "5. High-value claim, valid documents",
        "expected": "HUMAN_REVIEW",
        "claim": {
            "policy_number": "POL-1005", "claimant_name": "Sanjay Gupta", "claim_type": "Health",
            "incident_days_ago": 15, "amount": 650_000,
            "description": "Emergency cardiac bypass surgery with a 9-day ICU and hospital stay. All hospital bills and the surgeon's report are attached.",
            "documents_provided": ALL_HEALTH,
        },
    },
]


def make_claim(template: dict) -> dict:
    """Turn a scenario template into a claim with real ISO dates."""
    c = dict(template)
    days = c.pop("incident_days_ago", 7)
    c["incident_date"] = _d(-days)
    c["claim_date"] = date.today().isoformat()
    return c
