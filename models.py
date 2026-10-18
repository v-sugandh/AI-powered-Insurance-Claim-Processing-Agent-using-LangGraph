"""Pydantic schema for the LLM's fraud analysis (parsed with PydanticOutputParser)."""
from typing import List

from pydantic import BaseModel, Field


class FraudAnalysis(BaseModel):
    concern: bool = Field(description="True ONLY if there is a concrete red flag that needs human attention")
    findings: List[str] = Field(description="Specific red flags found, empty if none")
    reasoning: str = Field(description="One or two sentences explaining the assessment")
