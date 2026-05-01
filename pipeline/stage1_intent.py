"""
Stage 1 — Intent Extraction
Input:  raw user prompt (string)
Output: structured IntentSchema (validated Pydantic model → JSON)
"""

import os
import json
from pydantic import BaseModel, Field, ValidationError
from typing import Optional
from dotenv import load_dotenv

# Import shared LLM client
try:
    from pipeline.llm_client import call_llm, clean_json
except ImportError:
    from llm_client import call_llm, clean_json

load_dotenv()

# ─── Pydantic Models ────────────────────────────────────────────────────────

class AuthRequirement(BaseModel):
    needed: bool
    type: str = Field(
        default="jwt",
        description="jwt | session | oauth | none"
    )
    social_login: list[str] = Field(default_factory=list)   # ["google", "github"]

class PaymentRequirement(BaseModel):
    needed: bool
    provider: str = Field(default="stripe", description="stripe | razorpay | none")
    plans: list[str] = Field(default_factory=list)           # ["free", "premium"]

class IntentSchema(BaseModel):
    app_name: str
    app_type: str = Field(
        description="crm | ecommerce | saas | blog | dashboard | social | other"
    )
    domain: str = Field(description="Normalized domain name, e.g. 'crm', 'store', 'social'")
    summary: str = Field(description="1-2 sentence plain-English summary of the app")

    # Core building blocks
    entities: list[str] = Field(
        description="Main data entities, e.g. ['User', 'Contact', 'Invoice']"
    )
    features: list[str] = Field(
        description="Feature list, e.g. ['dashboard', 'analytics', 'payments', 'auth']"
    )
    roles: list[str] = Field(
        description="User roles, e.g. ['admin', 'user', 'guest']"
    )

    # Sub-systems
    auth: AuthRequirement
    payments: PaymentRequirement

    # Ambiguities the model detected
    assumptions_made: list[str] = Field(
        default_factory=list,
        description="Anything unclear that was assumed"
    )
    clarifications_needed: list[str] = Field(
        default_factory=list,
        description="Things too ambiguous to assume — should ask user"
    )

# ─── Prompt ─────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are Stage 1 of an app-generation compiler.

Your ONLY job: extract structured intent from a user's natural language app description.

Rules:
1. Output ONLY valid JSON — no markdown, no explanation, no code fences.
2. Every field in the schema is required. Use empty arrays [] for missing lists.
3. Be specific: extract REAL entity names ("Contact", "Invoice") not generic ones ("Item").
4. domain should be a single lowercase slug (e.g. "crm", "ecommerce", "saas").
5. features should list the high-level capabilities (e.g. ["analytics", "role-based access"]).
6. app_type must be one of: crm | ecommerce | saas | blog | dashboard | social | other

Output schema:
{
  "app_name": "string",
  "app_type": "string",
  "domain": "string",
  "summary": "string",
  "entities": ["string"],
  "features": ["string"],
  "roles": ["string"],
  "auth": {
    "needed": true,
    "type": "jwt",
    "social_login": []
  },
  "payments": {
    "needed": false,
    "provider": "none",
    "plans": []
  },
  "assumptions_made": ["string"],
  "clarifications_needed": ["string"]
}"""

# ─── Core Function ───────────────────────────────────────────────────────────

def extract_intent(user_prompt: str, debug: bool = False) -> IntentSchema:
    """
    Calls Groq, parses JSON, validates with Pydantic.
    Raises ValueError if validation fails after 2 repair attempts.
    """
    raw_json = _call_llm(user_prompt, debug=debug)
    return _parse_and_validate(raw_json, user_prompt, attempt=1, debug=debug)


def _call_llm(user_prompt: str, debug: bool = False) -> str:
    return call_llm(
        system_prompt=SYSTEM_PROMPT,
        user_content=f"Extract intent from this app description:\n\n{user_prompt}",
        label="1",
        max_tokens=1500,
        debug=debug
    )


def _parse_and_validate(
    raw: str,
    original_prompt: str,
    attempt: int,
    debug: bool = False
) -> IntentSchema:
    """Parse JSON → validate Pydantic → repair once if needed."""

    # Step 1: parse JSON
    cleaned = clean_json(raw)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        if attempt > 2:
            raise ValueError(f"[Stage 1] JSON parse failed after 2 attempts: {e}")
        if debug:
            print(f"[Stage 1] JSON parse error (attempt {attempt}): {e} — retrying…")
        repaired = _call_llm(original_prompt, debug=debug)
        return _parse_and_validate(repaired, original_prompt, attempt + 1, debug=debug)

    # Step 2: validate schema
    try:
        intent = IntentSchema(**data)
        return intent
    except ValidationError as e:
        if attempt > 2:
            raise ValueError(f"[Stage 1] Schema validation failed after 2 attempts:\n{e}")
        if debug:
            print(f"[Stage 1] Validation error (attempt {attempt}):\n{e}\nRetrying…")
        # Feed the validation errors back to the model for surgical repair
        repair_prompt = (
            f"Your previous output had these validation errors:\n{e}\n\n"
            f"Original prompt: {original_prompt}\n\n"
            f"Your broken output:\n{raw}\n\n"
            "Fix ONLY the invalid fields and return the corrected JSON."
        )
        repaired = _call_llm(repair_prompt, debug=debug)
        return _parse_and_validate(repaired, original_prompt, attempt + 1, debug=debug)


# ─── CLI Test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_prompt = (
        "Build a CRM with login, contacts, dashboard, role-based access, "
        "and premium plan with payments. Admins can see analytics."
    )
    print("Input prompt:", test_prompt)
    print("-" * 60)

    result = extract_intent(test_prompt, debug=True)

    print("\n[Stage 1] Validated IntentSchema:")
    print(result.model_dump_json(indent=2))