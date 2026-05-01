"""
Stage 2 — System Design Layer
Input:  IntentSchema (from Stage 1)
Output: SystemDesignSchema (validated Pydantic model -> JSON)

Converts intent -> full app architecture:
  - Pages & navigation
  - API groups & endpoints (high-level)
  - Database entities & relations
  - Auth & permission model
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

# Stage 1 import
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline.stage1_intent import IntentSchema, extract_intent

load_dotenv()

# --- Pydantic Models ---------------------------------------------------------

class PageModel(BaseModel):
    name: str                          # e.g. "Dashboard"
    route: str                         # e.g. "/dashboard"
    access: list[str]                  # roles that can access, e.g. ["admin", "user"]
    components: list[str]              # e.g. ["StatsCard", "ContactsTable"]
    requires_auth: bool

class RelationModel(BaseModel):
    from_entity: str                   # e.g. "User"
    to_entity: str                     # e.g. "Contact"
    type: str                          # one_to_many | many_to_many | one_to_one
    description: str                   # e.g. "A user owns many contacts"

class EntityModel(BaseModel):
    name: str                          # e.g. "Contact"
    description: str
    key_fields: list[str]              # e.g. ["id", "name", "email", "owner_id"]
    relations: list[RelationModel] = Field(default_factory=list)

class PermissionModel(BaseModel):
    role: str                          # e.g. "admin"
    can_access: list[str]              # page names
    can_manage: list[str]              # entity names (full CRUD)
    can_view: list[str]                # entity names (read-only)

class AuthDesign(BaseModel):
    strategy: str                      # jwt | session | oauth
    token_expiry: str                  # e.g. "7d"
    refresh_token: bool
    protected_routes: list[str]        # routes that need auth
    public_routes: list[str]           # routes that don't

class APIGroup(BaseModel):
    name: str                          # e.g. "Contacts"
    base_path: str                     # e.g. "/api/contacts"
    endpoints: list[str]               # e.g. ["GET /", "POST /", "PUT /:id", "DELETE /:id"]
    auth_required: bool
    roles_allowed: list[str]

class SystemDesignSchema(BaseModel):
    app_name: str
    app_type: str

    pages: list[PageModel]
    entities: list[EntityModel]
    api_groups: list[APIGroup]
    auth_design: AuthDesign
    permissions: list[PermissionModel]

    design_decisions: list[str] = Field(
        default_factory=list,
        description="Key architectural decisions made and why"
    )
    flagged_conflicts: list[str] = Field(
        default_factory=list,
        description="Any conflicting requirements detected"
    )

# --- Prompt ------------------------------------------------------------------

SYSTEM_PROMPT = """You are Stage 2 of an app-generation compiler — the System Design Layer.

Your job: take a structured intent JSON and produce a complete app architecture.

Rules:
1. Output ONLY valid JSON — no markdown, no explanation, no code fences.
2. Standardize Naming: Use 'Billing' or 'Subscriptions' for payment-related pages. Avoid generic names like 'Payment Plan'.
3. Mandatory CRM Pages: If domain is 'crm', you MUST include: ['Dashboard', 'Contacts', 'Analytics', 'Billing', 'Subscriptions'].
4. Entity Modeling: entities[] must be REAL data entities that store rows (users, contacts, plans, subscriptions, payments). NEVER create entities for views/aggregations like 'analytics' or 'billing' — those are UI features computed from real entities.
5. Relationships: Every child entity MUST list its parent's foreign key (e.g. contacts has user_id, subscriptions has user_id + plan_id, payments has subscription_id).
6. entities[].key_fields must include id, created_at, updated_at, and all foreign keys.
7. api_groups must cover every entity with standard CRUD + any special endpoints.
8. permissions must be defined for EVERY role from the intent.
9. auth_design.protected_routes must list every route that needs login.
10. Add design_decisions explaining why you chose specific pages/entities.

Output schema (follow exactly):
{
  "app_name": "string",
  "app_type": "string",
  "pages": [
    {
      "name": "string",
      "route": "string",
      "access": ["role"],
      "components": ["ComponentName"],
      "requires_auth": true
    }
  ],
  "entities": [
    {
      "name": "string",
      "description": "string",
      "key_fields": ["string"],
      "relations": [
        {
          "from_entity": "string",
          "to_entity": "string",
          "type": "one_to_many",
          "description": "string"
        }
      ]
    }
  ],
  "api_groups": [
    {
      "name": "string",
      "base_path": "string",
      "endpoints": ["METHOD /path"],
      "auth_required": true,
      "roles_allowed": ["role"]
    }
  ],
  "auth_design": {
    "strategy": "jwt",
    "token_expiry": "7d",
    "refresh_token": true,
    "protected_routes": ["/string"],
    "public_routes": ["/string"]
  },
  "permissions": [
    {
      "role": "string",
      "can_access": ["page_name"],
      "can_manage": ["entity_name"],
      "can_view": ["entity_name"]
    }
  ],
  "design_decisions": ["string"],
  "flagged_conflicts": ["string"]
}"""

# --- Core Function ------------------------------------------------------------

def design_system(intent: IntentSchema, debug: bool = False) -> SystemDesignSchema:
    """
    Takes validated IntentSchema -> returns validated SystemDesignSchema.
    Repairs automatically on JSON/validation failure (max 2 attempts).
    """
    intent_json = intent.model_dump_json(indent=2)
    raw = _call_llm(intent_json, debug=debug)
    return _parse_and_validate(raw, intent_json, attempt=1, debug=debug)


def _call_llm(intent_json: str, debug: bool = False) -> str:
    return call_llm(
        system_prompt=SYSTEM_PROMPT,
        user_content=(
            "Design the full system architecture for this app intent:\n\n"
            f"{intent_json}"
        ),
        label="2",
        max_tokens=3000,
        debug=debug
    )


def _parse_and_validate(
    raw: str,
    intent_json: str,
    attempt: int,
    debug: bool = False
) -> SystemDesignSchema:

    # Step 1: parse JSON
    cleaned = clean_json(raw)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        if attempt > 2:
            raise ValueError(f"[Stage 2] JSON parse failed after 2 attempts: {e}")
        if debug:
            print(f"[Stage 2] JSON parse error (attempt {attempt}): {e} — retrying…")
        repaired = _call_llm(intent_json, debug=debug)
        return _parse_and_validate(repaired, intent_json, attempt + 1, debug=debug)

    # Step 2: validate schema
    try:
        design = SystemDesignSchema(**data)
        return design
    except ValidationError as e:
        if attempt > 2:
            raise ValueError(f"[Stage 2] Schema validation failed after 2 attempts:\n{e}")
        if debug:
            print(f"[Stage 2] Validation error (attempt {attempt}):\n{e}\nRetrying…")
        repair_prompt = (
            f"Your previous output had these validation errors:\n{e}\n\n"
            f"Original intent:\n{intent_json}\n\n"
            f"Your broken output:\n{raw}\n\n"
            "Fix ONLY the invalid fields and return the corrected JSON."
        )
        repaired = _call_llm(repair_prompt, debug=debug)
        return _parse_and_validate(repaired, intent_json, attempt + 1, debug=debug)


# --- CLI Test ---------------------------------------------------------------──

if __name__ == "__main__":
    test_prompt = (
        "Build a CRM with login, contacts, dashboard, role-based access, "
        "and premium plan with payments. Admins can see analytics."
    )

    print("=" * 60)
    print("Running Stage 1 -> Stage 2 pipeline")
    print("=" * 60)

    # Stage 1
    print("\n[Stage 1] Extracting intent…")
    intent = extract_intent(test_prompt, debug=False)
    print(f"[OK] Intent extracted: {intent.app_name} ({intent.app_type})")
    print(f"   Entities: {intent.entities}")
    print(f"   Roles:    {intent.roles}")

    # Stage 2
    print("\n[Stage 2] Designing system…")
    design = design_system(intent, debug=True)

    print("\n[Stage 2] [OK] Validated SystemDesignSchema:")
    print(design.model_dump_json(indent=2))

    print("\n" + "=" * 60)
    print(f"Pages:      {[p.name for p in design.pages]}")
    print(f"Entities:   {[e.name for e in design.entities]}")
    print(f"API Groups: {[a.name for a in design.api_groups]}")
    print(f"Roles:      {[p.role for p in design.permissions]}")
    print("=" * 60)