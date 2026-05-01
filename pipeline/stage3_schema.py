"""
Stage 3 — Schema Generation
Input:  SystemDesignSchema (from Stage 2)
Output: FullSchemaOutput containing:
          - UISchema    (pages -> components -> fields)
          - APISchema   (endpoints -> request/response bodies)
          - DBSchema    (tables -> columns -> constraints -> indexes)

All 3 sub-schemas are generated in PARALLEL using threads,
then validated individually with Pydantic.
"""

import os
import json
import sys
import concurrent.futures
from pydantic import BaseModel, Field, ValidationError
from dotenv import load_dotenv

# Import shared LLM client
try:
    from pipeline.llm_client import call_llm, clean_json
except ImportError:
    from llm_client import call_llm, clean_json

load_dotenv()

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline.stage1_intent import extract_intent
from pipeline.stage2_design import SystemDesignSchema, design_system

load_dotenv()

# --- Pydantic Models — UI Schema ---------------------------------------------

class FieldModel(BaseModel):
    name: str                        # e.g. "email"
    label: str                       # e.g. "Email Address"
    type: str                        # text | email | password | select | checkbox | date | number | textarea
    required: bool
    placeholder: str = ""
    options: list[str] = Field(default_factory=list)   # for select fields
    validation: list[str] = Field(default_factory=list) # e.g. ["min:3", "max:100", "email"]

class ComponentModel(BaseModel):
    name: str                        # e.g. "LoginForm"
    type: str                        # form | table | card | chart | modal | navbar | sidebar
    fields: list[FieldModel] = Field(default_factory=list)
    actions: list[str] = Field(default_factory=list)   # e.g. ["submit", "reset", "delete"]
    data_source: str = ""            # e.g. "/api/contacts"

class UIPage(BaseModel):
    name: str
    route: str
    title: str
    requires_auth: bool
    allowed_roles: list[str]
    layout: str                      # sidebar | fullpage | centered | dashboard
    components: list[ComponentModel]

class UISchema(BaseModel):
    pages: list[UIPage]

# --- Pydantic Models — API Schema ------------------------------------------──

class RequestBody(BaseModel):
    content_type: str = "application/json"
    fields: dict[str, str]           # field_name -> type  e.g. {"email": "string", "age": "number"}
    required_fields: list[str]

class ResponseBody(BaseModel):
    success_status: int              # e.g. 200, 201
    fields: dict[str, str]           # field_name -> type
    is_array: bool = False

class EndpointModel(BaseModel):
    method: str                      # GET | POST | PUT | DELETE | PATCH
    path: str                        # e.g. "/api/contacts/:id"
    summary: str
    auth_required: bool
    roles_allowed: list[str]
    request_body: RequestBody | None = None
    response: ResponseBody
    error_codes: list[int] = Field(default_factory=list)  # e.g. [400, 401, 404]

class APISchema(BaseModel):
    endpoints: list[EndpointModel]

# --- Pydantic Models — DB Schema ---------------------------------------------

class ColumnModel(BaseModel):
    name: str                        # e.g. "email"
    type: str                        # VARCHAR | TEXT | INTEGER | BOOLEAN | TIMESTAMP | DECIMAL | UUID
    nullable: bool = False
    unique: bool = False
    primary_key: bool = False
    foreign_key: str = ""            # e.g. "users.id"
    default: str = ""                # e.g. "NOW()" or "false"

class TableModel(BaseModel):
    name: str                        # e.g. "contacts"
    description: str
    columns: list[ColumnModel]
    indexes: list[str] = Field(default_factory=list)  # e.g. ["email", "user_id"]

class DBSchema(BaseModel):
    tables: list[TableModel]
    migration_order: list[str]       # table creation order respecting FK dependencies

# --- Combined Output ---------------------------------------------------------─

class FullSchemaOutput(BaseModel):
    app_name: str
    ui: UISchema
    api: APISchema
    db: DBSchema

# --- Prompts ------------------------------------------------------------------

UI_SYSTEM_PROMPT = """You are Stage 3a of an app-generation compiler — UI Schema Generator.

Generate a complete UI schema from the system design.

Rules:
1. Output ONLY valid JSON — no markdown, no explanation, no code fences.
2. Every page from the design must have a UIPage entry.
3. Every component must have realistic fields (for forms) or actions (for tables).
4. data_source must be a real API path from the design (e.g. "/api/contacts").
5. layout options: sidebar | fullpage | centered | dashboard
6. component type options: form | table | card | chart | modal | navbar | sidebar
7. field type options: text | email | password | select | checkbox | date | number | textarea

Output schema:
{
  "pages": [
    {
      "name": "string",
      "route": "string",
      "title": "string",
      "requires_auth": true,
      "allowed_roles": ["role"],
      "layout": "string",
      "components": [
        {
          "name": "string",
          "type": "string",
          "fields": [
            {
              "name": "string",
              "label": "string",
              "type": "string",
              "required": true,
              "placeholder": "string",
              "options": [],
              "validation": []
            }
          ],
          "actions": ["string"],
          "data_source": "string"
        }
      ]
    }
  ]
}"""

API_SYSTEM_PROMPT = """You are Stage 3b of an app-generation compiler — API Schema Generator.

Generate a complete API schema with full request/response definitions.

Rules:
1. Output ONLY valid JSON — no markdown, no explanation, no code fences.
2. Every endpoint from every api_group must be fully defined.
3. request_body is null for GET and DELETE requests.
4. response.fields must list ALL fields returned (include id, created_at, etc.).
5. error_codes must be realistic (401 for auth, 404 for not found, 400 for validation, etc.).
6. fields in request_body and response are objects: {"field_name": "type_string"}.
7. type strings: string | number | boolean | date | uuid | array | object

Output schema:
{
  "endpoints": [
    {
      "method": "string",
      "path": "string",
      "summary": "string",
      "auth_required": true,
      "roles_allowed": ["role"],
      "request_body": {
        "content_type": "application/json",
        "fields": {"field": "type"},
        "required_fields": ["field"]
      },
      "response": {
        "success_status": 200,
        "fields": {"field": "type"},
        "is_array": false
      },
      "error_codes": [400, 401, 404]
    }
  ]
}"""

DB_SYSTEM_PROMPT = """You are Stage 3c of an app-generation compiler — Database Schema Generator.

Generate a complete relational database schema.

Rules:
1. Output ONLY valid JSON — no markdown, no explanation, no code fences.
2. Tables MUST be real data entities (users, contacts, plans, subscriptions, payments). NEVER create tables for views/aggregations like 'analytics', 'billing', 'dashboard' — those are computed in the application layer.
3. Every table MUST have: id (UUID, PK), created_at (TIMESTAMP), updated_at (TIMESTAMP).
4. Relationships: Every child table MUST have explicit foreign_key columns. Example: subscriptions needs user_id (-> users.id) and plan_id (-> plans.id). payments needs subscription_id (-> subscriptions.id).
5. Column types: UUID | VARCHAR | TEXT | INTEGER | BOOLEAN | TIMESTAMP | DECIMAL | JSONB.
6. migration_order: MUST list tables in dependency order (parents before children, e.g. 'users' before 'contacts').
7. Add indexes for all foreign keys and frequently queried fields (email, name, status).

Output schema:
{
  "tables": [
    {
      "name": "string",
      "description": "string",
      "columns": [
        {
          "name": "string",
          "type": "string",
          "nullable": false,
          "unique": false,
          "primary_key": false,
          "foreign_key": "",
          "default": ""
        }
      ],
      "indexes": ["column_name"]
    }
  ],
  "migration_order": ["table_name"]
}"""

# --- Individual Generators ---------------------------------------------------─

def _generate_ui(design_json: str, debug: bool = False) -> UISchema:
    raw = _call_llm(UI_SYSTEM_PROMPT, design_json, "UI Schema", max_tokens=6000, debug=debug)
    return _validate(raw, UISchema, "UI", design_json, UI_SYSTEM_PROMPT, debug=debug)

def _generate_api(design_json: str, debug: bool = False) -> APISchema:
    raw = _call_llm(API_SYSTEM_PROMPT, design_json, "API Schema", max_tokens=6000, debug=debug)
    return _validate(raw, APISchema, "API", design_json, API_SYSTEM_PROMPT, debug=debug)

def _generate_db(design_json: str, debug: bool = False) -> DBSchema:
    raw = _call_llm(DB_SYSTEM_PROMPT, design_json, "DB Schema", max_tokens=6000, debug=debug)
    return _validate(raw, DBSchema, "DB", design_json, DB_SYSTEM_PROMPT, debug=debug)


def _call_llm(system: str, design_json: str, label: str, max_tokens: int, debug: bool) -> str:
    return call_llm(
        system_prompt=system,
        user_content=f"Generate the {label} for this system design:\n\n{design_json}",
        label=f"3-{label}",
        max_tokens=max_tokens,
        debug=debug
    )




def _validate(raw: str, model_class, label: str, design_json: str, system: str,
              attempt: int = 1, debug: bool = False):
    # Clean and Parse JSON
    cleaned = clean_json(raw)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        if attempt > 2:
            raise ValueError(f"[Stage 3 — {label}] JSON parse failed: {e}")
        if debug:
            print(f"[Stage 3 — {label}] JSON error (attempt {attempt}): {e} — retrying…")
        repaired = _call_llm(system, design_json, label, 6000, debug)
        return _validate(repaired, model_class, label, design_json, system, attempt + 1, debug)

    # Validate Pydantic
    try:
        return model_class(**data)
    except ValidationError as e:
        if attempt > 2:
            raise ValueError(f"[Stage 3 — {label}] Validation failed:\n{e}")
        if debug:
            print(f"[Stage 3 — {label}] Validation error (attempt {attempt}):\n{e} — repairing…")
        repair_prompt = (
            f"Validation errors:\n{e}\n\n"
            f"System design:\n{design_json}\n\n"
            f"Your broken output:\n{raw}\n\n"
            "Fix ONLY the invalid fields. Return corrected JSON only."
        )
        repaired = _call_llm(system, repair_prompt, label, 6000, debug)
        return _validate(repaired, model_class, label, design_json, system, attempt + 1, debug)


# --- Core Function (runs 3 generators in parallel) ---------------------------─

def generate_schemas(design: SystemDesignSchema, debug: bool = False, use_cache: bool = True) -> FullSchemaOutput:
    """
    Runs UI, API, DB generation in parallel using ThreadPoolExecutor.
    Returns FullSchemaOutput with all 3 validated schemas.
    """
    design_json = design.model_dump_json(indent=2)
    
    # High-level cache check
    from pipeline.llm_client import get_cache_key, CACHE_DIR
    cache_key = get_cache_key("stage3_full_schema", design_json)
    cache_path = os.path.join(CACHE_DIR, f"stage3_{cache_key}.json")
    
    if use_cache and os.path.exists(cache_path):
        if debug:
            print("  [Stage 3] Loading full schema from cache.")
        with open(cache_path, "r") as f:
            return FullSchemaOutput.model_validate_json(f.read())

    print("  -> Launching UI, API, DB generators sequentially…")

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future_ui  = executor.submit(_generate_ui,  design_json, debug)
        future_api = executor.submit(_generate_api, design_json, debug)
        future_db  = executor.submit(_generate_db,  design_json, debug)

        ui_schema  = future_ui.result()
        api_schema = future_api.result()
        db_schema  = future_db.result()

    full_output = FullSchemaOutput(
        app_name=design.app_name,
        ui=ui_schema,
        api=api_schema,
        db=db_schema
    )
    
    # Save to high-level cache
    with open(cache_path, "w") as f:
        f.write(full_output.model_dump_json(indent=2))
        
    return full_output


# --- CLI Test ---------------------------------------------------------------──

if __name__ == "__main__":
    test_prompt = (
        "Build a CRM with login, contacts, dashboard, role-based access, "
        "and premium plan with payments. Admins can see analytics."
    )

    print("=" * 60)
    print("Running Stage 1 -> 2 -> 3 pipeline")
    print("=" * 60)

    print("\n[Stage 1] Extracting intent…")
    intent = extract_intent(test_prompt, debug=False)
    print(f"Intent extracted: {intent.app_name} ({intent.app_type})")

    print("\n[Stage 2] Designing system…")
    design = design_system(intent, debug=False)
    print(f"Pages: {[p.name for p in design.pages]}")
    print(f"   Entities: {[e.name for e in design.entities]}")

    print("\n[Stage 3] Generating UI + API + DB schemas in parallel…")
    schemas = generate_schemas(design, debug=False)

    print("\n[Stage 3] FullSchemaOutput:")
    print(schemas.model_dump_json(indent=2))

    print("\n" + "=" * 60)
    print(f"UI Pages:      {[p.name for p in schemas.ui.pages]}")
    print(f"API Endpoints: {len(schemas.api.endpoints)} total")
    print(f"DB Tables:     {[t.name for t in schemas.db.tables]}")
    print(f"Migration order: {schemas.db.migration_order}")
    print("=" * 60)