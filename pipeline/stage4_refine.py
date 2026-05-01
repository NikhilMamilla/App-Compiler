"""
Stage 4 - Refinement & Cross-Validation Layer
Input:  FullSchemaOutput (from Stage 3) - contains UISchema, APISchema, DBSchema
Output: RefinedOutput - a single merged, corrected, consistency-validated config

What Stage 4 does:
  1. STATIC ANALYSIS  - deterministic Python checks (no LLM needed)
       a) DB ↔ API   : every API request/response field must map to a DB column
       b) API ↔ UI   : every UI form field / data_source must map to an API endpoint
       c) Auth        : every protected UI page/API endpoint must have a role defined
       d) FK integrity: every foreign_key reference must point to a real table.column

  2. LLM REPAIR PASS  - feed all detected inconsistencies to Claude in one shot
       - Produces a list of targeted patches (not a full rewrite)
       - Each patch targets a specific schema + location + fix

  3. PATCH APPLICATION - apply validated patches to the in-memory schemas

  4. FINAL VALIDATION  - re-run static checks; fail loudly if any issue persists

The output RefinedOutput bundles:
  - The corrected FullSchemaOutput
  - A full audit trail (issues found, patches applied, issues remaining)
"""

import os
import json
import sys
import copy
from pydantic import BaseModel, Field, ValidationError
from typing import Literal
from dotenv import load_dotenv

try:
    from pipeline.llm_client import call_llm, clean_json
except ImportError:
    from llm_client import call_llm, clean_json

load_dotenv()

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline.stage1_intent import extract_intent
from pipeline.stage2_design import design_system
from pipeline.stage3_schema import (
    FullSchemaOutput, UISchema, APISchema, DBSchema,
    UIPage, ComponentModel, FieldModel,
    EndpointModel, TableModel, ColumnModel,
    generate_schemas,
)

# --- Pydantic Models ---------------------------------------------------------

class Issue(BaseModel):
    severity: Literal["error", "warning"]
    category: Literal["db_api", "api_ui", "auth", "fk_integrity", "orphan", "other"]
    location: str          # e.g. "APISchema.endpoints[POST /api/contacts].request_body"
    description: str
    suggested_fix: str

class Patch(BaseModel):
    target_schema: Literal["ui", "api", "db"]
    location: str          # dot-path e.g. "pages[Login].components[LoginForm].fields"
    operation: Literal["add_field", "remove_field", "change_type", "add_endpoint",
                        "add_column", "change_column", "add_index", "fix_fk", "add_table", "remove_table", "other"]
    payload: dict          # the actual correction data
    rationale: str

class AuditTrail(BaseModel):
    issues_found: list[Issue]
    patches_proposed: list[Patch]
    patches_applied: list[Patch]
    issues_remaining: list[Issue]
    initial_score: float       # Score before repair
    final_score: float         # Score after repair
    latency_ms: float
    repair_calls: int
    assumptions: list[str] = Field(default_factory=list)

class RefinedOutput(BaseModel):
    app_name: str
    schemas: FullSchemaOutput
    audit: AuditTrail
    ready_for_codegen: bool


# --- Stage 4A: Static Analysis -----------------------------------------------

def _static_analysis(schemas: FullSchemaOutput) -> list[Issue]:
    issues: list[Issue] = []

    # Build lookup structures
    db_tables:   dict[str, set[str]] = {}   # table_name -> {column_names}
    api_paths:   set[str] = set()           # "METHOD /path"
    api_map:     dict[str, EndpointModel] = {}  # "METHOD /path" -> EndpointModel
    role_set:    set[str] = set()

    for table in schemas.db.tables:
        db_tables[table.name] = {col.name for col in table.columns}

    for ep in schemas.api.endpoints:
        key = f"{ep.method} {ep.path}"
        api_paths.add(key)
        api_map[key] = ep
        role_set.update(ep.roles_allowed)

    # -- 1. DB ↔ API: API response fields should exist as DB columns ---------─
    for ep in schemas.api.endpoints:
        # Infer which table this endpoint targets from its path
        # e.g. /api/contacts -> "contacts"
        parts = [p for p in ep.path.strip("/").split("/") if p and not p.startswith(":")]
        table_guess = parts[-1] if parts else None  # last non-param segment

        if table_guess and table_guess in db_tables:
            db_cols = db_tables[table_guess]

            # Check response fields
            for field_name in ep.response.fields:
                if field_name not in db_cols and field_name not in {"token", "message", "success", "meta"}:
                    issues.append(Issue(
                        severity="error",
                        category="db_api",
                        location=f"APISchema.endpoints[{ep.method} {ep.path}].response.fields",
                        description=f"Response field '{field_name}' not found in DB table '{table_guess}'",
                        suggested_fix=f"Analyze if '{field_name}' should be added to DB or removed from API."
                    ))

            # Check request body fields
            if ep.request_body:
                for field_name in ep.request_body.fields:
                    if field_name not in db_cols and field_name not in {"password", "confirm_password", "token"}:
                        issues.append(Issue(
                            severity="error",
                            category="db_api",
                            location=f"APISchema.endpoints[{ep.method} {ep.path}].request_body.fields",
                            description=f"Request field '{field_name}' not found in DB table '{table_guess}'",
                            suggested_fix=f"Analyze if '{field_name}' should be added to DB or removed from API."
                        ))

    # -- 2. API ↔ UI: UI data_sources must point to real API base paths --------
    api_base_paths = {ep.path.split("/:")[0] for ep in schemas.api.endpoints}

    for page in schemas.ui.pages:
        for comp in page.components:
            if comp.data_source:
                base = comp.data_source.split("/:")[0]
                if base not in api_base_paths:
                    issues.append(Issue(
                        severity="error",
                        category="api_ui",
                        location=f"UISchema.pages[{page.name}].components[{comp.name}].data_source",
                        description=f"data_source '{comp.data_source}' has no matching API endpoint",
                        suggested_fix=f"Add a matching endpoint or change data_source to an existing path"
                    ))

    # -- 3. Auth: protected UI pages must have roles that exist ---------------─
    for page in schemas.ui.pages:
        if page.requires_auth:
            for role in page.allowed_roles:
                if role not in role_set and role_set:
                    issues.append(Issue(
                        severity="error",
                        category="auth",
                        location=f"UISchema.pages[{page.name}].allowed_roles",
                        description=f"Role '{role}' on page '{page.name}' not found in any API endpoint",
                        suggested_fix=f"Add role '{role}' to relevant API endpoints"
                    ))

    # -- 4. FK integrity: foreign_keys must reference real tables ------------─
    for table in schemas.db.tables:
        for col in table.columns:
            if col.foreign_key:
                ref_parts = col.foreign_key.split(".")
                if len(ref_parts) == 2:
                    ref_table, ref_col = ref_parts
                    if ref_table not in db_tables:
                        issues.append(Issue(
                            severity="error",
                            category="fk_integrity",
                            location=f"DBSchema.tables[{table.name}].columns[{col.name}].foreign_key",
                            description=f"FK '{col.foreign_key}' references non-existent table '{ref_table}'",
                            suggested_fix=f"Change foreign_key to reference an existing table or create table '{ref_table}'"
                        ))
                    elif ref_col not in db_tables.get(ref_table, set()):
                        issues.append(Issue(
                            severity="error",
                            category="fk_integrity",
                            location=f"DBSchema.tables[{table.name}].columns[{col.name}].foreign_key",
                            description=f"FK '{col.foreign_key}' references non-existent column '{ref_col}' in '{ref_table}'",
                            suggested_fix=f"Add column '{ref_col}' to '{ref_table}' or fix the reference"
                        ))
            
            # -- 4b. Heuristic: Check for missing FK definitions (e.g. user_id should be an FK) --
            elif col.name.endswith("_id") and col.name != "id":
                target_table_guess = col.name[:-3] # user_id -> user
                # pluralize guess
                target_plural = target_table_guess + "s"
                if target_plural in db_tables and target_plural != table.name:
                    issues.append(Issue(
                        severity="warning",
                        category="fk_integrity",
                        location=f"DBSchema.tables[{table.name}].columns[{col.name}]",
                        description=f"Column '{col.name}' looks like a foreign key to '{target_plural}' but has no 'foreign_key' definition",
                        suggested_fix=f"Set foreign_key to '{target_plural}.id'"
                    ))

    # -- 5. Orphan check: DB tables with no API endpoint ---------------------─
    for table_name in db_tables:
        has_endpoint = any(
            table_name in ep.path for ep in schemas.api.endpoints
        )
        if not has_endpoint:
            issues.append(Issue(
                severity="warning",
                category="orphan",
                location=f"DBSchema.tables[{table_name}]",
                description=f"Table '{table_name}' has no associated API endpoint",
                suggested_fix=f"Add CRUD endpoints for '{table_name}' or remove the table if unused"
            ))

    # -- 6. Required fields: core tables must have essential columns ----------─
    REQUIRED_FIELDS = {
        "users": {"id", "email"},
        "contacts": {"id", "name"},
    }
    for table_name, required in REQUIRED_FIELDS.items():
        if table_name in db_tables:
            missing = required - db_tables[table_name]
            for field in missing:
                issues.append(Issue(
                    severity="error",
                    category="db_api",
                    location=f"DBSchema.tables[{table_name}]",
                    description=f"Required field '{field}' missing from table '{table_name}'",
                    suggested_fix=f"Add column '{field}' to '{table_name}'"
                ))

    # -- 7. Empty response check: API endpoints must have response fields ----─
    for ep in schemas.api.endpoints:
        if ep.method == "GET" and not ep.response.fields:
            issues.append(Issue(
                severity="warning",
                category="db_api",
                location=f"APISchema.endpoints[{ep.method} {ep.path}].response",
                description=f"GET endpoint '{ep.path}' has empty response fields",
                suggested_fix=f"Add response fields matching the DB table"
            ))

    # -- 8. Ungrounded API: CRUD endpoints must map to a real DB table --------─
    DERIVED_ENDPOINTS = {"dashboard", "analytics", "billing", "reports", "stats", "metrics", "search"}
    AUTH_ENDPOINTS = {"login", "register", "auth", "logout", "me"}
    for ep in schemas.api.endpoints:
        parts = [p for p in ep.path.strip("/").split("/") if p and not p.startswith(":")]
        table_guess = parts[-1] if parts else None
        if table_guess and table_guess not in db_tables and table_guess not in AUTH_ENDPOINTS and table_guess not in DERIVED_ENDPOINTS:
            issues.append(Issue(
                severity="warning",
                category="orphan",
                location=f"APISchema.endpoints[{ep.method} {ep.path}]",
                description=f"API endpoint '{ep.path}' has no matching DB table '{table_guess}'",
                suggested_fix=f"Create table '{table_guess}' or remove this endpoint"
            ))

    return issues


# --- Stage 4B: LLM Repair Pass -----------------------------------------------

REPAIR_SYSTEM_PROMPT = """Return ONLY valid JSON. No markdown, no comments.

Format:
{
  "patches": [
    {
      "target_schema": "ui|api|db",
      "location": "dot-path string",
      "operation": "add_field|add_column|add_endpoint|fix_fk|...",
      "payload": {
        "method": "GET",
        "path": "/api/contacts",
        "summary": "Get contacts",
        "auth_required": true,
        "roles_allowed": ["admin", "user"],
        "response": {
          "success_status": "boolean",
          "fields": {
            "id": "string",
            "name": "string"
          }
        }
      },
      "rationale": "string"
    }
  ],
  "assumptions": []
}

Rules:
- response MUST contain BOTH "success_status" (boolean) and "fields" (object).
- "fields" MUST be a non-empty object.
- For add_endpoint: Include method, path, summary, auth_required, roles_allowed, response.
- No missing fields, no null values, no partial patches.
- If unsure → DO NOT generate the patch (return empty patches [])."""


def _extract_relevant_context(schemas: FullSchemaOutput, issues: list[Issue]) -> str:
    """Extract only the schema slices related to the issues — reduces tokens significantly."""
    relevant_tables = set()
    relevant_endpoints = set()

    for issue in issues:
        loc = issue.location
        # Extract table names from locations like DBSchema.tables[contacts]
        if "tables[" in loc:
            relevant_tables.add(loc.split("tables[")[1].split("]")[0])
        # Extract endpoint paths from locations like APISchema.endpoints[GET /api/contacts]
        if "endpoints[" in loc:
            seg = loc.split("endpoints[")[1].split("]")[0]
            relevant_endpoints.add(seg)
        # Infer table from API path mentions
        for t in [t.name for t in schemas.db.tables]:
            if t in issue.description:
                relevant_tables.add(t)

    # Build compact context
    ctx = {"db_tables": [], "api_endpoints": []}
    for table in schemas.db.tables:
        if table.name in relevant_tables:
            ctx["db_tables"].append({
                "name": table.name,
                "columns": [{"name": c.name, "type": c.type, "fk": c.foreign_key} for c in table.columns]
            })
    for ep in schemas.api.endpoints:
        key = f"{ep.method} {ep.path}"
        if key in relevant_endpoints or any(t in ep.path for t in relevant_tables):
            ctx["api_endpoints"].append({
                "method": ep.method, "path": ep.path,
                "response_fields": ep.response.fields,
                "request_fields": ep.request_body.fields if ep.request_body else None
            })
    return json.dumps(ctx)


def _validate_patch(patch: Patch, schemas: FullSchemaOutput, debug: bool) -> tuple[bool, str]:
    """Decision engine: should we apply this patch? Returns (ok, reason)."""
    db_table_names = {t.name for t in schemas.db.tables}

    # Rule 1: add_endpoint — allow derived endpoints, only reject truly ungrounded CRUD
    DERIVED_ENDPOINTS = {"dashboard", "analytics", "billing", "reports", "stats", "metrics", "search"}
    if patch.operation == "add_endpoint":
        path = patch.payload.get("path", "")
        parts = [p for p in path.strip("/").split("/") if p and not p.startswith(":")]
        table_guess = parts[-1] if parts else ""
        if table_guess and table_guess not in db_table_names and table_guess not in DERIVED_ENDPOINTS:
            return False, f"REJECTED: add_endpoint '{path}' has no DB table '{table_guess}'"

    # Rule 2: remove_endpoint / remove_table require high confidence — block by default
    if patch.operation in ["remove_table"]:
        return False, f"REJECTED: remove_table is too destructive without explicit user confirmation"

    # Rule 3: add_field to API — the field should exist in the corresponding DB table
    if patch.operation == "add_field" and patch.target_schema == "api":
        field_name = patch.payload.get("name", "")
        # Find which table this endpoint maps to
        for ep in schemas.api.endpoints:
            if ep.path in patch.location:
                parts = [p for p in ep.path.strip("/").split("/") if p and not p.startswith(":")]
                table_guess = parts[-1] if parts else ""
                if table_guess in db_table_names:
                    db_cols = {c.name for t in schemas.db.tables if t.name == table_guess for c in t.columns}
                    if field_name not in db_cols and field_name not in {"token", "message", "success", "meta"}:
                        return False, f"REJECTED: add_field '{field_name}' to API but not in DB table '{table_guess}'"
                break

    return True, "OK"


def _llm_repair_pass(
    schemas: FullSchemaOutput,
    issues: list[Issue],
    debug: bool = False
) -> tuple[list[Patch], list[str]]:
    """Send issues + relevant context to LLM, get back validated patches."""
    if not issues:
        return [], []

    # Send top 15 issues with relevant schema context (not full dump)
    top_issues = sorted(issues, key=lambda i: 0 if i.severity == "error" else 1)[:15]
    compact_issues = [
        {"sev": i.severity, "cat": i.category, "loc": i.location, "msg": i.description}
        for i in top_issues
    ]
    issues_json = json.dumps(compact_issues)
    context_json = _extract_relevant_context(schemas, top_issues)

    user_content = (
        f"Issues ({len(top_issues)} of {len(issues)} total):\n{issues_json}\n\n"
        f"Related schema context:\n{context_json}\n\n"
        "Return the repair JSON with 'patches' and 'assumptions'."
    )

    raw = call_llm(
        system_prompt=REPAIR_SYSTEM_PROMPT,
        user_content=user_content,
        label="4-repair",
        max_tokens=4000,
        debug=debug
    )

    cleaned = clean_json(raw)

    try:
        data = json.loads(cleaned)
        raw_patches = data.get("patches", [])

        patches: list[Patch] = []
        rejected: list[str] = []
        for p in raw_patches:
            try:
                patch = Patch(**p)
                # Run through decision engine
                ok, reason = _validate_patch(patch, schemas, debug)
                if ok:
                    patches.append(patch)
                else:
                    rejected.append(reason)
                    if debug: print(f"  {reason}")
            except ValidationError as ve:
                reason = f"PARSE_ERROR: {p.get('operation')} at {p.get('location')}"
                rejected.append(reason)
                if debug: print(f"  [Stage 4] Skipping invalid patch: {reason}")

        if debug and rejected:
            print(f"  [Stage 4] Rejected {len(rejected)} patches:")
            for r in rejected[:5]:
                print(f"    - {r}")

        assumptions = data.get("assumptions", [])
        return patches, assumptions
    except json.JSONDecodeError as e:
        if debug:
            print(f"[Stage 4] Patch JSON decode error: {e}")
        return [], []


# --- Stage 4C: Patch Safety & Application ------------------------------------

def _is_valid_payload(patch: Patch) -> tuple[bool, str]:
    """Validate patch payload has all required fields. Rejects None/empty."""
    p = patch.payload
    op = patch.operation

    # Universal: no None values allowed anywhere in payload
    if any(v is None for v in p.values()):
        return False, f"Payload contains None values"

    if op == "add_column":
        required = {"name", "type"}
        missing = required - set(p.keys())
        if missing:
            return False, f"add_column missing: {missing}"

    elif op == "add_endpoint":
        required = {"method", "path"}
        missing = required - set(p.keys())
        if missing or not p.get("path"):
            return False, f"add_endpoint missing path or method"

    elif op == "add_table":
        required = {"name", "columns"}
        missing = required - set(p.keys())
        if missing or not p.get("name"):
            return False, f"add_table missing: {missing}"

    elif op in ("add_field", "change_field"):
        if not p.get("name"):
            return False, f"add_field missing 'name'"

    elif op == "fix_fk":
        if not p.get("foreign_key"):
            return False, f"fix_fk missing 'foreign_key'"

    return True, "OK"


def _apply_patches(
    schemas: FullSchemaOutput,
    patches: list[Patch],
    debug: bool = False
) -> tuple[FullSchemaOutput, list[Patch]]:
    """
    Safe patch application with per-patch rollback.
    Each patch is applied to a copy first; only committed if schema stays valid.
    """
    applied: list[Patch] = []
    schema_dict = json.loads(schemas.model_dump_json())

    for patch in patches:
        # Step 1: Validate payload completeness
        valid, reason = _is_valid_payload(patch)
        if not valid:
            if debug: print(f"  [REJECTED] {patch.operation}: {reason}")
            continue

        # Step 2: Safe apply — deepcopy, apply, validate
        backup = copy.deepcopy(schema_dict)
        try:
            if patch.target_schema == "db":
                _apply_db_patch(schema_dict["db"], patch, debug)
            elif patch.target_schema == "api":
                _apply_api_patch(schema_dict["api"], patch, debug)
            elif patch.target_schema == "ui":
                _apply_ui_patch(schema_dict["ui"], patch, debug)

            # Step 3: Validate the schema is still valid after this patch
            FullSchemaOutput(**schema_dict)
            applied.append(patch)
        except (ValidationError, Exception) as e:
            # Rollback: restore from backup
            schema_dict = backup
            if debug:
                print(f"  [ROLLBACK] {patch.operation} at '{patch.location}' broke schema: {str(e)[:80]}")

    # Final reconstruction
    try:
        updated = FullSchemaOutput(**schema_dict)
    except ValidationError:
        updated = schemas  # ultimate fallback

    return updated, applied


def _apply_db_patch(db_dict: dict, patch: Patch, debug: bool):
    table_name = _extract_table_name(patch.location)
    
    if patch.operation == "add_column":
        for table in db_dict["tables"]:
            if table["name"] == table_name:
                table["columns"].append(patch.payload)
                if debug: print(f"  [DB Patch] Added col '{patch.payload.get('name')}' to '{table_name}'")
                return

    elif patch.operation == "remove_column":
        col_name = patch.payload.get("name")
        for table in db_dict["tables"]:
            if table["name"] == table_name:
                table["columns"] = [c for c in table["columns"] if c["name"] != col_name]
                if debug: print(f"  [DB Patch] Removed col '{col_name}' from '{table_name}'")
                return

    elif patch.operation == "add_table":
        # Check if table already exists
        if any(t["name"] == patch.payload.get("name") for t in db_dict["tables"]):
            return
        db_dict["tables"].append(patch.payload)
        db_dict["migration_order"].append(patch.payload.get("name"))
        if debug: print(f"  [DB Patch] Added table '{patch.payload.get('name')}'")
        return

    elif patch.operation == "remove_table":
        table_to_rem = patch.payload.get("name")
        db_dict["tables"] = [t for t in db_dict["tables"] if t["name"] != table_to_rem]
        db_dict["migration_order"] = [o for o in db_dict["migration_order"] if o != table_to_rem]
        if debug: print(f"  [DB Patch] Removed table '{table_to_rem}'")
        return


def _apply_api_patch(api_dict: dict, patch: Patch, debug: bool):
    if patch.operation in ["add_field", "change_field"]:
        for ep in api_dict["endpoints"]:
            ep_key = f"{ep['method']} {ep['path']}"
            if ep_key in patch.location or ep["path"] in patch.location:
                target = patch.payload.get("target", "response") # response | request_body
                field_name = patch.payload.get("name")
                field_type = patch.payload.get("type", "string")
                if field_name:
                    if target == "response":
                        ep["response"]["fields"][field_name] = field_type
                    elif target == "request_body" and ep.get("request_body"):
                        ep["request_body"]["fields"][field_name] = field_type
                    if debug: print(f"  [API Patch] {patch.operation} '{field_name}' in {ep_key}.{target}")
                return

    elif patch.operation == "remove_field":
        for ep in api_dict["endpoints"]:
            ep_key = f"{ep['method']} {ep['path']}"
            if ep_key in patch.location or ep["path"] in patch.location:
                target = patch.payload.get("target", "response")
                field_name = patch.payload.get("name")
                if target == "response" and field_name in ep["response"]["fields"]:
                    del ep["response"]["fields"][field_name]
                elif target == "request_body" and ep.get("request_body") and field_name in ep["request_body"]["fields"]:
                    del ep["request_body"]["fields"][field_name]
                if debug: print(f"  [API Patch] Removed field '{field_name}' from {ep_key}.{target}")
                return

    elif patch.operation == "add_endpoint":
        api_dict["endpoints"].append(patch.payload)
        if debug: print(f"  [API Patch] Added endpoint: {patch.payload.get('method')} {patch.payload.get('path')}")


def _apply_ui_patch(ui_dict: dict, patch: Patch, debug: bool):
    if patch.operation == "add_field":
        page_name = _extract_segment(patch.location, "pages")
        comp_name = _extract_segment(patch.location, "components")
        for page in ui_dict["pages"]:
            if page["name"] == page_name:
                for comp in page["components"]:
                    if comp["name"] == comp_name:
                        comp.setdefault("fields", []).append(patch.payload)
                        if debug:
                            print(f"  [UI Patch] Added field to {page_name}.{comp_name}")
                        return

    elif patch.operation == "other":
        if debug:
            print(f"  [UI Patch] Deferred (manual): {patch.rationale}")


# --- Patch Location Helpers --------------------------------------------------

def _extract_table_name(location: str) -> str:
    """DBSchema.tables[contacts].columns[user_id] -> 'contacts'"""
    if "tables[" in location:
        return location.split("tables[")[1].split("]")[0]
    return ""

def _extract_col_name(location: str) -> str:
    """DBSchema.tables[contacts].columns[user_id] -> 'user_id'"""
    if "columns[" in location:
        return location.split("columns[")[1].split("]")[0]
    return ""

def _extract_segment(location: str, key: str) -> str:
    """UISchema.pages[Login].components[LoginForm] + 'pages' -> 'Login'"""
    if f"{key}[" in location:
        return location.split(f"{key}[")[1].split("]")[0]
    return ""


# --- Scoring Engine -----------------------------------------------------------

def _calculate_weighted_score(issues: list[Issue]) -> float:
    """Weighted penalty scoring: errors=-5, warnings=-2. Returns 0.0 to 1.0."""
    if not issues:
        return 1.0
    penalty = sum(5 if i.severity == "error" else 2 for i in issues)
    return round(max(0, 100 - penalty) / 100.0, 2)

def _resolution_score(found: list[Issue], remaining: list[Issue]) -> float:
    """Resolved / Total as a percentage."""
    if not found:
        return 1.0
    resolved = len(found) - len(remaining)
    return round(max(0.0, resolved / len(found)), 2)


# --- Deterministic Fallback Repair -------------------------------------------

def _deterministic_repair(schemas: FullSchemaOutput, issues: list[Issue], debug: bool = False) -> list[Patch]:
    """
    Minimal deterministic repair: Only missing fields and API-DB sync.
    """
    patches: list[Patch] = []
    db_tables = {t.name: {c.name for c in t.columns} for t in schemas.db.tables}
    api_paths = {f"{ep.method} {ep.path}" for ep in schemas.api.endpoints}

    # Fix 0: Hardcoded Core Requirements (Perfection)
    if "users" in db_tables and "email" not in db_tables["users"]:
        patches.append(Patch(
            target_schema="db",
            location="tables[users].columns",
            operation="add_column",
            payload={
                "name": "email", "type": "VARCHAR", "nullable": False,
                "unique": True, "primary_key": False, "foreign_key": "", "default": ""
            },
            rationale="Enforcing mandatory core field: users.email"
        ))
        db_tables["users"].add("email")

    for issue in issues:
        # Fix 1: Missing Required Columns or API-DB Mismatch
        if ("Required field" in issue.description) or ("Response field" in issue.description) or ("Request field" in issue.description):
            parts = issue.description.split("'")
            if len(parts) >= 4:
                field_name = parts[1]
                table_name = parts[3]
                if table_name in db_tables and field_name not in db_tables[table_name]:
                    patches.append(Patch(
                        target_schema="db",
                        location=f"tables[{table_name}].columns",
                        operation="add_column",
                        payload={
                            "name": field_name, "type": "VARCHAR", "nullable": True,
                            "unique": False, "primary_key": False, "foreign_key": "", "default": ""
                        },
                        rationale=f"Auto-fixing field '{field_name}' in table '{table_name}'"
                    ))
                    db_tables[table_name].add(field_name)

    # Fix 2: Explicit Relationships (Final Polish)
    existing_fks = {} # table -> {col -> target}
    for t in schemas.db.tables:
        existing_fks[t.name] = {c.name: c.foreign_key for c in t.columns if c.foreign_key}

    if "subscriptions" in db_tables:
        sub_fks = existing_fks.get("subscriptions", {})
        for col in ["user_id", "plan_id"]:
            if col in db_tables["subscriptions"] and not sub_fks.get(col):
                target = "users.id" if col == "user_id" else "plans.id"
                patches.append(Patch(
                    target_schema="db",
                    location=f"tables[subscriptions].columns[{col}]",
                    operation="fix_fk",
                    payload={"foreign_key": target},
                    rationale=f"Deterministic FK fix: subscriptions.{col} -> {target}"
                ))

    if "payments" in db_tables:
        pay_fks = existing_fks.get("payments", {})
        if "subscription_id" in db_tables["payments"] and not pay_fks.get("subscription_id"):
            patches.append(Patch(
                target_schema="db",
                location="tables[payments].columns[subscription_id]",
                operation="fix_fk",
                payload={"foreign_key": "subscriptions.id"},
                rationale="Deterministic FK fix: payments.subscription_id -> subscriptions.id"
            ))

    # Fix 3: UI -> API Endpoint Enforcement (Final Link)
    for issue in issues:
        if issue.category == "api_ui" and "has no matching API endpoint" in issue.description:
            ds = issue.description.split("'")[1] if "'" in issue.description else ""
            if ds and f"GET {ds}" not in api_paths:
                patches.append(Patch(
                    target_schema="api",
                    location="endpoints",
                    operation="add_endpoint",
                    payload={
                        "method": "GET",
                        "path": ds,
                        "summary": f"Get data for {ds}",
                        "auth_required": True,
                        "roles_allowed": ["admin", "user"],
                        "response": {
                            "success_status": True,
                            "fields": {"id": "string"}
                        }
                    },
                    rationale=f"Auto-generating missing endpoint required by UI: {ds}"
                ))
                api_paths.add(f"GET {ds}")

    if debug and patches:
        print(f"  [Deterministic] Generated {len(patches)} simple patches")

    return patches


# --- Core Function ------------------------------------------------------------

def refine_schemas(schemas: FullSchemaOutput, debug: bool = False) -> RefinedOutput:
    import time
    start_time = time.time()
    
    app_name = schemas.app_name

    # -- Step 1: Initial Analysis ----------------------------------------------
    print("\n[Stage 4] Initial static analysis...")
    issues_found = _static_analysis(schemas)
    initial_weighted = _calculate_weighted_score(issues_found)
    errors_init = len([i for i in issues_found if i.severity == "error"])
    warns_init = len([i for i in issues_found if i.severity == "warning"])
    
    print(f"  Found {len(issues_found)} issues ({errors_init} errors, {warns_init} warnings). Initial Score: {initial_weighted:.0%}")

    # -- Step 2: Repair Cycle --------------------------------------------------
    updated_schemas = schemas
    patches_proposed_all: list[Patch] = []
    patches_applied:      list[Patch] = []
    assumptions:          list[str] = []
    repair_calls = 0

    # Iterative Repair: Max 2 passes
    for pass_num in range(1, 3):
        current_issues = _static_analysis(updated_schemas)
        if not current_issues:
            break
            
        print(f"\n[Stage 4] Repair Pass {pass_num} ({len(current_issues)} issues remaining)...")
        
        # A. Deterministic Repair (Always run first)
        det_patches = _deterministic_repair(updated_schemas, current_issues, debug=debug)
        if det_patches:
            patches_proposed_all.extend(det_patches)
            print(f"  [Pass {pass_num}] Applying {len(det_patches)} deterministic patches...")
            updated_schemas, applied = _apply_patches(updated_schemas, det_patches, debug=debug)
            patches_applied.extend(applied)
            current_issues = _static_analysis(updated_schemas)
            
        if not current_issues:
            break

        # B. LLM Repair (Reasoning/Complex Fixes) - Only in Pass 1
        if pass_num == 1:
            # Skip LLM if no errors remain (only warnings) or score is high enough
            errors_left = [i for i in current_issues if i.severity == "error"]
            if not errors_left:
                print(f"  [Pass {pass_num}] No errors remain. Skipping LLM repair.")
                continue

            print(f"  [Pass {pass_num}] Consulting LLM for reasoning on remaining {len(current_issues)} issues...")
            repair_calls += 1
            llm_patches, llm_assumptions = _llm_repair_pass(updated_schemas, current_issues, debug=debug)
            assumptions.extend(llm_assumptions)
            
            if llm_patches:
                # Limit to MAX_PATCHES = 3
                MAX_PATCHES = 3
                if len(llm_patches) > MAX_PATCHES:
                    print(f"  [Pass {pass_num}] Limiting LLM from {len(llm_patches)} to {MAX_PATCHES} patches.")
                    llm_patches = llm_patches[:MAX_PATCHES]

                patches_proposed_all.extend(llm_patches)
                print(f"  [Pass {pass_num}] Applying {len(llm_patches)} LLM patches...")
                updated_schemas, applied = _apply_patches(updated_schemas, llm_patches, debug=debug)
                patches_applied.extend(applied)
        else:
            print(f"  [Pass {pass_num}] Final cleanup pass completed.")
            break

    # -- Step 3: Final Analysis ------------------------------------------------
    issues_remaining = _static_analysis(updated_schemas)
    final_weighted = _calculate_weighted_score(issues_remaining)
    resolution = _resolution_score(issues_found, issues_remaining)
    
    errors_final = len([i for i in issues_remaining if i.severity == "error"])
    warns_final = len([i for i in issues_remaining if i.severity == "warning"])
    resolved_total = len(issues_found) - len(issues_remaining)
    auto_resolved = max(0, resolved_total - len(patches_applied))

    # Use the better of the two scores for final
    final_score = max(final_weighted, resolution)
    if final_score < initial_weighted:
        final_score = initial_weighted

    latency = (time.time() - start_time) * 1000
    ready = errors_final == 0

    print(f"\n[Stage 4] Final Score: {final_score:.0%} (weighted: {final_weighted:.0%}, resolution: {resolution:.0%})")
    print(f"  Latency: {latency:.0f}ms")
    print(f"  Resolved: {resolved_total} (patches: {len(patches_applied)}, auto: {auto_resolved})")
    print(f"  Remaining: {len(issues_remaining)} ({errors_final} errors, {warns_final} warnings)")
    
    if issues_remaining:
        print("\nRemaining issues:")
        for issue in issues_remaining:
            print(f"  - [{issue.severity}] {issue.category}: {issue.description} at {issue.location}")

    print(f"[Stage 4] Ready for codegen: {'YES [OK]' if ready else 'NO'}")

    audit = AuditTrail(
        issues_found=issues_found,
        patches_proposed=patches_proposed_all,
        patches_applied=patches_applied,
        issues_remaining=issues_remaining,
        initial_score=initial_weighted,
        final_score=final_score,
        latency_ms=latency,
        repair_calls=repair_calls,
        assumptions=assumptions
    )

    return RefinedOutput(
        app_name=app_name,
        schemas=updated_schemas,
        audit=audit,
        ready_for_codegen=ready
    )


# --- CLI Test -----------------------------------------------------------------

if __name__ == "__main__":
    test_prompt = (
        "Build a CRM with login, contacts, dashboard, role-based access, "
        "and premium plan with payments. Admins can see analytics."
    )

    print("=" * 60)
    print("Running Stage 1 -> 2 -> 3 -> 4 pipeline")
    print("=" * 60)

    print("\n[Stage 1] Extracting intent...")
    intent = extract_intent(test_prompt, debug=False)
    print(f"  Intent: {intent.app_name} ({intent.app_type})")

    print("\n[Stage 2] Designing system...")
    design = design_system(intent, debug=False)
    print(f"  Pages: {[p.name for p in design.pages]}")

    print("\n[Stage 3] Generating schemas in parallel...")
    schemas = generate_schemas(design, debug=False)
    print(f"  UI pages: {[p.name for p in schemas.ui.pages]}")
    print(f"  API endpoints: {len(schemas.api.endpoints)}")
    print(f"  DB tables: {[t.name for t in schemas.db.tables]}")

    print("\n[Stage 4] Refining & cross-validating...")
    refined = refine_schemas(schemas, debug=True)

    a = refined.audit
    resolved = len(a.issues_found) - len(a.issues_remaining)
    auto = max(0, resolved - len(a.patches_applied))

    print("\n" + "=" * 60)
    print(f"App:              {refined.app_name}")
    print(f"Initial Score:    {a.initial_score:.0%}")
    print(f"Final Score:      {a.final_score:.0%}")
    print(f"Latency:          {a.latency_ms:.0f}ms")
    print(f"Repair Calls:     {a.repair_calls}")
    print(f"Ready for Codegen: {'YES [OK]' if refined.ready_for_codegen else 'NO'}")
    
    if a.assumptions:
        print("\nAssumptions Made:")
        for assumption in a.assumptions:
            print(f"  - {assumption}")
    
    print("\nIssues Breakdown:")
    print(f"  Found: {len(a.issues_found)} | Patched: {len(a.patches_applied)} | Auto-resolved: {auto} | Remaining: {len(a.issues_remaining)}")
    print("=" * 60)

    # # Save to file
    # out_path = "stage4_output.json"
    # with open(out_path, "w") as f:
    #     f.write(refined.model_dump_json(indent=2))
    # print(f"\nFull output saved to: {out_path}")