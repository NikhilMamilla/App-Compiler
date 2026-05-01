"""
api/main.py — FastAPI Server

Single POST /generate endpoint that runs the full pipeline:
  Stage 1 → Stage 2 → Stage 3 → Stage 4 → Validate → (Repair if needed)

Also exposes:
  GET  /health         → liveness check
  GET  /pipeline-info  → describes each stage
  POST /validate-only  → runs stages 1-3 + validator, skips Stage 4
"""

import time
import sys
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.stage1_intent import extract_intent
from pipeline.stage2_design import design_system
from pipeline.stage3_schema import generate_schemas
from pipeline.stage4_refine import refine_schemas
from pipeline.validator import validate_schemas
from pipeline.repair import repair_schemas

# ─── App Setup ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="App Compiler API",
    description="Natural language → validated app schema (UI + API + DB + Auth)",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Request / Response Models ────────────────────────────────────────────────

class GenerateRequest(BaseModel):
    prompt: str = Field(
        ...,
        min_length=10,
        max_length=2000,
        description="Natural language description of the app to build",
        example="Build a CRM with login, contacts, dashboard, role-based access, and premium plan with payments."
    )
    debug: bool = Field(
        default=False,
        description="If true, includes raw intermediate outputs in the response"
    )

class StageTimings(BaseModel):
    stage1_ms: float
    stage2_ms: float
    stage3_ms: float
    stage4_ms: float
    total_ms: float

class PipelineMetrics(BaseModel):
    initial_score: float
    final_score: float
    issues_found: int
    issues_resolved: int
    issues_remaining: int
    repair_calls: int
    ready_for_codegen: bool
    timings: StageTimings

class GenerateResponse(BaseModel):
    success: bool
    app_name: str
    prompt: str
    schemas: dict                   # full FullSchemaOutput as dict
    metrics: PipelineMetrics
    assumptions: list[str]
    errors: list[str]               # any remaining errors after repair
    warnings: list[str]

class ValidateOnlyResponse(BaseModel):
    success: bool
    app_name: str
    passed: bool
    score: float
    errors: list[str]
    warnings: list[str]
    schemas: dict

# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "version": "1.0.0"}


@app.get("/pipeline-info")
def pipeline_info():
    return {
        "stages": [
            {
                "stage": 1,
                "name": "Intent Extraction",
                "description": "Parses user prompt into structured intent (entities, roles, features, auth, payments)",
                "output": "IntentSchema"
            },
            {
                "stage": 2,
                "name": "System Design",
                "description": "Converts intent into app architecture (pages, API groups, DB entities, permissions)",
                "output": "SystemDesignSchema"
            },
            {
                "stage": 3,
                "name": "Schema Generation",
                "description": "Generates UI + API + DB schemas in parallel using 3 concurrent LLM calls",
                "output": "FullSchemaOutput"
            },
            {
                "stage": 4,
                "name": "Refinement & Repair",
                "description": "Cross-validates all 3 schemas, detects inconsistencies, applies deterministic + LLM patches",
                "output": "RefinedOutput"
            }
        ],
        "validation": "7 cross-layer checks: UI↔API, API↔DB, FK integrity, auth rules, migration order",
        "repair": "Surgical: only broken layers are repaired, max 2 rounds, deterministic-first then LLM"
    }


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """
    Full pipeline: prompt → validated, executable app schema.
    """
    total_start = time.time()

    try:
        # ── Stage 1 ──────────────────────────────────────────────────
        t1 = time.time()
        intent = extract_intent(request.prompt, debug=request.debug)
        stage1_ms = (time.time() - t1) * 1000

        # ── Stage 2 ──────────────────────────────────────────────────
        t2 = time.time()
        design = design_system(intent, debug=request.debug)
        stage2_ms = (time.time() - t2) * 1000

        # ── Stage 3 ──────────────────────────────────────────────────
        t3 = time.time()
        schemas = generate_schemas(design, debug=request.debug)
        stage3_ms = (time.time() - t3) * 1000

        # ── Stage 4 ──────────────────────────────────────────────────
        t4 = time.time()
        refined = refine_schemas(schemas, debug=request.debug)
        stage4_ms = (time.time() - t4) * 1000

        total_ms = (time.time() - total_start) * 1000
        audit = refined.audit

        # ── Build metrics ─────────────────────────────────────────────
        issues_resolved = len(audit.issues_found) - len(audit.issues_remaining)
        metrics = PipelineMetrics(
            initial_score=round(audit.initial_score * 100, 1),
            final_score=round(audit.final_score * 100, 1),
            issues_found=len(audit.issues_found),
            issues_resolved=issues_resolved,
            issues_remaining=len(audit.issues_remaining),
            repair_calls=audit.repair_calls,
            ready_for_codegen=refined.ready_for_codegen,
            timings=StageTimings(
                stage1_ms=round(stage1_ms, 1),
                stage2_ms=round(stage2_ms, 1),
                stage3_ms=round(stage3_ms, 1),
                stage4_ms=round(stage4_ms, 1),
                total_ms=round(total_ms, 1)
            )
        )

        remaining_errors   = [f"[{i.category}] {i.description}" for i in audit.issues_remaining if i.severity == "error"]
        remaining_warnings = [f"[{i.category}] {i.description}" for i in audit.issues_remaining if i.severity == "warning"]

        return GenerateResponse(
            success=True,
            app_name=refined.app_name,
            prompt=request.prompt,
            schemas=refined.schemas.model_dump(),
            metrics=metrics,
            assumptions=audit.assumptions,
            errors=remaining_errors,
            warnings=remaining_warnings
        )

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")


@app.post("/validate-only", response_model=ValidateOnlyResponse)
async def validate_only(request: GenerateRequest):
    """
    Runs stages 1-3 + validator only. Skips Stage 4 repair.
    Useful for quick checks and eval baseline comparisons.
    """
    try:
        intent  = extract_intent(request.prompt, debug=False)
        design  = design_system(intent, debug=False)
        schemas = generate_schemas(design, debug=False)
        report  = validate_schemas(schemas)

        return ValidateOnlyResponse(
            success=True,
            app_name=intent.app_name,
            passed=report.passed,
            score=round(report.score * 100, 1),
            errors=[f"[{e.code}] {e.message}" for e in report.errors],
            warnings=[f"[{w.code}] {w.message}" for w in report.warnings],
            schemas=schemas.model_dump()
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))