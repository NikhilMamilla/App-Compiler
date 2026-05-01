"""
repair.py — Public repair interface

Stage 4 already contains the full repair engine:
  - _deterministic_repair()  → rule-based fast patches
  - _llm_repair_pass()       → LLM surgical repair
  - _apply_patches()         → safe patch application with rollback

This module is a clean public wrapper so api/main.py and
eval/run_evals.py can trigger repairs without importing
Stage 4 internals directly.
"""

from __future__ import annotations
import sys, os
from pydantic import BaseModel

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline.stage3_schema import FullSchemaOutput
from pipeline.stage4_refine import (
    _static_analysis,
    _deterministic_repair,
    _llm_repair_pass,
    _apply_patches,
    _calculate_weighted_score,
    Patch,
)
from pipeline.validator import ValidationReport, validate_schemas


class RepairResult(BaseModel):
    schemas: FullSchemaOutput
    report: ValidationReport
    rounds: int
    repairs_made: list[str]

    class Config:
        arbitrary_types_allowed = True

    def summary(self) -> str:
        return (
            f"Rounds: {self.rounds} | "
            f"Repairs: {len(self.repairs_made)} | "
            f"{self.report.summary()}"
        )


def repair_schemas(
    schemas: FullSchemaOutput,
    report: ValidationReport,
    debug: bool = False,
    max_rounds: int = 2,
) -> RepairResult:
    """
    Surgically repairs schemas based on ValidationReport.
    Delegates to Stage 4's deterministic + LLM repair engines.
    Only touches layers that have errors — never rewrites the whole schema.
    """
    repairs_made: list[str] = []
    current = schemas
    rounds  = 0

    for round_num in range(1, max_rounds + 1):
        issues = _static_analysis(current)
        if not issues:
            break

        rounds = round_num
        if debug:
            print(f"\n[Repair Round {round_num}] {len(issues)} issues")

        # 1 — Deterministic patches first (fast, no LLM cost)
        det_patches = _deterministic_repair(current, issues, debug=debug)
        if det_patches:
            current, applied = _apply_patches(current, det_patches, debug=debug)
            repairs_made.append(
                f"Round {round_num}: {len(applied)} deterministic patches applied"
            )

        # Re-check after deterministic
        issues = _static_analysis(current)
        if not issues:
            break

        # 2 — LLM repair for remaining complex issues (round 1 only)
        if round_num == 1:
            llm_patches, assumptions = _llm_repair_pass(current, issues, debug=debug)
            if llm_patches:
                current, applied = _apply_patches(current, llm_patches, debug=debug)
                repairs_made.append(
                    f"Round {round_num}: {len(applied)} LLM patches applied"
                )
                if assumptions:
                    repairs_made.append(f"Assumptions: {'; '.join(assumptions)}")

    final_report = validate_schemas(current)
    return RepairResult(
        schemas=current,
        report=final_report,
        rounds=rounds,
        repairs_made=repairs_made,
    )